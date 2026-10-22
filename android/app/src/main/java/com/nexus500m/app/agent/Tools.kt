package com.nexus500m.app.agent

import com.nexus500m.app.data.MemoryStore
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import javax.net.ssl.HttpsURLConnection

/**
 * The local tool set for the permission-controlled autonomous task engine.
 *
 * Safety rules implemented here:
 * - The calculator is a real shunting-yard parser. NO eval() of any kind —
 *   model-generated text is NEVER executed as code.
 * - Notes/reports are written only inside the app's own workspace
 *   (filesDir/notes, filesDir/reports).
 * - Web research is consent-gated, uses only the Wikipedia REST API, and the
 *   retrieved text is treated strictly as DATA (never as instructions).
 * - No tool can touch files outside the app sandbox, and destructive tools
 *   (delete) are always routed through user confirmation by the TaskEngine.
 */
object Tools {

    /** Result of one tool invocation. */
    data class ToolResult(
        val success: Boolean,
        val output: String,
        val risky: Boolean = false,     // did this perform a lasting change?
    )

    // ------------------------------------------------ calculator
    /**
     * Safe arithmetic evaluator (+ - * / % parentheses, integers & decimals).
     * Classic shunting-yard -> RPN evaluation. Throws on anything unexpected.
     */
    fun calculate(expression: String): ToolResult {
        return try {
            val value = evaluateExpression(expression)
            ToolResult(true, "$expression = $value")
        } catch (t: Throwable) {
            ToolResult(false, "could not evaluate '$expression': ${t.message}")
        }
    }

    internal fun evaluateExpression(src: String): Double {
        val s = src.replace("×", "*").replace("÷", "/").replace(",", "")
        val tokens = tokenizeExpr(s)
        val rpn = toRpn(tokens)
        return evalRpn(rpn)
    }

    private enum class Tok { NUM, OP, LPAREN, RPAREN }

    private fun tokenizeExpr(s: String): List<Pair<Tok, String>> {
        val out = ArrayList<Pair<Tok, String>>()
        var i = 0
        while (i < s.length) {
            val c = s[i]
            when {
                c.isWhitespace() -> i++
                c.isDigit() || c == '.' -> {
                    val sb = StringBuilder()
                    while (i < s.length && (s[i].isDigit() || s[i] == '.')) sb.append(s[i++])
                    val numStr = sb.toString()
                    if (numStr.count { it == '.' } > 1) throw IllegalArgumentException("bad number $numStr")
                    out.add(Tok.NUM to numStr)
                }
                c in "+-*/%()" -> {
                    val tok = when (c) {
                        '(' -> Tok.LPAREN
                        ')' -> Tok.RPAREN
                        else -> Tok.OP
                    }
                    out.add(tok to c.toString())
                    i++
                }
                else -> throw IllegalArgumentException("character not allowed: '$c'")
            }
        }
        if (out.isEmpty()) throw IllegalArgumentException("empty expression")
        return out
    }

    private fun prec(op: String) = when (op) {
        "+", "-" -> 1
        "*", "/", "%" -> 2
        else -> 0
    }

    private fun toRpn(tokens: List<Pair<Tok, String>>): List<String> {
        val out = ArrayList<String>()
        val ops = ArrayList<String>()
        var prevWasValue = false
        for ((tok, text) in tokens) {
            when (tok) {
                Tok.NUM -> {
                    out.add(text)
                    prevWasValue = true
                }
                Tok.OP -> {
                    // unary minus -> 0 - x
                    if (!prevWasValue && text == "-") out.add("0")
                    while (ops.isNotEmpty() && prec(ops.last()) >= prec(text)) out.add(ops.removeAt(ops.size - 1))
                    ops.add(text)
                    prevWasValue = false
                }
                Tok.LPAREN -> {
                    ops.add("(")
                    prevWasValue = false
                }
                Tok.RPAREN -> {
                    while (ops.isNotEmpty() && ops.last() != "(") out.add(ops.removeAt(ops.size - 1))
                    if (ops.isEmpty()) throw IllegalArgumentException("unbalanced parentheses")
                    ops.removeAt(ops.size - 1)
                    prevWasValue = true
                }
            }
        }
        while (ops.isNotEmpty()) {
            val op = ops.removeAt(ops.size - 1)
            if (op == "(") throw IllegalArgumentException("unbalanced parentheses")
            out.add(op)
        }
        return out
    }

    private fun evalRpn(rpn: List<String>): Double {
        val st = ArrayList<Double>()
        for (t in rpn) {
            when {
                t == "+" -> { val b = st.removeAt(st.size - 1); val a = st.removeAt(st.size - 1); st.add(a + b) }
                t == "-" -> { val b = st.removeAt(st.size - 1); val a = st.removeAt(st.size - 1); st.add(a - b) }
                t == "*" -> { val b = st.removeAt(st.size - 1); val a = st.removeAt(st.size - 1); st.add(a * b) }
                t == "/" -> { val b = st.removeAt(st.size - 1); val a = st.removeAt(st.size - 1); st.add(a / b) }
                t == "%" -> { val b = st.removeAt(st.size - 1); val a = st.removeAt(st.size - 1); st.add(a % b) }
                else -> st.add(t.toDouble())
            }
        }
        if (st.size != 1) throw IllegalArgumentException("malformed expression")
        val result = st[0]
        if (result.isNaN() || result.isInfinite()) throw IllegalArgumentException("result is not finite")
        return result
    }

    // ------------------------------------------------ notes (app workspace only)
    fun notesDir(context: android.content.Context): File =
        File(context.filesDir, "notes").apply { mkdirs() }

    fun createNote(context: android.content.Context, title: String, body: String): ToolResult {
        val safe = sanitizeFilename(title.ifBlank { "note" })
        val f = File(notesDir(context), "$safe.md")
        if (f.exists()) {
            return ToolResult(false, "a note called '$safe' already exists (refusing to overwrite)", risky = true)
        }
        f.writeText("# $title\n\n$body\n", Charsets.UTF_8)
        return ToolResult(true, "created note: notes/$safe.md", risky = true)
    }

    fun appendNote(context: android.content.Context, title: String, body: String): ToolResult {
        val safe = sanitizeFilename(title)
        val f = File(notesDir(context), "$safe.md")
        if (!f.exists()) return ToolResult(false, "no note named '$safe'")
        f.appendText("\n$body\n", Charsets.UTF_8)
        return ToolResult(true, "appended to notes/$safe.md", risky = true)
    }

    fun listNotes(context: android.content.Context): ToolResult {
        val files = notesDir(context).listFiles()?.sortedBy { it.name } ?: emptyList()
        return ToolResult(true, if (files.isEmpty()) "no notes yet"
        else files.joinToString("\n") { "- ${it.name} (${it.length()} bytes)" })
    }

    fun readNote(context: android.content.Context, title: String): ToolResult {
        val safe = sanitizeFilename(title)
        val f = File(notesDir(context), "$safe.md")
        if (!f.exists()) return ToolResult(false, "no note named '$safe'")
        return ToolResult(true, f.readText(Charsets.UTF_8).take(4000))
    }

    /** DANGEROUS: only invoked after user confirmation by the TaskEngine. */
    fun deleteNote(context: android.content.Context, title: String): ToolResult {
        val safe = sanitizeFilename(title)
        val f = File(notesDir(context), "$safe.md")
        if (!f.exists()) return ToolResult(false, "no note named '$safe'")
        if (!f.delete()) return ToolResult(false, "could not delete '$safe'")
        return ToolResult(true, "deleted note '$safe'", risky = true)
    }

    private fun sanitizeFilename(name: String): String =
        name.trim().replace(Regex("[^A-Za-z0-9._ -]"), "_").take(48)

    // ------------------------------------------------ memory search
    fun searchMemory(memoryStore: MemoryStore, query: String): ToolResult {
        val results = memoryStore.retrieve(query, limit = 5)
        return ToolResult(true, if (results.isEmpty()) "no matching memories"
        else results.joinToString("\n") { "- ${it.text}" })
    }

    /** Should this text be refused as a memory? (secrets are never memorized) */
    fun looksSensitive(text: String): Boolean {
        val patterns = listOf(
            Regex("(?i)\\b(password|passwd|pin|otp)\\b\\s*[:=]"),
            Regex("github_pat_[A-Za-z0-9_]{16,}"),
            Regex("gh[pousr]_[A-Za-z0-9]{20,}"),
            Regex("AKIA[0-9A-Z]{16}"),
            Regex("-----BEGIN [A-Z ]*PRIVATE KEY-----"),
            Regex("\\b\\d{13,19}\\b"),   // card-like numbers
        )
        return patterns.any { it.containsMatchIn(text) }
    }

    // ------------------------------------------------ web research (consent-gated)
    /**
     * Optional, explicit-consent research over the public Wikipedia REST API.
     * - Offline: returns a clear offline message, no fake results.
     * - Retrieved text is DATA only: callers must never treat it as instructions.
     */
    fun webResearch(topic: String, timeoutMs: Int = 10_000): ToolResult {
        return try {
            val enc = URLEncoder.encode(topic.trim(), "UTF-8")
            val url = URL("https://en.wikipedia.org/api/rest_v1/page/summary/$enc")
            val conn = url.openConnection() as HttpURLConnection
            conn.connectTimeout = timeoutMs
            conn.readTimeout = timeoutMs
            conn.setRequestProperty("User-Agent", "NEXUS-500M personal assistant (offline-first)")
            try {
                val code = conn.responseCode
                if (code == 404) return ToolResult(false, "Wikipedia has no article called '$topic'")
                if (code != 200) return ToolResult(false, "network returned HTTP $code")
                val body = conn.inputStream.bufferedReader().readText()
                val json = JSONObject(body)
                val title = json.optString("title")
                val extract = json.optString("extract")
                if (extract.isBlank()) return ToolResult(false, "no summary available for '$topic'")
                ToolResult(true, "Wikipedia: $title\n$extract")
            } finally {
                conn.disconnect()
            }
        } catch (t: Throwable) {
            ToolResult(false, "web research unavailable (${t::class.java.simpleName}: ${t.message}). " +
                    "This tool needs internet access and your explicit consent.")
        }
    }

    // ------------------------------------------------ reports
    fun reportsDir(context: android.content.Context): File =
        File(context.filesDir, "reports").apply { mkdirs() }

    fun writeReport(context: android.content.Context, name: String, content: String): ToolResult {
        val safe = sanitizeFilename(name.ifBlank { "report-${System.currentTimeMillis()}" })
        val f = File(reportsDir(context), "$safe.md")
        f.writeText(content, Charsets.UTF_8)
        return ToolResult(true, "report saved: reports/$safe.md (${f.length()} bytes)", risky = true)
    }
}
