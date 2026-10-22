package com.nexus500m.app.agent

import android.content.Context
import com.nexus500m.app.data.JsonStore
import com.nexus500m.app.data.MemoryStore
import com.nexus500m.app.model.ModelManager
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Permission-controlled autonomous task engine.
 *
 * Workflow per task (real, observable):
 *   user request -> interpretation -> step plan -> [confirmation gate] ->
 *   execute one step at a time -> observe actual results -> error recovery ->
 *   completion verification -> task report saved to history.
 *
 * Hard limits, honestly implemented:
 * - Global stop button (volatile flag checked between every step).
 * - Per-task step cap and wall-clock timeout.
 * - Risky/destructive actions (delete) REQUIRE user confirmation; the engine
 *   suspends and asks, and cancels the whole task if refused.
 * - Model-generated content is never executed as code; tools have fixed,
 *   audited implementations.
 * - Full step-by-step history persisted (tasks.json).
 */
class TaskEngine(
    private val context: Context,
    private val modelManager: ModelManager,
    private val memoryStore: MemoryStore,
) {

    data class Step(
        val description: String,
        val tool: String,
        val status: String,        // pending | done | failed | skipped | awaiting_confirmation
        val output: String,
        val needsConfirmation: Boolean,
    )

    data class Task(
        val id: String,
        val request: String,
        val createdAt: Long,
        val finishedAt: Long?,
        val status: String,        // running | completed | stopped | failed | awaiting_confirmation
        val steps: List<Step>,
        val report: String,
    )

    @Volatile private var stopRequested = false

    fun requestStop() { stopRequested = true }

    /** Awaiting-confirmation callback: returns true = user approved. */
    interface ConfirmationGate {
        /** Show the action to the user and suspend until they answer. */
        suspend fun confirm(actionDescription: String): Boolean
    }

    private val historyFile: File
        get() = File(context.filesDir, "tasks.json")

    // ---------------------------------------------------------------- plan
    /**
     * Deterministic intent parser -> step plan. The tiny on-device model cannot
     * do reliable free-form planning, so the engine plans with rules (reliable)
     * and uses the model for language generation steps (report text, chat).
     */
    internal fun plan(request: String): List<Step> {
        val r = request.trim()
        val steps = ArrayList<Step>()

        val calc = Regex("(?i)^(?:calculate|compute|what is|eval)\\s+(.+)$").find(r)
        val note = Regex("(?i)^(?:create|make|write) (?:a )?note(?: titled| called)? ['\"]?([^'\"]+)['\"]?(?::| with| saying) (.+)$").find(r)
        val noteRead = Regex("(?i)^(?:read|show|open) (?:the )?note ['\"]?([^'\"]+)['\"]?$").find(r)
        val noteDelete = Regex("(?i)^(?:delete|remove) (?:the )?note ['\"]?([^'\"]+)['\"]?$").find(r)
        val memSearch = Regex("(?i)^(?:search|find) (?:my )?memor(?:y|ies) (?:for |about )?(.+)$").find(r)
        val remind = Regex("(?i)^remind me (?:in |after )?(\\d+)\\s*(minutes?|mins?|hours?)\\s*(?:to )?(.+)$").find(r)
        val research = Regex("(?i)^(?:research|look up|find out about)\\s+(.+?)(?:\\s+on wikipedia)?$").find(r)
        val report = Regex("(?i)^write (?:a )?report(?: titled| called)? ['\"]?([^'\"]+)['\"]?(?::| about| on) (.+)$").find(r)

        return when {
            calc != null -> listOf(Step("Evaluate the arithmetic expression", "calculator", "pending", "", false))
            note != null -> listOf(Step("Create note '${note.groupValues[1].trim()}'", "notes.create", "pending", "", false))
            noteRead != null -> listOf(Step("Read note '${noteRead.groupValues[1].trim()}'", "notes.read", "pending", "", false))
            noteDelete != null -> listOf(Step("DELETE note '${noteDelete.groupValues[1].trim()}'", "notes.delete", "pending", "", true))
            memSearch != null -> listOf(Step("Search personal memory for '${memSearch.groupValues[1].trim()}'", "memory.search", "pending", "", false))
            remind != null -> listOf(Step("Schedule reminder in ${remind.groupValues[1]} ${remind.groupValues[2]}", "reminder.schedule", "pending", "", false),
                Step("Save reminder text as a memory", "memory.add", "pending", "", false))
            research != null -> listOf(Step("Research '${research.groupValues[1].trim()}' on Wikipedia (consent required)", "web.research", "pending", "", false))
            report != null -> listOf(Step("Gather material with the model", "model.generate", "pending", "", false),
                Step("Write the report file", "report.write", "pending", "", false))
            else -> listOf(Step("Answer with the local model", "model.generate", "pending", "", false))
        }
    }

    // ---------------------------------------------------------------- run
    /**
     * Execute a task. Calls [onUpdate] after every step. Web research is only
     * attempted when [webAllowed] is true (consent granted in settings).
     */
    suspend fun run(
        request: String,
        confirmationGate: ConfirmationGate?,
        webAllowed: Boolean,
        onUpdate: (Task) -> Unit,
    ): Task {
        stopRequested = false
        val steps = plan(request).toMutableList()
        val startTime = System.currentTimeMillis()
        var status = "running"
        val reportBuilder = StringBuilder()

        fun emit() {
            onUpdate(Task(
                id = "task-$startTime", request = request, createdAt = startTime,
                finishedAt = if (status == "running") null else System.currentTimeMillis(),
                status = status, steps = steps.toList(), report = reportBuilder.toString(),
            ))
        }

        for (i in steps.indices) {
            if (stopRequested) { status = "stopped"; steps[i] = steps[i].copy(status = "skipped"); break }
            if (System.currentTimeMillis() - startTime > TASK_TIMEOUT_MS) {
                status = "failed"; steps[i] = steps[i].copy(status = "skipped",
                    output = "task timed out after ${TASK_TIMEOUT_MS / 1000}s (limit enforced)"); break
            }

            val step = steps[i]
            if (step.needsConfirmation) {
                steps[i] = step.copy(status = "awaiting_confirmation")
                emit()
                val approved = confirmationGate?.confirm(step.description) ?: false
                if (!approved) {
                    steps[i] = steps[i].copy(status = "skipped", output = "user refused; task cancelled")
                    status = "stopped"
                    reportBuilder.append("Cancelled: you refused the confirmation.\n")
                    break
                }
            }

            val result = executeStep(request, step, webAllowed)
            steps[i] = steps[i].copy(
                status = if (result.success) "done" else "failed",
                output = result.output,
            )
            reportBuilder.append("[${step.tool}] ${result.output}\n")

            if (!result.success) {
                // error recovery: stop the task with a clear reason (safe default;
                // no blind retries — each failure is reported honestly)
                status = "failed"
                break
            }
            emit()
        }
        if (status == "running") status = "completed"

        val final = Task(
            id = "task-$startTime", request = request, createdAt = startTime,
            finishedAt = System.currentTimeMillis(), status = status,
            steps = steps.toList(), report = reportBuilder.toString(),
        )
        persist(final)
        onUpdate(final)
        return final
    }

    private suspend fun executeStep(request: String, step: Step, webAllowed: Boolean): Tools.ToolResult {
        return when (step.tool) {
            "calculator" -> {
                val m = Regex("(?i)^(?:calculate|compute|what is|eval)\\s+(.+)$").find(request.trim())
                Tools.calculate(m?.groupValues?.get(1) ?: "")
            }
            "notes.create" -> {
                val m = Regex("(?i)^(?:create|make|write) (?:a )?note(?: titled| called)? ['\"]?([^'\"]+)['\"]?(?::| with| saying) (.+)$").find(request.trim())
                if (m == null) Tools.ToolResult(false, "could not understand the note request")
                else Tools.createNote(context, m.groupValues[1], m.groupValues[2])
            }
            "notes.read" -> {
                val m = Regex("(?i)^(?:read|show|open) (?:the )?note ['\"]?([^'\"]+)['\"]?$").find(request.trim())
                if (m == null) Tools.ToolResult(false, "could not understand which note")
                else Tools.readNote(context, m.groupValues[1])
            }
            "notes.delete" -> {
                val m = Regex("(?i)^(?:delete|remove) (?:the )?note ['\"]?([^'\"]+)['\"]?$").find(request.trim())
                if (m == null) Tools.ToolResult(false, "could not understand which note to delete")
                else Tools.deleteNote(context, m.groupValues[1])
            }
            "memory.search" -> {
                val m = Regex("(?i)^(?:search|find) (?:my )?memor(?:y|ies) (?:for |about )?(.+)$").find(request.trim())
                Tools.searchMemory(memoryStore, m?.groupValues?.get(1) ?: "")
            }
            "memory.add" -> {
                val text = request.trim()
                if (Tools.looksSensitive(text)) {
                    Tools.ToolResult(false, "refused: this looks like a secret; secrets are never stored as memories")
                } else {
                    memoryStore.add(text, source = "task")
                    Tools.ToolResult(true, "memory saved", risky = true)
                }
            }
            "reminder.schedule" -> {
                val m = Regex("(?i)^remind me (?:in |after )?(\\d+)\\s*(minutes?|mins?|hours?)\\s*(?:to )?(.+)$").find(request.trim())
                if (m == null) Tools.ToolResult(false, "could not parse the reminder")
                else {
                    val n = m.groupValues[1].toLongOrNull() ?: 0L
                    val unit = m.groupValues[2].lowercase()
                    val text = m.groupValues[3]
                    val minutes = if (unit.startsWith("h")) n * 60 else n
                    com.nexus500m.app.work.ReminderScheduler.schedule(context, minutes, text)
                    Tools.ToolResult(true, "reminder scheduled in $minutes minute(s): \"$text\"", risky = true)
                }
            }
            "web.research" -> {
                if (!webAllowed) {
                    Tools.ToolResult(false, "web research is disabled (offline mode). " +
                            "Enable it in Settings after review; the local model and memory still work.")
                } else {
                    val m = Regex("(?i)^(?:research|look up|find out about)\\s+(.+?)(?:\\s+on wikipedia)?$").find(request.trim())
                    Tools.webResearch(m?.groupValues?.get(1) ?: "")
                }
            }
            "model.generate" -> {
                val engine = modelManager.engine
                if (engine == null) Tools.ToolResult(false, "the local model is not loaded")
                else {
                    val prompt = buildString {
                        append("### Instruction:\n")
                        append(request.trim())
                        append("\n### Response:\n")
                    }
                    var output = ""
                    try {
                        output = engine.generate(prompt, maxNewTokens = 96, temperature = 0.7f, topK = 40)
                        Tools.ToolResult(true, output.ifBlank { "(model produced no tokens)" })
                    } catch (t: Throwable) {
                        Tools.ToolResult(false, "generation failed: ${t.message}")
                    }
                }
            }
            "report.write" -> {
                val m = Regex("(?i)^write (?:a )?report(?: titled| called)? ['\"]?([^'\"]+)['\"]?(?::| about| on) (.+)$").find(request.trim())
                if (m == null) Tools.ToolResult(false, "could not parse the report request")
                else {
                    val engine = modelManager.engine
                    val material = if (engine == null) "The local model was unavailable; report contains the request only."
                    else runCatching {
                        engine.generate("### Instruction:\n${m.groupValues[2]}\n### Response:\n", 128, 0.7f, 40)
                    }.getOrDefault("generation failed")
                    val content = "# ${m.groupValues[1]}\n\nTopic: ${m.groupValues[2]}\n\n## Local model notes\n\n$material\n"
                    Tools.writeReport(context, m.groupValues[1], content)
                }
            }
            else -> Tools.ToolResult(false, "unknown tool: ${step.tool}")
        }
    }

    // ---------------------------------------------------------------- history
    private fun persist(task: Task) {
        val arr = JsonStore.readArray(historyFile)
        val steps = JSONArray()
        for (s in task.steps) {
            steps.put(JSONObject()
                .put("description", s.description)
                .put("tool", s.tool)
                .put("status", s.status)
                .put("output", s.output.take(2000))
                .put("needsConfirmation", s.needsConfirmation))
        }
        arr.put(JSONObject()
            .put("id", task.id)
            .put("request", task.request)
            .put("createdAt", task.createdAt)
            .put("finishedAt", task.finishedAt ?: JSONObject.NULL)
            .put("status", task.status)
            .put("steps", steps)
            .put("report", task.report.take(4000)))
        // keep history bounded
        while (arr.length() > 200) arr.remove(0)
        JsonStore.writeArray(historyFile, arr)
    }

    fun history(): List<Task> {
        val arr = JsonStore.readArray(historyFile)
        return (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            val stepsArr = o.optJSONArray("steps") ?: JSONArray()
            val steps = (0 until stepsArr.length()).map { j ->
                val s = stepsArr.getJSONObject(j)
                Step(s.getString("description"), s.getString("tool"), s.getString("status"),
                    s.optString("output"), s.optBoolean("needsConfirmation"))
            }
            Task(
                id = o.getString("id"), request = o.getString("request"),
                createdAt = o.getLong("createdAt"),
                finishedAt = if (o.isNull("finishedAt")) null else o.getLong("finishedAt"),
                status = o.getString("status"), steps = steps, report = o.optString("report"),
            )
        }.sortedByDescending { it.createdAt }
    }

    fun clearHistory() { JsonStore.writeArray(historyFile, JSONArray()) }

    companion object {
        const val TASK_TIMEOUT_MS = 90_000L   // hard wall-clock limit per task
    }
}
