package com.nexus.assistant

import org.json.JSONArray
import java.util.regex.Pattern

/**
 * NEXUS BPE tokenizer - exact Kotlin port of nexus_lm/tokenizer.py.
 * Pretokenize: runs of whitespace / non-whitespace.
 * Encode: per pretoken, repeatedly apply the lowest-rank applicable merge.
 * Decode: concatenate token strings (lossless). Unknown chars -> id 0 (<unk>).
 */
class NexusTokenizer(vocabJson: JSONArray, mergesJson: JSONArray) {

    private val vocab: List<String>
    private val merges: List<Pair<String, String>>
    private val stoi = HashMap<String, Int>()
    private val ranks = HashMap<Pair<String, String>, Int>()

    private val pretokRe = Pattern.compile("\\s+|\\S+")

    init {
        val v = ArrayList<String>(vocabJson.length())
        v.add("<unk>")
        for (i in 0 until vocabJson.length()) v.add(vocabJson.getString(i))
        vocab = v
        for (i in vocab.indices) stoi[vocab[i]] = i

        val m = ArrayList<Pair<String, String>>(mergesJson.length())
        for (i in 0 until mergesJson.length()) {
            val pair = mergesJson.getJSONArray(i)
            val p = Pair(pair.getString(0), pair.getString(1))
            m.add(p)
            ranks[p] = i
        }
        merges = m
    }

    val vocabSize: Int get() = vocab.size

    /** Split string into code-point strings (mirrors Python per-char iteration). */
    private fun codePoints(s: String): List<String> {
        val out = ArrayList<String>(s.length)
        var i = 0
        while (i < s.length) {
            val cp = s.codePointAt(i)
            out.add(String(Character.toChars(cp)))
            i += Character.charCount(cp)
        }
        return out
    }

    private fun encodePretoken(syms: List<String>): List<Int> {
        if (syms.size == 1) return listOf(stoi[syms[0]] ?: 0)
        val cur = ArrayList(syms)
        while (cur.size > 1) {
            var bestIdx = -1
            var bestRank = Int.MAX_VALUE
            for (j in 0 until cur.size - 1) {
                val r = ranks[Pair(cur[j], cur[j + 1])] ?: continue
                if (r < bestRank) {
                    bestRank = r
                    bestIdx = j
                }
            }
            if (bestIdx < 0) break
            val merged = cur[bestIdx] + cur[bestIdx + 1]
            cur[bestIdx] = merged
            cur.removeAt(bestIdx + 1)
        }
        return cur.map { stoi[it] ?: 0 }
    }

    fun encode(text: String): List<Int> {
        val ids = ArrayList<Int>()
        val matcher = pretokRe.matcher(text)
        while (matcher.find()) {
            ids.addAll(encodePretoken(codePoints(matcher.group())))
        }
        return ids
    }

    fun decode(ids: List<Int>): String {
        val sb = StringBuilder()
        for (id in ids) if (id in 1 until vocab.size) sb.append(vocab[id])
        return sb.toString()
    }
}
