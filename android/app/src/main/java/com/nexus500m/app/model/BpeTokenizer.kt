package com.nexus500m.app.model

/**
 * Byte-level BPE tokenizer — EXACT mirror of nexus/tokenizer.py (Python).
 *
 * Base vocab: 256 byte tokens (GPT-2 byte-to-unicode mapping), token id == byte value.
 * Learned merges: id 256 + merge index. Reserved EOS token: id vocab_size - 1.
 *
 * Parity is enforced by the golden unit test: encoding the golden prompt with
 * this class must produce the exact token ids Python produced.
 */
class BpeTokenizer(merges: List<Pair<String, String>>) {

    val merges: List<Pair<String, String>> = merges
    private val ranks: Map<Pair<String, String>, Int> =
        merges.withIndex().associate { (i, m) -> m to i }
    private val mergeIds: Map<String, Int> =
        merges.withIndex().associate { (i, m) -> m.first + m.second to BASE_VOCAB + i }
    private val idToSym: Map<Int, String> =
        buildMap {
            for (b in 0..255) put(b, BYTE_ENCODER[b].toString())
            merges.withIndex().forEach { (i, m) -> putIfAbsent(BASE_VOCAB + i, m.first + m.second) }
        }

    val eosTokenId: Int = BASE_VOCAB + merges.size
    val vocabSize: Int = eosTokenId + 1

    /** Encode text to token ids (optionally appending the EOS id). */
    fun encode(text: String, addEos: Boolean = false): IntArray {
        val bytes = text.toByteArray(Charsets.UTF_8)
        var syms: List<String> = bytes.map { BYTE_ENCODER[it.toInt() and 0xFF].toString() }
        syms = bpe(syms)
        val ids = IntArray(syms.size + if (addEos) 1 else 0)
        syms.forEachIndexed { i, s ->
            ids[i] = if (s.length == 1) {
                BYTE_DECODER[s] ?: throw IllegalStateException("unmapped byte char $s")
            } else {
                mergeIds[s] ?: throw IllegalStateException("unmapped merged symbol $s")
            }
        }
        if (addEos) ids[ids.size - 1] = eosTokenId
        return ids
    }

    /** Decode ids back to text; unknown/eos ids are dropped (same as Python). */
    fun decode(ids: IntArray): String {
        val out = ArrayList<Byte>(ids.size * 2)
        for (id in ids) {
            val sym = idToSym[id] ?: continue
            for (ch in sym) {
                BYTE_DECODER[ch]?.let { out.add(it) }
            }
        }
        return String(out.toByteArray(), Charsets.UTF_8)  // replace malformed tails like Python
    }

    /** Apply merges by rank; all occurrences of the best pair per round (GPT-2 style). */
    private fun bpe(input: List<String>): List<String> {
        var syms = input
        while (syms.size >= 2) {
            var bestRank = Int.MAX_VALUE
            var bestPair: Pair<String, String>? = null
            for (i in 0 until syms.size - 1) {
                val r = ranks[syms[i] to syms[i + 1]] ?: continue
                if (r < bestRank) {
                    bestRank = r
                    bestPair = syms[i] to syms[i + 1]
                }
            }
            val pair = bestPair ?: break
            val newSym = pair.first + pair.second
            val out = ArrayList<String>(syms.size)
            var i = 0
            while (i < syms.size) {
                if (i < syms.size - 1 && syms[i] == pair.first && syms[i + 1] == pair.second) {
                    out.add(newSym)
                    i += 2
                } else {
                    out.add(syms[i])
                    i += 1
                }
            }
            syms = out
        }
        return syms
    }

    companion object {
        const val BASE_VOCAB = 256

        /** byte value -> printable char (identical algorithm to Python bytes_to_unicode) */
        val BYTE_ENCODER: Map<Int, Char> = buildByteToUnicode()
        val BYTE_DECODER: Map<Char, Int> = BYTE_ENCODER.entries.associate { (b, c) -> c to b }

        private fun buildByteToUnicode(): Map<Int, Char> {
            val bsList = ArrayList<Int>()
            bsList.addAll(33..126)     // '!'..'~'
            bsList.addAll(161..172)    // '¡'..'¬'
            bsList.addAll(174..255)    // '®'..'ÿ'
            val bs = bsList.toIntArray()
            val bsSet = bs.toHashSet()
            val cs = ArrayList<Int>()
            for (b in bs) cs.add(b)
            var n = 0
            for (b in 0..255) {
                if (b !in bsSet) {
                    cs.add(256 + n)
                    n += 1
                }
            }
            val map = HashMap<Int, Char>(256)
            for (i in bs.indices) map[bs[i]] = cs[i].toChar()
            return map
        }
    }
}
