package com.nexus500m.app.model

import java.io.File
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.math.tanh

data class EngineConfig(
    val dModel: Int,
    val nLayers: Int,
    val nHeads: Int,
    val dff: Int,
    val ctxLen: Int,
    val vocabSize: Int,
)

/**
 * Pure-Kotlin Transformer inference for the NEXUS architecture
 * (pre-LN, causal multi-head attention, GELU-tanh MLP, tied embeddings).
 *
 * Numerically mirrors nexus/model.py exactly; the JVM golden test compares
 * this engine's logits and greedy continuation against Python output for the
 * same int8-quantized checkpoint.
 *
 * Generation recomputes the full context window per token (no KV cache) —
 * perfectly fine at demo scale; a native runtime (e.g. GGUF) is the intended
 * path for future large NEXUS models and the interface here is designed so
 * that engine can drop in behind the same ModelManager.
 */
class InferenceEngine(
    val info: NexusFormat.ModelInfo,
    private val model: NexusFormat.LoadedModel,
) {
    val config: EngineConfig = info.config
    val tokenizer: BpeTokenizer = info.tokenizer
    private val d = config.dModel
    private val h = config.nHeads
    private val hd = d / h
    private val eps = 1e-5f

    // layer tensor lookup helpers --------------------------------------
    private fun wte() = model.tensor("wte.weight")              // (V, d)
    private fun wpe() = model.tensor("wpe.weight")              // (T, d)
    private fun qkv(l: Int) = model.tensor("blocks.$l.attn.qkv.weight")   // (3d, d)
    private fun proj(l: Int) = model.tensor("blocks.$l.attn.proj.weight") // (d, d)
    private fun ln1w(l: Int) = model.tensor("blocks.$l.ln1.weight")
    private fun ln1b(l: Int) = model.tensor("blocks.$l.ln1.bias")
    private fun ln2w(l: Int) = model.tensor("blocks.$l.ln2.weight")
    private fun ln2b(l: Int) = model.tensor("blocks.$l.ln2.bias")
    private fun fc(l: Int) = model.tensor("blocks.$l.mlp.fc.weight")      // (dff, d)
    private fun fcProj(l: Int) = model.tensor("blocks.$l.mlp.proj.weight") // (d, dff)
    private fun lnfw() = model.tensor("ln_f.weight")
    private fun lnfb() = model.tensor("ln_f.bias")

    /** Full forward pass; returns logits (vocabSize) at the LAST position. */
    fun lastPositionLogits(ids: IntArray): FloatArray {
        val n = ids.size
        require(n in 1..config.ctxLen) { "context length $n out of range" }

        // token + position embeddings -> (n, d)
        val x = FloatArray(n * d)
        val wte = wte(); val wpe = wpe()
        for (t in 0 until n) {
            val tok = ids[t]
            val base = t * d
            val emb = tok * d
            for (i in 0 until d) x[base + i] = wte[emb + i] + wpe[base + i]
        }

        val attnOut = FloatArray(n * d)
        val qkvOut = FloatArray(3 * d)   // kept for clarity of the qkv layout

        for (layer in 0 until config.nLayers) {
            // --- x = x + attn(ln1(x)) ---
            for (t in 0 until n) applyLayerNorm(x, t * d, ln1w(layer), ln1b(layer))

            val qw = qkv(layer); val pw = proj(layer)
            // compute q,k,v per position then causal attention per head
            val qkvAll = FloatArray(n * 3 * d)
            for (t in 0 until n) {
                matvec(qw, x, t * d, qkvAll, t * 3 * d, 3 * d)
            }
            val scores = FloatArray(n)
            val headOut = FloatArray(n * d)
            for (head in 0 until h) {
                val qOff = head * hd
                for (t in 0 until n) {
                    // scores over j <= t
                    var sum = 0f
                    for (j in 0..t) {
                        var dot = 0f
                        val qi = t * 3 * d + qOff
                        val ki = j * 3 * d + d + qOff
                        for (i in 0 until hd) dot += qkvAll[qi + i] * qkvAll[ki + i]
                        val s = dot / sqrt(hd.toFloat())
                        scores[j] = s
                        if (s > sum) sum = s
                    }
                    // softmax over 0..t with max subtraction
                    var total = 0f
                    for (j in 0..t) {
                        scores[j] = exp(scores[j] - sum)
                        total += scores[j]
                    }
                    val inv = 1f / total
                    // weighted sum of values
                    val outBase = t * d + qOff
                    for (i in 0 until hd) {
                        var acc = 0f
                        for (j in 0..t) {
                            acc += scores[j] * qkvAll[j * 3 * d + 2 * d + qOff + i]
                        }
                        headOut[outBase + i] = acc * inv
                    }
                }
            }
            for (t in 0 until n) matvec(pw, headOut, t * d, attnOut, t * d, d)
            for (t in 0 until n) {
                val base = t * d
                for (i in 0 until d) x[base + i] += attnOut[base + i]
            }

            // --- x = x + mlp(ln2(x)) ---
            for (t in 0 until n) applyLayerNorm(x, t * d, ln2w(layer), ln2b(layer))
            val fcw = fc(layer); val fpw = fcProj(layer)
            val hidden = FloatArray(config.dff)
            for (t in 0 until n) {
                matvec(fcw, x, t * d, hidden, 0, config.dff)
                for (i in 0 until config.dff) {
                    val v = hidden[i]
                    hidden[i] = 0.5f * v * (1f + tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)))
                }
                matvec(fpw, hidden, 0, attnOut, t * d, d)
                val base = t * d
                for (i in 0 until d) x[base + i] += attnOut[base + i]
            }
        }

        // final layernorm on last position, then tied unembedding
        val last = (n - 1) * d
        applyLayerNorm(x, last, lnfw(), lnfb())
        val logits = FloatArray(config.vocabSize)
        matvecT(wte(), x, last, logits, d, config.vocabSize)
        return logits
    }

    /**
     * Generate continuation for [prompt]. Calls [onText] with the full text so
     * far after each token (streaming). Stops on EOS, [maxNewTokens], or when
     * [shouldStop] returns true (global stop button). Returns final text.
     */
    fun generate(
        prompt: String,
        maxNewTokens: Int,
        temperature: Float = 0.8f,
        topK: Int = 40,
        rng: kotlin.random.Random = kotlin.random.Random.Default,
        shouldStop: () -> Boolean = { false },
        onText: (String) -> Unit = {},
    ): String {
        val promptIds = tokenizer.encode(prompt)
        val ids = ArrayList<Int>(promptIds.size + maxNewTokens)
        promptIds.forEach { ids.add(it) }

        var produced = StringBuilder()
        var newTokens = 0
        while (newTokens < maxNewTokens) {
            if (shouldStop()) break
            val window = if (ids.size > config.ctxLen) {
                ids.subList(ids.size - config.ctxLen, ids.size).toIntArray()
            } else ids.toIntArray()
            val logits = lastPositionLogits(window)

            val next: Int = if (temperature <= 0.01f) {
                argmax(logits)
            } else {
                val scaled = FloatArray(logits.size) { logits[it] / temperature }
                topKSample(scaled, topK, rng)
            }
            if (next == tokenizer.eosTokenId) break
            ids.add(next)
            newTokens++
            val text = tokenizer.decode(ids.toIntArray())
            val continuation = text.substring(prompt.length.coerceAtMost(text.length))
            produced = StringBuilder(continuation)
            onText(continuation)
        }
        return produced.toString()
    }

    // math helpers -----------------------------------------------------

    /** out = W @ x  where W is (outDim, inDim) row-major; x read from x[xOff]. */
    private fun matvec(w: FloatArray, x: FloatArray, xOff: Int, out: FloatArray, outOff: Int, outDim: Int) {
        val inDim = w.size / outDim
        for (j in 0 until outDim) {
            var acc = 0f
            val wOff = j * inDim
            for (i in 0 until inDim) acc += w[wOff + i] * x[xOff + i]
            out[outOff + j] = acc
        }
    }

    /** out = W^T @ x (embedding transpose for tied unembedding), W is (rows=d, cols=V). */
    private fun matvecT(w: FloatArray, x: FloatArray, xOff: Int, out: FloatArray, outOff: Int, d: Int, v: Int) {
        for (j in 0 until v) {
            var acc = 0f
            val wOff = j * d
            for (i in 0 until d) acc += w[wOff + i] * x[xOff + i]
            out[outOff + j] = acc
        }
    }

    private fun applyLayerNorm(x: FloatArray, off: Int, gamma: FloatArray, beta: FloatArray) {
        var mean = 0f
        for (i in 0 until d) mean += x[off + i]
        mean /= d
        var variance = 0f
        for (i in 0 until d) {
            val diff = x[off + i] - mean
            variance += diff * diff
        }
        variance /= d
        val invStd = 1f / sqrt(variance + eps)
        for (i in 0 until d) {
            x[off + i] = (x[off + i] - mean) * invStd * gamma[i] + beta[i]
        }
    }

    private fun argmax(logits: FloatArray): Int {
        var best = 0
        var bestV = logits[0]
        for (i in 1 until logits.size) {
            if (logits[i] > bestV) {
                bestV = logits[i]
                best = i
            }
        }
        return best
    }

    private fun topKSample(logits: FloatArray, topK: Int, rng: kotlin.random.Random): Int {
        val k = if (topK <= 0 || topK >= logits.size) logits.size else topK
        // partial selection: indices sorted desc by logit
        val idx = IntArray(logits.size) { it }
        val order = idx.sortedByDescending { logits[it] }.take(k)
        // softmax over top-k
        var mx = logits[order[0]]
        var sum = 0f
        val probs = FloatArray(k)
        for (i in 0 until k) {
            val e = exp(logits[order[i]] - mx)
            probs[i] = e
            sum += e
        }
        var r = rng.nextDouble() * sum
        for (i in 0 until k) {
            r -= probs[i]
            if (r <= 0) return order[i]
        }
        return order[k - 1]
    }

    private fun nLayersWorkspace(): Int = config.nLayers

    companion object {
        fun load(file: File): InferenceEngine {
            val loaded = NexusFormat.load(file)
            return InferenceEngine(loaded.info, loaded)
        }
    }
}
