package com.nexus.assistant

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

/**
 * NEXUS on-device inference engine (pure Kotlin, zero native dependencies).
 * Loads the fp32 export of the genuinely-trained NEXUS checkpoint and runs a
 * GPT-style decoder with incremental K/V caching.
 *
 * Tensor plan (fp32 little-endian, exact order - mirrors export_android.py):
 *   tokEmb[V,d]; posEmb[T,d];
 *   per layer: ln1w[d],ln1b[d], qkvW[3d,d],qkvB[3d], projW[d,d],projB[d],
 *              ln2w[d],ln2b[d], fcW[ff,d],fcB[ff], proj2W[d,ff],proj2B[d];
 *   lnfw[d],lnfb[d]; headW[V,d]
 * Matrices row-major ([out,in]), matching PyTorch Linear.weight.
 */
class NexusEngine private constructor(
    val modelName: String,
    val paramCount: Long,
    private val d: Int,
    private val L: Int,
    private val H: Int,
    private val ff: Int,
    private val T: Int,
    private val V: Int,
    private val w: FloatArray,
    val tokenizer: NexusTokenizer
) {
    private val dh = d / H

    private var oTok = 0L
    private var oPos = 0L
    private val oL = Array(L) { LongArray(12) }
    private var oLnfw = 0L
    private var oLnfb = 0L

    private val kCache = Array(L) { FloatArray(T * d) }
    private val vCache = Array(L) { FloatArray(T * d) }
    private var pos = 0
    private val idsHistory = ArrayList<Int>(1024)

    companion object {
        private const val TAG = "NexusEngine"

        fun load(context: Context): NexusEngine? {
            return try {
                val t0 = System.currentTimeMillis()
                val cfg = JSONObject(String(readAsset(context, "nexus_config.json"), Charsets.UTF_8))
                val d = cfg.getInt("d_model")
                val L = cfg.getInt("n_layers")
                val ff = cfg.getInt("d_ff")
                val T = cfg.getInt("block_size")
                val V = cfg.getInt("vocab_size")
                val expectedBytes = cfg.getLong("bin_bytes")

                val bin = readAsset(context, "nexus_model.bin")
                val bb = ByteBuffer.wrap(bin).order(ByteOrder.LITTLE_ENDIAN)
                val w = FloatArray(bin.size / 4)
                bb.asFloatBuffer().get(w)
                check(bin.size.toLong() == expectedBytes) {
                    "model size mismatch: ${bin.size} != $expectedBytes"
                }

                val vocab = JSONArray(String(readAsset(context, "tokenizer_vocab.json"), Charsets.UTF_8))
                val merges = JSONArray(String(readAsset(context, "tokenizer_merges.json"), Charsets.UTF_8))
                val tok = NexusTokenizer(vocab, merges)

                val eng = NexusEngine(
                    modelName = cfg.optString("name", "NEXUS"),
                    paramCount = cfg.getLong("params_total"),
                    d = d, L = L, H = cfg.getInt("n_heads"), ff = ff, T = T, V = V,
                    w = w, tokenizer = tok
                )
                eng.computeOffsets()
                Log.i(TAG, "loaded ${eng.modelName}: ${eng.paramCount} params " +
                        "in ${System.currentTimeMillis() - t0} ms")
                eng
            } catch (e: Exception) {
                Log.e(TAG, "model load failed", e)
                null
            }
        }

        private fun readAsset(context: Context, name: String): ByteArray {
            var buf = ByteArray(1 shl 20)
            var len = 0
            BufferedInputStream(context.assets.open(name)).use { ins ->
                val chunk = ByteArray(1 shl 16)
                while (true) {
                    val n = ins.read(chunk)
                    if (n < 0) break
                    if (len + n > buf.size) {
                        var cap = buf.size
                        while (cap < len + n) cap = cap shl 1
                        buf = buf.copyOf(cap)
                    }
                    System.arraycopy(chunk, 0, buf, len, n)
                    len += n
                }
            }
            return if (len == buf.size) buf else buf.copyOf(len)
        }
    }

    private fun computeOffsets() {
        var o = 0L
        oTok = o; o += V.toLong() * d
        oPos = o; o += T.toLong() * d
        for (l in 0 until L) {
            oL[l][0] = o; o += d
            oL[l][1] = o; o += d
            oL[l][2] = o; o += 3L * d * d
            oL[l][3] = o; o += 3L * d
            oL[l][4] = o; o += d.toLong() * d
            oL[l][5] = o; o += d
            oL[l][6] = o; o += d
            oL[l][7] = o; o += d
            oL[l][8] = o; o += ff.toLong() * d
            oL[l][9] = o; o += ff
            oL[l][10] = o; o += d.toLong() * ff
            oL[l][11] = o; o += d
        }
        oLnfw = o; o += d
        oLnfb = o; o += d
        oHead = o; o += V.toLong() * d
        check(o == w.size.toLong()) { "offset plan mismatch: $o != ${w.size}" }
    }

    private var oHead = 0L

    // ---------- math kernels ----------
    private fun layerNorm(dst: FloatArray, x: FloatArray, wOff: Long, bOff: Long) {
        var mean = 0f
        for (i in 0 until d) mean += x[i]
        mean /= d
        var vr = 0f
        for (i in 0 until d) { val v = x[i] - mean; vr += v * v }
        vr /= d
        val inv = 1f / sqrt(vr + 1e-5f)
        for (i in 0 until d) {
            dst[i] = (x[i] - mean) * inv * w[(wOff + i).toInt()] + w[(bOff + i).toInt()]
        }
    }

    /** out = bias + W·x  (W row-major [outDim,inDim]) */
    private fun matvecBias(bOff: Long, wOff: Long, x: FloatArray, out: FloatArray,
                           outDim: Int, inDim: Int) {
        for (o in 0 until outDim) out[o] = w[(bOff + o).toInt()]
        for (o in 0 until outDim) {
            var s = 0f
            val wo = (wOff + o.toLong() * inDim).toInt()
            for (i in 0 until inDim) s += w[wo + i] * x[i]
            out[o] += s
        }
    }

    /** Error function (Abramowitz & Stegun 7.1.26, |err| < 1.5e-7). */
    private fun erf(x0: Float): Float {
        val x = x0.toDouble()
        val t = 1.0 / (1.0 + 0.5 * abs(x))
        val tau = t * exp(-x * x - 1.26551223 + t * (1.00002368 +
                t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
                t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
                t * (-0.82215223 + t * 0.17087277)))))))))
        return (if (x >= 0) 1.0 - tau else tau - 1.0).toFloat()
    }

    private fun gelu(v: FloatArray) {
        val c = 1f / sqrt(2f)
        for (i in v.indices) {
            val x = v[i]
            v[i] = 0.5f * x * (1f + erf(x * c))
        }
    }

    // ---------- forward one token ----------
    private val xBuf = FloatArray(d)
    private val hBuf = FloatArray(d)
    private val qkvBuf = FloatArray(3 * d)
    private val attnBuf = FloatArray(d)
    private val fBuf = FloatArray(maxOf(ff, T))

    /** Processes one token into cache slot `pos-1` (post-increment). */
    private fun forwardToken(id: Int) {
        check(pos < T) { "context overflow" }
        val p = pos
        val te = (oTok + id.toLong() * d).toInt()
        val pe = (oPos + p.toLong() * d).toInt()
        for (i in 0 until d) xBuf[i] = w[te + i] + w[pe + i]

        val scale = 1f / sqrt(dh.toFloat())
        for (l in 0 until L) {
            val o = oL[l]
            layerNorm(hBuf, xBuf, o[0], o[1])
            matvecBias(o[3], o[2], hBuf, qkvBuf, 3 * d, d)

            System.arraycopy(qkvBuf, d, kCache[l], p * d, d)
            System.arraycopy(qkvBuf, 2 * d, vCache[l], p * d, d)

            for (h in 0 until H) {
                val hOff = h * dh
                var maxS = Float.NEGATIVE_INFINITY
                for (t in 0..p) {
                    var s = 0f
                    val ko = t * d + hOff
                    for (i in 0 until dh) s += qkvBuf[hOff + i] * kCache[l][ko + i]
                    s *= scale
                    fBuf[t] = s
                    if (s > maxS) maxS = s
                }
                var sum = 0f
                for (t in 0..p) { val e = exp(fBuf[t] - maxS); fBuf[t] = e; sum += e }
                val invSum = 1f / sum
                for (i in 0 until dh) {
                    var acc = 0f
                    for (t in 0..p) acc += fBuf[t] * vCache[l][t * d + hOff + i]
                    attnBuf[hOff + i] = acc * invSum
                }
            }
            matvecBias(o[5], o[4], attnBuf, hBuf, d, d)   // hBuf -> attn proj out
            for (i in 0 until d) xBuf[i] += hBuf[i]

            layerNorm(hBuf, xBuf, o[6], o[7])
            matvecBias(o[9], o[8], hBuf, fBuf, ff, d)
            gelu(fBuf)
            matvecBias(o[11], o[10], fBuf, hBuf, d, ff)
            for (i in 0 until d) xBuf[i] += hBuf[i]
        }
        layerNorm(hBuf, xBuf, oLnfw, oLnfb)   // final hidden state -> hBuf
        pos++
    }

    private fun lastLogits(): FloatArray {
        val out = FloatArray(V)
        for (o in 0 until V) {
            var s = 0f
            val wo = (oHead + o.toLong() * d).toInt()
            for (i in 0 until d) s += w[wo + i] * hBuf[i]
            out[o] = s
        }
        return out
    }

    // ---------- public API ----------
    fun reset() { pos = 0 }
    fun contextLength(): Int = pos

    private val lock = Any()

    /** Append tokens to the running context (replay if window slides). */
    fun append(ids: List<Int>) {
        synchronized(lock) {
            idsHistory.addAll(ids)
            reset()
            val n = minOf(idsHistory.size, T)
            val start = idsHistory.size - n
            for (i in start until idsHistory.size) forwardToken(idsHistory[i])
        }
    }

    /** Greedy argmax next-token id (used for the engine self-check). */
    fun argmaxNext(): Int {
        val lg = lastLogits()
        var best = 0
        var bv = Float.NEGATIVE_INFINITY
        for (i in lg.indices) if (lg[i] > bv) { bv = lg[i]; best = i }
        return best
    }

    /**
     * Generate up to maxNew tokens. Callback receives decoded text deltas.
     * Stops on the "<|end|>" marker. Runs synchronously - invoke off the UI thread.
     */
    fun generate(maxNew: Int, temperature: Float, topK: Int, onDelta: (String) -> Unit): String {
        synchronized(lock) {
            val sb = StringBuilder()
            val temp = max(temperature, 0.05f)
            for (step in 0 until maxNew) {
                val logits = lastLogits()
                var maxL = Float.NEGATIVE_INFINITY
                for (i in logits.indices) {
                    logits[i] /= temp
                    if (logits[i] > maxL) maxL = logits[i]
                }
                val k = topK.coerceIn(1, V)
                if (k < V) {
                    val sorted = logits.copyOf().also { it.sort() }
                    val cutoff = sorted[V - k]
                    for (i in logits.indices) if (logits[i] < cutoff) logits[i] = Float.NEGATIVE_INFINITY
                }
                var sum = 0f
                for (i in logits.indices) {
                    val e = if (logits[i].isFinite()) exp(logits[i] - maxL) else 0f
                    logits[i] = e
                    sum += e
                }
                var r = Math.random() * sum
                var chosen = V - 1
                for (i in logits.indices) {
                    r -= logits[i]
                    if (r <= 0f) { chosen = i; break }
                }
                val piece = if (chosen in 1 until tokenizer.vocabSize) tokenizer.decode(listOf(chosen)) else ""
                if (piece.contains("<|end|>")) break
                sb.append(piece)
                onDelta(piece)
                if (pos >= T) break
                forwardToken(chosen)
                idsHistory.add(chosen)
            }
            return sb.toString()
        }
    }
}
