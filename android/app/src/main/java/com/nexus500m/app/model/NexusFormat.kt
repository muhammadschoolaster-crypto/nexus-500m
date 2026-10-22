package com.nexus500m.app.model

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.zip.ZipFile

/**
 * Reader for the `.nexus` model format produced by nexus/export.py:
 * a ZIP containing manifest.json (config + tensor table + tokenizer) and
 * weights.bin (per-tensor symmetric int8 quantization).
 *
 * Integrity: weights.bin is SHA-256-verified against the manifest before any
 * tensor is dequantized — a tampered or truncated model never loads.
 */
object NexusFormat {

    const val MAGIC = "NEXUS1"
    const val FORMAT_VERSION = 1

    data class TensorSpec(
        val name: String,
        val shape: IntArray,
        val offset: Int,
        val nBytes: Int,
        val scale: Float,
    )

    data class ModelInfo(
        val modelName: String,
        val modelVersion: String,
        val notes: String,
        val checkpointStep: Int,
        val config: EngineConfig,
        val tokenizer: BpeTokenizer,
        val weightsSha256: String,
    ) {
        val parameterCount: Long get() = _paramCount
        private var _paramCount: Long = 0

        internal fun setParamCount(v: Long) { _paramCount = v }
    }

    data class LoadedModel(
        val info: ModelInfo,
        private val weights: ByteArray,
        val specs: List<TensorSpec>,
    ) {
        private val cache = HashMap<String, FloatArray>()

        /** Dequantized float32 tensor (row-major). */
        fun tensor(name: String): FloatArray = cache.getOrPut(name) {
            val spec = specs.first { it.name == name }
            val n = spec.nBytes
            val out = FloatArray(n)
            for (i in 0 until n) {
                out[i] = (weights[spec.offset + i].toInt()) * spec.scale
            }
            out
        }
    }

    fun load(file: File): LoadedModel {
        ZipFile(file).use { zf ->
            val manifestEntry = zf.getEntry("manifest.json")
                ?: throw IllegalArgumentException(".nexus missing manifest.json")
            val weightsEntry = zf.getEntry("weights.bin")
                ?: throw IllegalArgumentException(".nexus missing weights.bin")
            val m = JSONObject(String(zf.getInputStream(manifestEntry).readBytes(), Charsets.UTF_8))

            val magic = m.getString("magic")
            require(magic == MAGIC) { "not a .nexus file (magic=$magic)" }
            val version = m.getInt("format_version")
            require(version == FORMAT_VERSION) { "unsupported .nexus format version $version" }

            val weights = zf.getInputStream(weightsEntry).readBytes()
            val sha = sha256Hex(weights)
            require(sha == m.getString("weights_sha256")) {
                "model integrity check FAILED: weights.bin does not match its manifest fingerprint"
            }

            val cfgJson = m.getJSONObject("config")
            val config = EngineConfig(
                dModel = cfgJson.getInt("d_model"),
                nLayers = cfgJson.getInt("n_layers"),
                nHeads = cfgJson.getInt("n_heads"),
                dff = cfgJson.getInt("d_ff"),
                ctxLen = cfgJson.getInt("ctx_len"),
                vocabSize = cfgJson.getInt("vocab_size"),
            )

            val tensorsJson = m.getJSONArray("tensors")
            val specs = ArrayList<TensorSpec>(tensorsJson.length())
            var paramCount = 0L
            for (i in 0 until tensorsJson.length()) {
                val t = tensorsJson.getJSONObject(i)
                val shapeArr = t.getJSONArray("shape")
                val shapeInts = IntArray(shapeArr.length()) { k -> shapeArr.getInt(k) }
                val n = t.getInt("n_bytes")
                specs.add(
                    TensorSpec(
                        name = t.getString("name"),
                        shape = shapeInts,
                        offset = t.getInt("offset"),
                        nBytes = n,
                        scale = t.getDouble("scale").toFloat(),
                    )
                )
                paramCount += n
            }

            val tokJson = m.getJSONObject("tokenizer")
            val mergesArr = tokJson.getJSONArray("merges")
            val merges = ArrayList<Pair<String, String>>(mergesArr.length())
            for (i in 0 until mergesArr.length()) {
                val pair = mergesArr.getJSONArray(i)
                merges.add(pair.getString(0) to pair.getString(1))
            }
            val tokenizer = BpeTokenizer(merges)
            require(tokJson.getInt("vocab_size") == config.vocabSize) {
                "tokenizer vocab does not match model config"
            }

            val info = ModelInfo(
                modelName = m.getString("model_name"),
                modelVersion = m.getString("model_version"),
                notes = if (m.has("notes") && !m.isNull("notes")) m.getString("notes") else "",
                checkpointStep = m.optJSONObject("training")?.optInt("checkpoint_step") ?: -1,
                config = config,
                tokenizer = tokenizer,
                weightsSha256 = sha,
            )
            info.setParamCount(paramCount)
            return LoadedModel(info, weights, specs)
        }
    }

    fun sha256Hex(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(1 shl 16)
            while (true) {
                val r = input.read(buf)
                if (r < 0) break
                md.update(buf, 0, r)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    private fun sha256Hex(data: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(data)
            .joinToString("") { "%02x".format(it) }
}
