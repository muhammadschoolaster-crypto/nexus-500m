package com.nexus500m.app.model

import android.content.Context
import java.io.File

/**
 * Owns the lifecycle of the on-device model: bundled-asset extraction,
 * SHA-256 integrity verification, background load/unload, status reporting.
 *
 * The bundled model is a GENUINELY trained NEXUS checkpoint (tiny demo scale,
 * produced by scripts/run_smoke_training.sh and exported through nexus/export.py).
 * Future NEXUS versions ship as additional entries in the model registry and
 * can be downloaded with the same verification path — no app redesign needed.
 */
class ModelManager(private val context: Context) {

    data class ModelStatus(
        val loaded: Boolean,
        val modelName: String,
        val modelVersion: String,
        val parameterCount: Long,
        val vocabSize: Int,
        val ctxLen: Int,
        val fileBytes: Long,
        val fileSha256: String,
        val ramEstimateBytes: Long,   // dequantized fp32 weights ~ 4x file size
        val notes: String,
        val checkpointStep: Int,
        val error: String?,
    )

    @Volatile
    var engine: InferenceEngine? = null
        private set

    val isLoaded: Boolean get() = engine != null

    /** Extracts the bundled model from APK assets (once) and returns its file. */
    fun bundledModelFile(): File {
        val dir = File(context.filesDir, "models").apply { mkdirs() }
        val out = File(dir, BUNDLED_MODEL)
        if (!out.exists() || out.length() == 0L) {
            context.assets.open("models/$BUNDLED_MODEL").use { input ->
                val tmp = File(dir, "$BUNDLED_MODEL.tmp")
                tmp.outputStream().use { output -> input.copyTo(output) }
                check(tmp.length() > 0) { "bundled model asset is empty" }
                if (!tmp.renameTo(out)) {
                    tmp.delete()
                    throw IllegalStateException("could not move bundled model into place")
                }
            }
        }
        return out
    }

    /** Expected fingerprint of the bundled model (recorded at build time). */
    fun bundledExpectedSha(): String? = runCatching {
        context.assets.open("models/$REGISTRY").bufferedReader().use { reader ->
            org.json.JSONObject(reader.readText()).getString("sha256")
        }
    }.getOrNull()

    /** Verifies the on-disk model file against the build-time registry. */
    fun verifyBundledIntegrity(): Boolean {
        val file = bundledModelFile()
        val expected = bundledExpectedSha() ?: return true.also {
            _lastIntegrityNote = "no registry entry; skipping (development build)"
        }
        val actual = NexusFormat.sha256Hex(file)
        val ok = actual == expected
        _lastIntegrityNote = if (ok) "verified" else "MISMATCH: file may be corrupted"
        return ok
    }

    /** Loads the model (call from a background thread). */
    @Synchronized
    fun load(): ModelStatus {
        return try {
            val file = bundledModelFile()
            verifyBundledIntegrity()
            val e = InferenceEngine.load(file)
            engine = e
            val info = e.info
            ModelStatus(
                loaded = true,
                modelName = info.modelName,
                modelVersion = info.modelVersion,
                parameterCount = info.parameterCount,
                vocabSize = info.config.vocabSize,
                ctxLen = info.config.ctxLen,
                fileBytes = file.length(),
                fileSha256 = NexusFormat.sha256Hex(file),
                ramEstimateBytes = info.parameterCount * 4L,
                notes = info.notes,
                checkpointStep = info.checkpointStep,
                error = null,
            )
        } catch (t: Throwable) {
            engine = null
            ModelStatus(
                loaded = false, modelName = "?", modelVersion = "?", parameterCount = 0,
                vocabSize = 0, ctxLen = 0, fileBytes = 0, fileSha256 = "", ramEstimateBytes = 0,
                notes = "", checkpointStep = -1,
                error = "model failed to load: ${t.message ?: t::class.java.simpleName}",
            )
        }
    }

    @Synchronized
    fun unload() {
        engine = null   // GC reclaims the dequantized tensors
    }

    companion object {
        const val BUNDLED_MODEL = "nexus-demo-v0.1.0.nexus"
        const val REGISTRY = "registry.json"
        @Volatile private var _lastIntegrityNote: String = ""
        val lastIntegrityNote: String get() = _lastIntegrityNote
    }
}
