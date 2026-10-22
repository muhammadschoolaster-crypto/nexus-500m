package com.nexus500m.app.data

import org.json.JSONObject
import java.io.File

/**
 * App settings, persisted as JSON. Everything user-controllable in one place:
 * memory on/off, web research consent, agent permission, generation params,
 * theme, storage cap.
 */
class SettingsStore(private val file: File) {

    data class Settings(
        val memoryEnabled: Boolean,
        val webResearchEnabled: Boolean,
        val agentEnabled: Boolean,
        val maxNewTokens: Int,
        val temperature: Float,
        val topK: Int,
        val darkTheme: Boolean,
        val storageCapMb: Long,
    )

    fun load(): Settings {
        val o = JsonStore.readObject(file)
        return Settings(
            memoryEnabled = o.optBoolean("memoryEnabled", true),
            webResearchEnabled = o.optBoolean("webResearchEnabled", false),
            agentEnabled = o.optBoolean("agentEnabled", true),
            maxNewTokens = o.optInt("maxNewTokens", 96),
            temperature = o.optDouble("temperature", 0.7).toFloat(),
            topK = o.optInt("topK", 40),
            darkTheme = o.optBoolean("darkTheme", true),
            storageCapMb = o.optLong("storageCapMb", 512L),
        )
    }

    fun save(s: Settings) {
        JsonStore.writeObject(file, JSONObject()
            .put("memoryEnabled", s.memoryEnabled)
            .put("webResearchEnabled", s.webResearchEnabled)
            .put("agentEnabled", s.agentEnabled)
            .put("maxNewTokens", s.maxNewTokens)
            .put("temperature", s.temperature.toDouble())
            .put("topK", s.topK)
            .put("darkTheme", s.darkTheme)
            .put("storageCapMb", s.storageCapMb))
    }
}
