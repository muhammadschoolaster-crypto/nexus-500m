package com.nexus500m.app.data

import org.json.JSONArray
import org.json.JSONObject
import java.io.File

data class Memory(
    val id: String,
    var text: String,
    val createdAt: Long,
    val source: String,      // e.g. "user" | "task"
    val supersededBy: String?,  // id of a newer memory that contradicts+replaces this one
)

/**
 * Persistent personal memory. IMPORTANT design facts (kept honest in the UI too):
 * - This is RETRIEVAL AUGMENTATION, not model retraining: memories are injected
 *   into the prompt as context. The base model weights never change from chat.
 * - Fully user-controlled: view / edit / delete / export / disable.
 * - No secrets are ever stored as memories (the task engine refuses to save
 *   anything that looks like a password or key).
 */
class MemoryStore(private val file: File) {

    var enabled: Boolean = true   // mirrored from settings by the app layer

    private fun loadAll(): MutableList<Memory> {
        val arr = JsonStore.readArray(file)
        val list = ArrayList<Memory>(arr.length())
        for (i in 0 until arr.length()) {
            val m = arr.getJSONObject(i)
            list.add(
                Memory(
                    id = m.getString("id"),
                    text = m.getString("text"),
                    createdAt = m.getLong("createdAt"),
                    source = m.optString("source", "user"),
                    supersededBy = if (m.isNull("supersededBy")) null else m.getString("supersededBy"),
                )
            )
        }
        return list
    }

    private fun saveAll(list: List<Memory>) {
        val arr = JSONArray()
        for (m in list) {
            arr.put(JSONObject()
                .put("id", m.id)
                .put("text", m.text)
                .put("createdAt", m.createdAt)
                .put("source", m.source)
                .put("supersededBy", m.supersededBy ?: JSONObject.NULL))
        }
        JsonStore.writeArray(file, arr)
    }

    /** Add a memory. Detects contradictions: same topic, different content -> old one marked superseded. */
    @Synchronized
    fun add(text: String, source: String = "user"): Memory {
        val all = loadAll()
        val now = System.currentTimeMillis()
        val mem = Memory(
            id = java.util.UUID.randomUUID().toString(),
            text = text.trim(),
            createdAt = now,
            source = source,
            supersededBy = null,
        )
        // contradiction handling: if an existing memory shares its "subject key"
        // (first ~5 significant words) but differs, mark it superseded by this one
        val key = subjectKey(mem.text)
        for (other in all) {
            if (other.supersededBy == null && other.id != mem.id &&
                subjectKey(other.text) == key && other.text.trim() != mem.text
            ) {
                other.supersededBy = mem.id
            }
        }
        all.add(mem)
        saveAll(all)
        return mem
    }

    fun list(includeSuperseded: Boolean = false): List<Memory> =
        loadAll().filter { includeSuperseded || it.supersededBy == null }
            .sortedByDescending { it.createdAt }

    fun update(id: String, newText: String) {
        val all = loadAll()
        all.firstOrNull { it.id == id }?.let {
            it.text = newText.trim()
            saveAll(all)
        }
    }

    fun delete(id: String) {
        saveAll(loadAll().filterNot { it.id == id })
    }

    fun clear() {
        saveAll(emptyList())
    }

    /** Export all memories (including superseded) for backup. */
    fun exportJson(): String = JsonStore.readArray(file).toString()

    /**
     * Keyword-overlap retrieval (BM25-lite). Returns memories most relevant to
     * the query, most recent first on ties, never superseded ones.
     */
    fun retrieve(query: String, limit: Int = 3): List<Memory> {
        if (!enabled) return emptyList()
        val qTokens = tokenize(query)
        if (qTokens.isEmpty()) return emptyList()
        return list()
            .map { m ->
                val t = tokenize(m.text)
                val overlap = qTokens.count { it in t }
                m to (overlap * 100 + m.createdAt / 1_000_000_000L)
            }
            .filter { (_, score) -> score >= 100 }   // require at least one shared keyword
            .sortedByDescending { it.second }
            .take(limit)
            .map { it.first }
    }

    private fun subjectKey(text: String): String =
        tokenize(text).take(5).joinToString(" ")

    private fun tokenize(text: String): List<String> =
        text.lowercase().split(Regex("[^a-z0-9]+")).filter { it.length >= 3 }

    fun totalStoredBytes(): Long = if (file.exists()) file.length() else 0L
}
