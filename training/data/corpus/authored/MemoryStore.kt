package com.nexus.assistant

/* Minimal on-device memory store example (Kotlin) */
class MemoryStore(private val db: MemDb) {
    fun save(kind: String, content: String): Long =
        db.insert("memories", mapOf("kind" to kind, "content" to content))

    fun search(query: String): List<String> =
        db.query("SELECT content FROM memories WHERE content LIKE '%' || ? || '%'", query)
            .map { it.getString("content") }

    fun approve(id: Long) { db.update("memories", id, mapOf("approved" to 1)) }
}
