package com.nexus500m.app.data

import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** One chat message. */
data class ChatMessage(
    val id: String,
    val role: String,        // "user" | "assistant" | "system"
    val content: String,
    val timestamp: Long,
)

/** A conversation with its message list. */
data class Conversation(
    val id: String,
    var title: String,
    val createdAt: Long,
    var updatedAt: Long,
    var archived: Boolean,
    var messages: MutableList<ChatMessage>,
)

/**
 * Conversation persistence: JSON file in the app sandbox. Supports
 * search, rename, archive, delete, append — all atomic-write safe.
 */
class ConversationStore(private val file: File) {

    private fun loadAll(): MutableList<Conversation> {
        val arr = JsonStore.readArray(file)
        val list = ArrayList<Conversation>(arr.length())
        for (i in 0 until arr.length()) {
            val c = arr.getJSONObject(i)
            val msgArr = c.optJSONArray("messages") ?: JSONArray()
            val messages = ArrayList<ChatMessage>(msgArr.length())
            for (j in 0 until msgArr.length()) {
                val m = msgArr.getJSONObject(j)
                messages.add(
                    ChatMessage(
                        id = m.getString("id"),
                        role = m.getString("role"),
                        content = m.getString("content"),
                        timestamp = m.getLong("timestamp"),
                    )
                )
            }
            list.add(
                Conversation(
                    id = c.getString("id"),
                    title = c.getString("title"),
                    createdAt = c.getLong("createdAt"),
                    updatedAt = c.getLong("updatedAt"),
                    archived = c.optBoolean("archived", false),
                    messages = messages,
                )
            )
        }
        return list
    }

    private fun saveAll(list: List<Conversation>) {
        val arr = JSONArray()
        for (c in list) {
            val msgs = JSONArray()
            for (m in c.messages) {
                msgs.put(JSONObject()
                    .put("id", m.id)
                    .put("role", m.role)
                    .put("content", m.content)
                    .put("timestamp", m.timestamp))
            }
            arr.put(JSONObject()
                .put("id", c.id)
                .put("title", c.title)
                .put("createdAt", c.createdAt)
                .put("updatedAt", c.updatedAt)
                .put("archived", c.archived)
                .put("messages", msgs))
        }
        JsonStore.writeArray(file, arr)
    }

    fun newConversation(title: String = "New chat"): Conversation {
        val all = loadAll()
        val now = System.currentTimeMillis()
        val c = Conversation(
            id = java.util.UUID.randomUUID().toString(),
            title = title, createdAt = now, updatedAt = now,
            archived = false, messages = ArrayList(),
        )
        all.add(c)
        saveAll(all)
        return c
    }

    fun get(id: String): Conversation? = loadAll().firstOrNull { it.id == id }

    fun list(includeArchived: Boolean = false): List<Conversation> =
        loadAll().filter { includeArchived || !it.archived }
            .sortedByDescending { it.updatedAt }

    fun search(query: String, includeArchived: Boolean = true): List<Conversation> {
        if (query.isBlank()) return list(includeArchived)
        val q = query.trim().lowercase()
        return loadAll().filter { c ->
            (!c.archived || includeArchived) &&
                (c.title.lowercase().contains(q) ||
                    c.messages.any { it.content.lowercase().contains(q) })
        }.sortedByDescending { it.updatedAt }
    }

    fun rename(id: String, newTitle: String) {
        val all = loadAll()
        all.firstOrNull { it.id == id }?.let {
            it.title = newTitle.trim().ifEmpty { it.title }
            it.updatedAt = System.currentTimeMillis()
            saveAll(all)
        }
    }

    fun setArchived(id: String, archived: Boolean) {
        val all = loadAll()
        all.firstOrNull { it.id == id }?.let {
            it.archived = archived
            it.updatedAt = System.currentTimeMillis()
            saveAll(all)
        }
    }

    fun delete(id: String) {
        saveAll(loadAll().filterNot { it.id == id })
    }

    fun append(conversationId: String, message: ChatMessage) {
        val all = loadAll()
        all.firstOrNull { it.id == conversationId }?.let {
            it.messages.add(message)
            it.updatedAt = System.currentTimeMillis()
            // auto-title from the first user message
            if (it.title == "New chat" && message.role == "user") {
                it.title = message.content.trim().take(40).ifEmpty { it.title }
            }
            saveAll(all)
        }
    }

    fun removeLastMessage(conversationId: String, role: String): Boolean {
        val all = loadAll()
        val conv = all.firstOrNull { it.id == conversationId } ?: return false
        val idx = conv.messages.indexOfLast { it.role == role }
        if (idx < 0) return false
        conv.messages.removeAt(idx)
        conv.updatedAt = System.currentTimeMillis()
        saveAll(all)
        return true
    }

    fun updateMessage(conversationId: String, messageId: String, content: String) {
        val all = loadAll()
        val conv = all.firstOrNull { it.id == conversationId } ?: return
        val idx = conv.messages.indexOfFirst { it.id == messageId }
        if (idx >= 0) {
            conv.messages[idx] = conv.messages[idx].copy(content = content)
            conv.updatedAt = System.currentTimeMillis()
            saveAll(all)
        }
    }

    fun totalStoredBytes(): Long = if (file.exists()) file.length() else 0L
}
