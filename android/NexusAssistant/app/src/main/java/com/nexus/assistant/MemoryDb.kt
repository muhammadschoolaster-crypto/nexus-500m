package com.nexus.assistant

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

/**
 * NEXUS persistent memory subsystem (on-device only).
 * Stores conversations + separately-managed personal memories.
 * Memories are NEVER auto-ingested from chats: user approves via "remember:".
 * All data lives in local SQLite - exportable, deletable (Settings).
 */
class MemoryDb(context: Context) : SQLiteOpenHelper(context, "nexus_memory.db", null, 1) {

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("""
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""")
        db.execSQL("""
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )""")
        db.execSQL("""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                approved INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""")
    }

    override fun onUpgrade(db: SQLiteDatabase, o: Int, n: Int) {
        db.execSQL("DROP TABLE IF EXISTS messages")
        db.execSQL("DROP TABLE IF EXISTS conversations")
        db.execSQL("DROP TABLE IF EXISTS memories")
        onCreate(db)
    }

    // ---------- conversations ----------
    fun newConversation(title: String): Long =
        writableDatabase.insert("conversations", null, ContentValues().apply {
            put("title", title)
        })

    fun addMessage(convId: Long, role: String, content: String): Long =
        writableDatabase.insert("messages", null, ContentValues().apply {
            put("conversation_id", convId)
            put("role", role)
            put("content", content)
        })

    data class ChatMessage(val id: Long, val role: String, val content: String)

    fun loadMessages(convId: Long): List<ChatMessage> {
        val out = ArrayList<ChatMessage>()
        readableDatabase.rawQuery(
            "SELECT id, role, content FROM messages WHERE conversation_id=? ORDER BY id",
            arrayOf(convId.toString())).use { c ->
            while (c.moveToNext()) {
                out.add(ChatMessage(c.getLong(0), c.getString(1), c.getString(2)))
            }
        }
        return out
    }

    fun searchConversations(query: String): List<Pair<Long, String>> {
        val out = ArrayList<Pair<Long, String>>()
        readableDatabase.rawQuery(
            """SELECT c.id, c.title FROM conversations c
               WHERE c.title LIKE ? OR EXISTS (
                 SELECT 1 FROM messages m WHERE m.conversation_id = c.id AND m.content LIKE ?)""",
            arrayOf("%$query%", "%$query%")).use { c ->
            while (c.moveToNext()) out.add(Pair(c.getLong(0), c.getString(1)))
        }
        return out
    }

    // ---------- memories ----------
    fun saveMemory(kind: String, content: String): Long =
        writableDatabase.insert("memories", null, ContentValues().apply {
            put("kind", kind)
            put("content", content)
            put("approved", 1)
        })

    fun deleteMemory(id: Long) {
        writableDatabase.delete("memories", "id=?", arrayOf(id.toString()))
    }

    data class Memory(val id: Long, val kind: String, val content: String)

    fun allMemories(): List<Memory> {
        val out = ArrayList<Memory>()
        readableDatabase.rawQuery(
            "SELECT id, kind, content FROM memories WHERE approved=1 ORDER BY id DESC",
            null).use { c ->
            while (c.moveToNext()) out.add(Memory(c.getLong(0), c.getString(1), c.getString(2)))
        }
        return out
    }

    fun exportAll(): String {
        val sb = StringBuilder()
        sb.append("=== NEXUS memory export ===\n\nConversations:\n")
        readableDatabase.rawQuery(
            "SELECT id, title, created_at FROM conversations ORDER BY id", null).use { c ->
            while (c.moveToNext()) {
                sb.append("- [${c.getLong(0)}] ${c.getString(1)} (${c.getString(2)})\n")
            }
        }
        sb.append("\nMemories:\n")
        allMemories().forEach { sb.append("- (${it.kind}) ${it.content}\n") }
        return sb.toString()
    }

    fun wipeAll() {
        val db = writableDatabase
        db.execSQL("DELETE FROM messages")
        db.execSQL("DELETE FROM conversations")
        db.execSQL("DELETE FROM memories")
    }
}
