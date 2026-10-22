package com.nexus500m.app

import com.nexus500m.app.data.ConversationStore
import com.nexus500m.app.data.MemoryStore
import com.nexus500m.app.data.ChatMessage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.util.UUID

/** Persistence round-trip tests for the local stores (JVM, temp dirs). */
class StoresTest {

    private fun tempFile(name: String): File =
        File(System.getProperty("java.io.tmpdir"), "nexus-test-${UUID.randomUUID()}")
            .apply { mkdirs() }
            .let { File(it, name) }

    @Test
    fun conversationPersistenceRoundTrip() {
        val file = tempFile("conversations.json")
        val store = ConversationStore(file)
        val conv = store.newConversation()
        store.append(conv.id, ChatMessage(UUID.randomUUID().toString(), "user", "hello there", 1L))
        store.append(conv.id, ChatMessage(UUID.randomUUID().toString(), "assistant", "hi!", 2L))

        // a NEW instance over the same file must see the same data
        val reloaded = ConversationStore(file)
        val list = reloaded.list()
        assertEquals(1, list.size)
        assertEquals("hello there", list[0].messages[0].content)
        assertEquals(2, list[0].messages.size)
        // auto-title from first user message
        assertEquals("hello there", list[0].title)
    }

    @Test
    fun conversationSearchRenameArchiveDelete() {
        val store = ConversationStore(tempFile("conversations.json"))
        val a = store.newConversation()
        val b = store.newConversation()
        store.append(a.id, ChatMessage(UUID.randomUUID().toString(), "user", "kotlin compose question", 1L))
        store.append(b.id, ChatMessage(UUID.randomUUID().toString(), "user", "python pandas question", 1L))

        assertEquals(1, store.search("kotlin").size)   // only conversation 'a' mentions kotlin
        store.rename(a.id, "Kotlin help")
        assertEquals("Kotlin help", store.get(a.id)!!.title)
        store.setArchived(a.id, true)
        assertEquals(1, store.list().size)
        assertEquals(2, store.list(includeArchived = true).size)
        store.delete(b.id)
        // 'a' was archived but NOT deleted: archive keeps the data, so it is still there
        assertEquals(1, store.list(includeArchived = true).size)
        store.delete(a.id)
        assertEquals(0, store.list(includeArchived = true).size)
    }

    @Test
    fun memoryStoreCrudAndRetrieval() {
        val store = MemoryStore(tempFile("memories.json"))
        store.add("My project is called nexus-500m")
        store.add("I prefer short answers")
        val hits = store.retrieve("which project am I working on? project nexus")
        assertTrue(hits.isNotEmpty())
        assertTrue(hits.any { it.text.contains("nexus-500m") })
        val all = store.list()
        assertEquals(2, all.size)
        store.update(all[0].id, "My project is called NEXUS (renamed)")
        assertEquals("My project is called NEXUS (renamed)", store.list().first { it.id == all[0].id }.text)
        store.delete(all[1].id)
        assertEquals(1, store.list().size)
    }

    @Test
    fun memoryContradictionMarksSuperseded() {
        val store = MemoryStore(tempFile("memories.json"))
        store.add("My favorite color is blue")
        store.add("My favorite color is green")
        val live = store.list()
        assertEquals(1, live.size)                 // older contradictory memory superseded
        assertEquals("My favorite color is green", live[0].text)
        assertEquals(2, store.list(includeSuperseded = true).size)
    }

    @Test
    fun memoryDisabledReturnsNothing() {
        val store = MemoryStore(tempFile("memories.json"))
        store.enabled = false
        store.add("offline model facts")
        assertTrue(store.retrieve("offline model").isEmpty())
    }
}
