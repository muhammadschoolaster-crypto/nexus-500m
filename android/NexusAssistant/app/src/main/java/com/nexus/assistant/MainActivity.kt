package com.nexus.assistant

import android.content.DialogInterface
import android.os.Bundle
import android.text.InputType
import android.view.View
import android.widget.EditText
import android.widget.ImageButton
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity(), CoroutineScope by MainScope() {

    private lateinit var recycler: RecyclerView
    private lateinit var adapter: ChatAdapter
    private lateinit var input: EditText
    private lateinit var sendBtn: ImageButton
    private lateinit var statusText: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var db: MemoryDb

    private var engine: NexusEngine? = null
    private var conversationId: Long = -1
    private var mode: Int = R.id.modeChat
    private var temperature = 0.8f
    private var topK = 40

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        recycler = findViewById(R.id.recycler)
        adapter = ChatAdapter()
        recycler.layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        recycler.adapter = adapter
        input = findViewById(R.id.input)
        sendBtn = findViewById(R.id.sendBtn)
        statusText = findViewById(R.id.statusText)
        progressBar = findViewById(R.id.progressBar)
        db = MemoryDb(this)

        findViewById<ImageButton>(R.id.newChatBtn).setOnClickListener { newConversation() }
        findViewById<ImageButton>(R.id.searchBtn).setOnClickListener { showSearch() }
        findViewById<ImageButton>(R.id.memoriesBtn).setOnClickListener { showMemories() }
        findViewById<ImageButton>(R.id.infoBtn).setOnClickListener { showModelInfo() }

        // mode toggle
        val chatMode = findViewById<TextView>(R.id.modeChat)
        val codeMode = findViewById<TextView>(R.id.modeCode)
        val writeMode = findViewById<TextView>(R.id.modeWrite)
        fun select(m: TextView) {
            mode = m.id
            for (t in listOf(chatMode, codeMode, writeMode)) {
                val active = t.id == mode
                t.isActivated = active
            }
        }
        chatMode.setOnClickListener { select(chatMode) }
        codeMode.setOnClickListener { select(codeMode) }
        writeMode.setOnClickListener { select(writeMode) }
        select(chatMode)

        sendBtn.setOnClickListener { sendMessage() }

        newConversation()
        loadEngine()
    }

    private fun loadEngine() {
        statusText.text = getString(R.string.loading_model)
        progressBar.visibility = View.VISIBLE
        launch {
            engine = withContext(Dispatchers.Default) { NexusEngine.load(this@MainActivity) }
            progressBar.visibility = View.GONE
            val e = engine
            if (e == null) {
                statusText.text = getString(R.string.model_load_failed)
            } else {
                statusText.text = getString(R.string.ready_status, e.modelName, e.paramCount)
            }
        }
    }

    private fun newConversation() {
        conversationId = db.newConversation("Chat " + System.currentTimeMillis() % 100000)
        adapter.setAll(emptyList())
        addAssistant(getString(R.string.welcome_message))
    }

    private fun addAssistant(text: String) {
        adapter.add(ChatAdapter.Role.ASSISTANT, text)
        db.addMessage(conversationId, "assistant", text)
    }

    private fun sendMessage() {
        val e = engine ?: run { addAssistant(getString(R.string.model_not_ready)); return }
        val text = input.text.toString().trim()
        if (text.isEmpty()) return
        input.setText("")
        adapter.add(ChatAdapter.Role.USER, text)
        db.addMessage(conversationId, "user", text)
        recycler.scrollToPosition(adapter.itemCount - 1)

        // memory command: remember this
        if (text.startsWith("remember:", ignoreCase = true)) {
            val mem = text.substringAfter(":").trim()
            if (mem.isNotEmpty()) {
                db.saveMemory("personal", mem)
                addAssistant(getString(R.string.memory_saved, mem))
            }
            return
        }

        statusText.text = getString(R.string.generating)
        progressBar.visibility = View.VISIBLE
        sendBtn.isEnabled = false
        adapter.add(ChatAdapter.Role.ASSISTANT, "")
        val assistantSlot = adapter.itemCount - 1

        launch {
            val reply = withContext(Dispatchers.Default) {
                val prompt = buildPrompt(text)
                e.append(e.tokenizer.encode(prompt))
                val acc = StringBuilder()
                e.generate(
                    maxNew = 60, temperature = temperature, topK = topK
                ) { delta ->
                    acc.append(delta)
                    runOnUiThread { adapter.updateLast(acc.toString()) }
                }
                acc.toString()
            }
            progressBar.visibility = View.GONE
            sendBtn.isEnabled = true
            val cleaned = reply.trim().ifEmpty { getString(R.string.empty_reply) }
            adapter.updateLast(cleaned)
            db.addMessage(conversationId, "assistant", cleaned)
            recycler.scrollToPosition(adapter.itemCount - 1)
            statusText.text = getString(R.string.ready_status, e.modelName, e.paramCount)
        }
    }

    /** Builds the text prompt with domain tags matching training corpus format. */
    private fun buildPrompt(userText: String): String {
        val tag = when (mode) {
            R.id.modeCode -> "<|code:python|>"
            R.id.modeWrite -> "<|write:email|>"
            else -> "<|text|>"
        }
        val mems = db.allMemories().takeLast(3).joinToString("\n") { "- ${it.content}" }
        val memBlock = if (mems.isEmpty()) "" else "Personal memories:\n$mems\n"
        return "\n$tag\n$memBlock$userText\n"
    }

    // ---------- dialogs ----------
    private fun showModelInfo() {
        val e = engine
        val msg = if (e == null) getString(R.string.model_not_ready)
        else getString(R.string.model_info_body, e.modelName, e.paramCount,
            e.tokenizer.vocabSize, e.contextLength())
        AlertDialog.Builder(this)
            .setTitle(R.string.model_info_title)
            .setMessage(msg)
            .setPositiveButton(android.R.string.ok, null)
            .setNegativeButton(R.string.export) { _, _ ->
                launch {
                    val out = withContext(Dispatchers.Default) { db.exportAll() }
                    openFileOutput("nexus_memory_export.txt", MODE_PRIVATE).use {
                        it.write(out.toByteArray())
                    }
                    addAssistant(getString(R.string.exported_to,
                        "${filesDir.absolutePath}/nexus_memory_export.txt"))
                }
            }
            .show()
    }

    private fun showMemories() {
        val mems = db.allMemories()
        val items = mems.map { "(${it.kind}) ${it.content}" }.toTypedArray()
        val ids = mems.map { it.id }.toLongArray()
        AlertDialog.Builder(this)
            .setTitle(R.string.memories_title)
            .setItems(if (items.isEmpty()) arrayOf(getString(R.string.no_memories)) else items) { d, which ->
                if (items.isNotEmpty()) {
                    AlertDialog.Builder(this)
                        .setTitle(R.string.delete_memory_q)
                        .setPositiveButton(android.R.string.yes) { _, _ -> db.deleteMemory(ids[which]); showMemories() }
                        .setNegativeButton(android.R.string.no, null)
                        .show()
                }
            }
            .setNegativeButton(R.string.wipe_all) { _, _ ->
                AlertDialog.Builder(this)
                    .setTitle(R.string.wipe_confirm)
                    .setPositiveButton(android.R.string.yes) { _, _ -> db.wipeAll(); newConversation() }
                    .setNegativeButton(android.R.string.no, null)
                    .show()
            }
            .show()
    }

    private fun showSearch() {
        val et = EditText(this).apply { inputType = InputType.TYPE_CLASS_TEXT }
        AlertDialog.Builder(this)
            .setTitle(R.string.search_title)
            .setView(et)
            .setPositiveButton(android.R.string.search_go) { _, _ ->
                val q = et.text.toString().trim()
                if (q.isEmpty()) return@setPositiveButton
                val res = db.searchConversations(q)
                val names = res.map { it.second }.toTypedArray()
                AlertDialog.Builder(this)
                    .setTitle(getString(R.string.search_results, res.size))
                    .setItems(names) { d, which ->
                        val conv = res[which].first
                        val msgs = db.loadMessages(conv)
                        adapter.setAll(msgs.map {
                            ChatAdapter.Item(
                                if (it.role == "user") ChatAdapter.Role.USER else ChatAdapter.Role.ASSISTANT,
                                it.content)
                        })
                    }
                    .setPositiveButton(android.R.string.ok, null)
                    .show()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }
}
