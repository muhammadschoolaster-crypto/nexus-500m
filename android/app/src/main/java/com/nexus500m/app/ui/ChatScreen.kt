package com.nexus500m.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import com.nexus500m.app.NexusApp
import com.nexus500m.app.data.ChatMessage
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * The chat screen: streaming offline generation, stop / regenerate / copy,
 * markdown-lite rendering, memory-aware prompts.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    conversationId: String,
    onBack: () -> Unit,
) {
    val app = NexusApp.get().appContext
    val scope = rememberCoroutineScope()
    val clipboard = LocalClipboardManager.current
    val snackbar = remember { SnackbarHostState() }

    var conversation by remember { mutableStateOf(app.conversationStore.get(conversationId)) }
    var input by remember { mutableStateOf(TextFieldValue("")) }
    var generating by remember { mutableStateOf(false) }
    var streamText by remember { mutableStateOf("") }
    var genJob by remember { mutableStateOf<Job?>(null) }
    val listState = rememberLazyListState()

    LaunchedEffect(conversationId) {
        // background model load with clear error reporting
        if (!app.modelManager.isLoaded) {
            withContext(Dispatchers.IO) { app.modelManager.load() }
        }
        conversation = app.conversationStore.get(conversationId)
    }

    LaunchedEffect(streamText, conversation) {
        if ((conversation?.messages?.size ?: 0) > 0 || streamText.isNotEmpty()) {
            listState.animateScrollToItem((conversation?.messages?.size ?: 0) + 1)
        }
    }

    fun doSend(text: String, isNewTurn: Boolean = true) {
        val conv = conversation ?: return
        if (text.isBlank() || generating) return
        if (!app.modelManager.isLoaded) {
            scope.launch { snackbar.showSnackbar("Model not loaded yet — please wait a moment.") }
            return
        }
        val settings = app.settingsStore.load()
        val engine = app.modelManager.engine!!

        if (isNewTurn) {
            val userMsg = ChatMessage(UUID.randomUUID().toString(), "user", text.trim(), System.currentTimeMillis())
            app.conversationStore.append(conv.id, userMsg)
            conversation = app.conversationStore.get(conv.id)
        }

        // build prompt: memory context + short history + current instruction
        val memories = if (settings.memoryEnabled) {
            app.memoryStore.retrieve(text).joinToString("\n") { "- ${it.text}" }
        } else ""
        val history = (conversation ?: return).messages
            .filter { it.role == "user" || it.role == "assistant" }
            .takeLast(4)
            .joinToString("\n") { m ->
                if (m.role == "user") "### Instruction:\n${m.content.take(200)}\n### Response:\n"
                else "${m.content.take(200)}\n"
            }

        val prompt = buildString {
            if (memories.isNotBlank()) append("### Context:\n").append(memories).append("\n")
            if (history.isNotBlank()) append(history).append("\n")
            append("### Instruction:\n").append(text.trim()).append("\n### Response:\n")
        }

        val assistantId = UUID.randomUUID().toString()
        streamText = ""
        generating = true
        genJob = scope.launch(Dispatchers.Default) {
            try {
                val result = engine.generate(
                    prompt = prompt,
                    maxNewTokens = settings.maxNewTokens,
                    temperature = settings.temperature,
                    topK = settings.topK,
                    shouldStop = { false },
                    onText = { full ->
                        streamText = full
                    },
                )
                withContext(Dispatchers.Main) {
                    app.conversationStore.append(
                        conv.id,
                        ChatMessage(assistantId, "assistant", result, System.currentTimeMillis()),
                    )
                    conversation = app.conversationStore.get(conv.id)
                    streamText = ""
                }
            } catch (t: Throwable) {
                withContext(Dispatchers.Main) {
                    app.conversationStore.append(
                        conv.id,
                        ChatMessage(assistantId, "assistant",
                            "[generation failed: ${t.message ?: "unknown error"}]", System.currentTimeMillis()),
                    )
                    conversation = app.conversationStore.get(conv.id)
                    streamText = ""
                }
            } finally {
                withContext(Dispatchers.Main) { generating = false }
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(conversation?.title ?: "Chat", maxLines = 1) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    if (generating) {
                        IconButton(onClick = {
                            genJob?.cancel()
                            generating = false
                        }) {
                            Icon(Icons.Filled.Stop, contentDescription = "Stop generation")
                        }
                    }
                    IconButton(
                        onClick = {
                            val conv = conversation ?: return@IconButton
                            if (!generating && app.conversationStore.removeLastMessage(conv.id, "assistant")) {
                                conversation = app.conversationStore.get(conv.id)
                                val lastUser = conversation?.messages?.lastOrNull { it.role == "user" }?.content
                                if (lastUser != null) doSend(lastUser, isNewTurn = false)
                            }
                        },
                        enabled = !generating,
                    ) {
                        Icon(Icons.Filled.Refresh, contentDescription = "Regenerate last response")
                    }
                },
            )
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { padding ->
        Column(Modifier.fillMaxSize().padding(padding)) {
            LazyColumn(
                state = listState,
                modifier = Modifier.weight(1f).fillMaxWidth().padding(horizontal = 12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                val msgs = conversation?.messages ?: emptyList()
                items(msgs) { m ->
                    MessageBubble(
                        text = m.content,
                        isUser = m.role == "user",
                        onCopy = { clipboard.setText(AnnotatedString(m.content)) },
                    )
                }
                if (streamText.isNotEmpty() || generating) {
                    item {
                        MessageBubble(
                            text = streamText.ifBlank { " " },
                            isUser = false,
                            showSpinner = streamText.isEmpty(),
                        )
                    }
                }
            }

            Row(
                Modifier.fillMaxWidth().imePadding().padding(10.dp),
                verticalAlignment = Alignment.Bottom,
            ) {
                OutlinedTextField(
                    value = input,
                    onValueChange = { input = it },
                    modifier = Modifier.weight(1f),
                    placeholder = { Text(if (generating) "Generating (local model)..." else "Ask NEXUS anything (offline)") },
                    maxLines = 4,
                    enabled = !generating,
                )
                if (generating) {
                    CircularProgressIndicator(Modifier.padding(start = 10.dp).align(Alignment.CenterVertically))
                } else {
                    IconButton(onClick = {
                        val text = input.text
                        input = TextFieldValue("")
                        doSend(text)
                    }) {
                        Icon(Icons.AutoMirrored.Filled.Send, contentDescription = "Send message")
                    }
                }
            }
        }
    }
}

@Composable
private fun MessageBubble(
    text: String,
    isUser: Boolean,
    onCopy: (() -> Unit)? = null,
    showSpinner: Boolean = false,
) {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
    ) {
        Column(
            Modifier
                .widthIn(max = 320.dp)
                .background(
                    color = if (isUser) MaterialTheme.colorScheme.primary.copy(alpha = 0.15f)
                    else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f),
                    shape = androidx.compose.foundation.shape.RoundedCornerShape(
                        topStart = 14.dp, topEnd = 14.dp,
                        bottomStart = if (isUser) 14.dp else 4.dp,
                        bottomEnd = if (isUser) 4.dp else 14.dp,
                    ),
                )
                .padding(10.dp),
        ) {
            if (showSpinner) {
                Text("thinking...", style = MaterialTheme.typography.bodySmall)
            } else {
                MarkdownText(text)
            }
            if (onCopy != null && !isUser && text.isNotBlank() && !showSpinner) {
                TextButton(onClick = onCopy, contentPadding = androidx.compose.foundation.layout.PaddingValues(0.dp)) {
                    Icon(Icons.Filled.ContentCopy, contentDescription = null, Modifier.padding(end = 4.dp))
                    Text("copy", style = MaterialTheme.typography.labelSmall)
                }
            }
        }
    }
}
