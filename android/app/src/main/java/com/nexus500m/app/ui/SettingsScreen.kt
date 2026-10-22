package com.nexus500m.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.nexus500m.app.NexusApp

/**
 * Settings: model status, memory management, storage management, agent and
 * web-research consent switches, generation parameters, theme.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(onBack: () -> Unit) {
    val app = NexusApp.get().appContext
    var settings by remember { mutableStateOf(app.settingsStore.load()) }
    var modelStatus by remember { mutableStateOf(app.modelManager.load()) }
    var usage by remember { mutableStateOf(app.storage.usage()) }
    var memories by remember { mutableStateOf(app.memoryStore.list(includeSuperseded = false)) }
    var newMemory by remember { mutableStateOf("") }
    var editingMemory by remember { mutableStateOf<String?>(null) }
    var confirmDeleteAll by remember { mutableStateOf(false) }

    fun persist(new: com.nexus500m.app.data.SettingsStore.Settings) {
        settings = new
        app.settingsStore.save(new)
        app.memoryStore.enabled = new.memoryEnabled
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
            )
        },
    ) { padding ->
        Column(
            Modifier.fillMaxSize().padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            // ---------------- model status
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("Local model", style = MaterialTheme.typography.titleMedium)
                    if (modelStatus.error != null) {
                        Text("ERROR: ${modelStatus.error}", color = MaterialTheme.colorScheme.error)
                    }
                    Text("name: ${modelStatus.modelName} v${modelStatus.modelVersion}")
                    Text("parameters: ${modelStatus.parameterCount}")
                    Text("vocabulary: ${modelStatus.vocabSize} tokens · context: ${modelStatus.ctxLen} tokens")
                    Text("file: ${modelStatus.fileBytes / 1024} KB · RAM estimate: ${modelStatus.ramEstimateBytes / 1024 / 1024} MB")
                    Text("SHA-256: ${modelStatus.fileSha256.take(16)}…")
                    Text("integrity: ${com.nexus500m.app.model.ModelManager.lastIntegrityNote.ifEmpty { "checked at load" }}")
                    Text("checkpoint step: ${modelStatus.checkpointStep}", style = MaterialTheme.typography.bodySmall)
                    Text(
                        "Note: this is the tiny demo-scale NEXUS checkpoint (trained on a small corpus). " +
                                "It runs fully offline; it is not a large cloud model.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    TextButton(onClick = { modelStatus = app.modelManager.load() }) { Text("Reload model") }
                }
            }

            // ---------------- privacy / capabilities
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("Privacy & capabilities", style = MaterialTheme.typography.titleMedium)
                    ListItem(
                        headlineContent = { Text("Personal memory") },
                        supportingContent = { Text("Reminders of your facts, retrieved into prompts. Not retraining. Fully under your control.") },
                        trailingContent = {
                            Switch(checked = settings.memoryEnabled, onCheckedChange = {
                                persist(settings.copy(memoryEnabled = it))
                            })
                        },
                    )
                    ListItem(
                        headlineContent = { Text("Autonomous tasks") },
                        supportingContent = { Text("Permission-gated task engine with confirmation for risky steps.") },
                        trailingContent = {
                            Switch(checked = settings.agentEnabled, onCheckedChange = {
                                persist(settings.copy(agentEnabled = it))
                            })
                        },
                    )
                    ListItem(
                        headlineContent = { Text("Web research (optional)") },
                        supportingContent = { Text("OFF by default. When on, tasks may query the public Wikipedia API. Retrieved text is treated as data, never instructions.") },
                        trailingContent = {
                            Switch(checked = settings.webResearchEnabled, onCheckedChange = {
                                persist(settings.copy(webResearchEnabled = it))
                            })
                        },
                    )
                }
            }

            // ---------------- generation
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("Generation", style = MaterialTheme.typography.titleMedium)
                    Text("Max new tokens: ${settings.maxNewTokens}")
                    Slider(
                        value = settings.maxNewTokens.toFloat(),
                        onValueChange = { persist(settings.copy(maxNewTokens = it.toInt().coerceIn(16, 200))) },
                        valueRange = 16f..200f,
                    )
                    Text("Temperature: %.2f".format(settings.temperature))
                    Slider(
                        value = settings.temperature,
                        onValueChange = { persist(settings.copy(temperature = it)) },
                        valueRange = 0.1f..1.2f,
                    )
                    TextButton(onClick = { persist(settings.copy(darkTheme = !settings.darkTheme)) }) {
                        Text(if (settings.darkTheme) "Theme: dark (tap for light)" else "Theme: light (tap for dark)")
                    }
                }
            }

            // ---------------- memory management
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("Memories", style = MaterialTheme.typography.titleMedium)
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        OutlinedTextField(
                            value = newMemory,
                            onValueChange = { newMemory = it },
                            modifier = Modifier.weight(1f),
                            placeholder = { Text("Add a fact to remember...") },
                        )
                        TextButton(onClick = {
                            if (newMemory.isNotBlank()) {
                                app.memoryStore.add(newMemory, source = "user")
                                newMemory = ""
                                memories = app.memoryStore.list()
                            }
                        }) { Text("Add") }
                    }
                    for (m in memories.take(30)) {
                        ListItem(
                            headlineContent = { Text(m.text, style = MaterialTheme.typography.bodyMedium) },
                            supportingContent = { Text("from: ${m.source}") },
                            trailingContent = {
                                Row {
                                    TextButton(onClick = { editingMemory = m.id }) { Text("Edit") }
                                    TextButton(onClick = {
                                        app.memoryStore.delete(m.id)
                                        memories = app.memoryStore.list()
                                    }) { Text("Delete") }
                                }
                            },
                        )
                    }
                    if (memories.isEmpty()) {
                        Text("No memories yet. Everything you add is stored only on this device.",
                            style = MaterialTheme.typography.bodySmall)
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        TextButton(onClick = {
                            // export backup inside app sandbox; user can copy it out via device files
                            val target = java.io.File(app.filesDir, "exports/nexus-backup.zip")
                            val ok = app.storage.exportBackup(target)
                            memories = memories // refresh UI noop
                        }) { Text("Export backup (device file)") }
                        TextButton(onClick = { confirmDeleteAll = true }) { Text("Delete ALL memories") }
                    }
                }
            }

            // ---------------- storage
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("Storage", style = MaterialTheme.typography.titleMedium)
                    Text("Model: ${usage.modelBytes / 1024} KB")
                    Text("Conversations: ${usage.conversationsBytes / 1024} KB")
                    Text("Memories: ${usage.memoriesBytes / 1024} KB")
                    Text("Notes: ${usage.notesBytes / 1024} KB · Reports: ${usage.reportsBytes / 1024} KB")
                    Text("Task history: ${usage.taskHistoryBytes / 1024} KB · Caches: ${usage.cacheBytes / 1024} KB")
                    Text("Total app data: ${usage.totalBytes / 1024} KB · Free on device: ${app.storage.availableBytes() / 1024 / 1024} MB")
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        TextButton(onClick = {
                            app.storage.cleanup()
                            usage = app.storage.usage()
                        }) { Text("Clean caches (safe)") }
                    }
                    Text("Cleanup only removes caches and temporary files. Your conversations, memories, notes and the model are never touched without explicit confirmation.",
                        style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }

    editingMemory?.let { id ->
        val current = app.memoryStore.list(true).firstOrNull { it.id == id }?.text ?: ""
        var value by remember(id) { mutableStateOf(current) }
        AlertDialog(
            onDismissRequest = { editingMemory = null },
            title = { Text("Edit memory") },
            text = { OutlinedTextField(value = value, onValueChange = { value = it }) },
            confirmButton = {
                TextButton(onClick = {
                    app.memoryStore.update(id, value)
                    editingMemory = null
                    memories = app.memoryStore.list()
                }) { Text("Save") }
            },
            dismissButton = { TextButton(onClick = { editingMemory = null }) { Text("Cancel") } },
        )
    }
    if (confirmDeleteAll) {
        AlertDialog(
            onDismissRequest = { confirmDeleteAll = false },
            title = { Text("Delete ALL memories?") },
            text = { Text("This permanently removes every stored memory from this device. Conversations are not affected.") },
            confirmButton = {
                TextButton(onClick = {
                    app.memoryStore.clear()
                    confirmDeleteAll = false
                    memories = app.memoryStore.list()
                }) { Text("Delete all") }
            },
            dismissButton = { TextButton(onClick = { confirmDeleteAll = false }) { Text("Cancel") } },
        )
    }
}
