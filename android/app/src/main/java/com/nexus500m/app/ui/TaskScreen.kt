package com.nexus500m.app.ui

import android.Manifest
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.nexus500m.app.NexusApp
import com.nexus500m.app.agent.TaskEngine
import kotlinx.coroutines.launch

/**
 * Autonomous task screen: submit requests, watch the engine execute its plan
 * step by step, confirm risky steps, stop everything, review history.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TaskScreen(onBack: () -> Unit) {
    val app = NexusApp.get().appContext
    val scope = rememberCoroutineScope()
    var input by remember { mutableStateOf("") }
    var running by remember { mutableStateOf(false) }
    var currentTask by remember { mutableStateOf<TaskEngine.Task?>(null) }
    var pendingConfirmation by remember { mutableStateOf<String?>(null) }
    var confirmApprove by remember { mutableStateOf<(Boolean) -> Unit>({}) }
    var history by remember { mutableStateOf(app.taskEngine.history()) }
    val settings = app.settingsStore.load()

    // Notification permission (needed for reminders on Android 13+)
    val notifPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { }

    val gate = object : TaskEngine.ConfirmationGate {
        override suspend fun confirm(actionDescription: String): Boolean {
            return kotlinx.coroutines.suspendCancellableCoroutine { cont ->
                pendingConfirmation = actionDescription
                confirmApprove = { approved: Boolean ->
                    pendingConfirmation = null
                    if (cont.isActive) cont.resumeWith(Result.success(approved))
                }
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Autonomous tasks") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    if (running) {
                        IconButton(onClick = {
                            app.taskEngine.requestStop()
                        }) {
                            Icon(Icons.Filled.Stop, contentDescription = "Global stop - cancel the running task")
                        }
                    }
                },
            )
        },
    ) { padding ->
        Column(Modifier.fillMaxSize().padding(padding).padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        "Tasks run with your permissions only. Risky steps (like deleting a note) " +
                                "require your confirmation. A global stop button cancels anything.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier.fillMaxWidth()) {
                        OutlinedTextField(
                            value = input,
                            onValueChange = { input = it },
                            modifier = Modifier.weight(1f),
                            placeholder = { Text("e.g. calculate 25 * 4 + 10") },
                            enabled = !running,
                        )
                        IconButton(
                            onClick = {
                                if (input.isBlank()) return@IconButton
                                if (Build.VERSION.SDK_INT >= 33) {
                                    notifPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
                                }
                                running = true
                                val request = input
                                input = ""
                                scope.launch {
                                    app.taskEngine.run(
                                        request = request,
                                        confirmationGate = gate,
                                        webAllowed = settings.webResearchEnabled,
                                        onUpdate = { t -> currentTask = t },
                                    )
                                    running = false
                                    history = app.taskEngine.history()
                                }
                            },
                            enabled = !running,
                        ) {
                            Icon(Icons.Filled.PlayArrow, contentDescription = "Run task")
                        }
                    }
                    Text(
                        "Examples: 'calculate (12+8)*3' · 'create note shopping: buy rice' · " +
                                "'read note shopping' · 'remind me in 5 minutes to stretch' · " +
                                "'search my memories for project' · 'research gravity on wikipedia' (needs consent) · " +
                                "'delete note shopping' (needs confirmation)",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            currentTask?.let { task ->
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text("Status: ${task.status}", style = MaterialTheme.typography.titleSmall)
                        for (s in task.steps) {
                            Text(
                                "[${s.status}] ${s.description}" +
                                        (if (s.output.isBlank()) "" else "\n${s.output.take(400)}"),
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                    }
                }
            }

            Text("History", style = MaterialTheme.typography.titleMedium)
            LazyColumn(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                items(history, key = { it.id }) { t ->
                    Card(Modifier.fillMaxWidth()) {
                        Column(Modifier.padding(8.dp)) {
                            Text("[${t.status}] ${t.request}", style = MaterialTheme.typography.titleSmall)
                            Text("${t.steps.size} step(s)",
                                style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
        }
    }

    pendingConfirmation?.let { action ->
        AlertDialog(
            onDismissRequest = {},
            title = { Text("Confirm action") },
            text = { Text("The task engine wants to:\n\n$action\n\nApprove this?") },
            confirmButton = {
                Button(onClick = { confirmApprove(true) }) { Text("Approve") }
            },
            dismissButton = {
                TextButton(onClick = { confirmApprove(false) }) { Text("Refuse (cancels task)") }
            },
        )
    }
}
