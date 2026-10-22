package com.nexus500m.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Archive
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.TaskAlt
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
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
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.nexus500m.app.NexusApp
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Conversation history: list, search, new, rename, archive, delete.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConversationsScreen(
    onOpenChat: (String) -> Unit,
    onOpenSettings: () -> Unit,
    onOpenTasks: () -> Unit,
) {
    val app = NexusApp.get().appContext
    var query by remember { mutableStateOf("") }
    var showArchived by remember { mutableStateOf(false) }
    var renaming by remember { mutableStateOf<String?>(null) }
    var deleting by remember { mutableStateOf<String?>(null) }
    var refresh by remember { mutableStateOf(0) }
    val fmt = remember { SimpleDateFormat("dd MMM HH:mm", Locale.getDefault()) }

    val convs = remember(query, showArchived, refresh) {
        if (query.isBlank()) app.conversationStore.list(includeArchived = showArchived)
        else app.conversationStore.search(query, includeArchived = true)
    }

    fun bump() { refresh += 1 }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("NEXUS") },
                actions = {
                    IconButton(onClick = onOpenTasks) {
                        Icon(Icons.Filled.TaskAlt, contentDescription = "Autonomous tasks")
                    }
                    IconButton(onClick = onOpenSettings) {
                        Icon(Icons.Filled.Settings, contentDescription = "Settings")
                    }
                },
            )
        },
        floatingActionButton = {
            androidx.compose.material3.ExtendedFloatingActionButton(
                onClick = {
                    val conv = app.conversationStore.newConversation()
                    bump()
                    onOpenChat(conv.id)
                },
                icon = { Icon(Icons.Filled.Add, contentDescription = null) },
                text = { Text("New chat") },
            )
        },
    ) { padding ->
        Column(Modifier.fillMaxSize().padding(padding)) {
            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp),
                placeholder = { Text("Search conversations...") },
                singleLine = true,
            )
            TextButton(
                onClick = { showArchived = !showArchived; bump() },
                modifier = Modifier.padding(horizontal = 12.dp),
            ) {
                Text(if (showArchived) "Hide archived" else "Show archived")
            }
            LazyColumn(
                Modifier.fillMaxSize().padding(horizontal = 8.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                items(convs, key = { it.id }) { c ->
                    ListItem(
                        headlineContent = { Text(c.title, maxLines = 1) },
                        supportingContent = {
                            Text("${c.messages.size} messages · ${fmt.format(Date(c.updatedAt))}" +
                                    if (c.archived) " · archived" else "",
                                style = MaterialTheme.typography.bodySmall)
                        },
                        trailingContent = {
                            Row {
                                IconButton(onClick = { renaming = c.id }) {
                                    Icon(Icons.Filled.Edit, contentDescription = "Rename conversation")
                                }
                                IconButton(onClick = {
                                    app.conversationStore.setArchived(c.id, !c.archived); bump()
                                }) {
                                    Icon(Icons.Filled.Archive, contentDescription = "Archive conversation")
                                }
                                IconButton(onClick = { deleting = c.id }) {
                                    Icon(Icons.Filled.Delete, contentDescription = "Delete conversation")
                                }
                            }
                        },
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            }
        }
    }

    renaming?.let { id ->
        RenameDialog(
            initial = app.conversationStore.get(id)?.title ?: "",
            onConfirm = { newName ->
                app.conversationStore.rename(id, newName)
                renaming = null
                bump()
            },
            onDismiss = { renaming = null },
        )
    }
    deleting?.let { id ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("Delete this conversation?") },
            text = { Text("This permanently removes the conversation and its messages from this device. This cannot be undone.") },
            confirmButton = {
                TextButton(onClick = {
                    app.conversationStore.delete(id)
                    deleting = null
                    bump()
                }) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun RenameDialog(initial: String, onConfirm: (String) -> Unit, onDismiss: () -> Unit) {
    var value by remember { mutableStateOf(initial) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Rename conversation") },
        text = {
            OutlinedTextField(value = value, onValueChange = { value = it }, singleLine = true)
        },
        confirmButton = {
            TextButton(onClick = { onConfirm(value) }, enabled = value.isNotBlank()) { Text("Save") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}
