package com.nexus500m.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import com.nexus500m.app.ui.ChatScreen
import com.nexus500m.app.ui.ConversationsScreen
import com.nexus500m.app.ui.NexusTheme
import com.nexus500m.app.ui.SettingsScreen
import com.nexus500m.app.ui.TaskScreen

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val app = NexusApp.get().appContext
            val settings = remember { app.settingsStore.load() }
            NexusTheme(darkTheme = settings.darkTheme) {
                AppNav()
            }
        }
    }
}

/** Minimal screen routing (no navigation dependency needed at this size). */
@Composable
private fun AppNav() {
    var screen by remember { mutableStateOf("chats") }
    var openChatId by remember { mutableStateOf<String?>(null) }

    when (screen) {
        "chats" -> ConversationsScreen(
            onOpenChat = { id -> openChatId = id; screen = "chat" },
            onOpenSettings = { screen = "settings" },
            onOpenTasks = { screen = "tasks" },
        )
        "chat" -> ChatScreen(
            conversationId = openChatId ?: return@AppNav,
            onBack = { screen = "chats" },
        )
        "settings" -> SettingsScreen(onBack = { screen = "chats" })
        "tasks" -> TaskScreen(onBack = { screen = "chats" })
    }
}
