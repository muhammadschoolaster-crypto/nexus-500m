package com.nexus500m.app

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.os.Build
import androidx.work.Configuration
import java.io.File

/**
 * Application entry point: owns the WorkManager configuration, the
 * notification channel, and the lazy singletons (manual DI — no framework
 * needed at this size).
 */
class NexusApp : Application(), Configuration.Provider {

    lateinit var appContext: AppState
        private set

    override fun onCreate() {
        super.onCreate()
        instance = this
        appContext = AppState(this)
        createNotificationChannel()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= 26) {
            val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
            nm.createNotificationChannel(
                NotificationChannel(
                    "nexus_reminders",
                    "Reminders",
                    NotificationManager.IMPORTANCE_DEFAULT,
                ).apply { description = "Reminders scheduled by NEXUS tasks" }
            )
        }
    }

    override val workManagerConfiguration: Configuration
        get() = Configuration.Builder().build()

    companion object {
        private lateinit var instance: NexusApp
        fun get(): NexusApp = instance
    }
}

/**
 * App-wide singletons. Everything is local to the app sandbox.
 */
class AppState(private val app: NexusApp) {
    val filesDir: File get() = app.filesDir

    val conversationStore by lazy {
        com.nexus500m.app.data.ConversationStore(File(app.filesDir, "conversations.json"))
    }
    val memoryStore by lazy {
        com.nexus500m.app.data.MemoryStore(File(app.filesDir, "memories.json"))
    }
    val settingsStore by lazy {
        com.nexus500m.app.data.SettingsStore(File(app.filesDir, "settings.json"))
    }
    val modelManager by lazy { com.nexus500m.app.model.ModelManager(app) }
    val taskEngine by lazy {
        com.nexus500m.app.agent.TaskEngine(app, modelManager, memoryStore)
    }
    val storage by lazy { com.nexus500m.app.util.StorageManager(app) }
}
