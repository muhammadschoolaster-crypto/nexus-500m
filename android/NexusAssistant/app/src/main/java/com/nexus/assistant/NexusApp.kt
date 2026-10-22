package com.nexus.assistant

import android.app.Application

class NexusApp : Application() {
    override fun onCreate() {
        super.onCreate()
        instance = this
    }

    companion object {
        lateinit var instance: NexusApp
            private set
    }
}
