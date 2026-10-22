package com.nexus500m.app.work

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.work.Constraints
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.Worker
import androidx.work.WorkerParameters
import java.util.concurrent.TimeUnit

/**
 * Background reminder via WorkManager — survives app restarts and reboots,
 * respects Doze/battery policies, and honestly reports if it cannot run
 * (Android may defer background work; reminders are best-effort, guaranteed
 * only while constraints allow).
 */
class ReminderWorker(context: Context, params: WorkerParameters) : Worker(context, params) {

    override fun doWork(): Result {
        val text = inputData.getString(KEY_TEXT) ?: return Result.failure()
        val id = inputData.getString(KEY_ID) ?: "reminder"
        return try {
            showNotification(applicationContext, text, id)
            Result.success()
        } catch (t: Throwable) {
            Result.retry()
        }
    }

    companion object {
        const val KEY_TEXT = "text"
        const val KEY_ID = "id"
        const val CHANNEL_ID = "nexus_reminders"

        fun showNotification(context: Context, text: String, id: String) {
            val nm = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            if (Build.VERSION.SDK_INT >= 26) {
                nm.createNotificationChannel(
                    NotificationChannel(CHANNEL_ID, "Reminders",
                        NotificationManager.IMPORTANCE_DEFAULT)
                )
            }
            val notif = NotificationCompat.Builder(context, CHANNEL_ID)
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .setContentTitle("NEXUS reminder")
                .setContentText(text)
                .setStyle(NotificationCompat.BigTextStyle().bigText(text))
                .setAutoCancel(true)
                .build()
            nm.notify(id.hashCode(), notif)
        }
    }
}

object ReminderScheduler {
    /** Schedule a reminder [minutesFromNow] minutes in the future (WorkManager). */
    fun schedule(context: Context, minutesFromNow: Long, text: String) {
        val request = OneTimeWorkRequestBuilder<ReminderWorker>()
            .setInitialDelay(minutesFromNow.coerceAtLeast(1L), TimeUnit.MINUTES)
            .setInputData(
                Data.Builder()
                    .putString(ReminderWorker.KEY_TEXT, text)
                    .putString(ReminderWorker.KEY_ID, "r-${System.currentTimeMillis()}")
                    .build()
            )
            .setConstraints(
                Constraints.Builder()
                    .setRequiredNetworkType(NetworkType.NOT_REQUIRED)
                    .build()
            )
            .addTag("nexus_reminder")
            .build()
        WorkManager.getInstance(context)
            .enqueueUniqueWork("nexus_reminder_${System.currentTimeMillis()}", ExistingWorkPolicy.KEEP, request)
    }

    /** Pending scheduled reminders (for the task history UI). */
    fun pendingCount(context: Context): Int =
        WorkManager.getInstance(context)
            .getWorkInfosByTag("nexus_reminder").get()
            .count { !it.state.isFinished }
}
