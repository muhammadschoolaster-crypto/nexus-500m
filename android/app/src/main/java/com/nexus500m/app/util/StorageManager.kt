package com.nexus500m.app.util

import android.content.Context
import android.os.StatFs
import java.io.File

/**
 * Storage management. Honest rules:
 * - Reports real usage by category (model, conversations, memories, notes,
 *   reports, task history, caches).
 * - Cleanup touches ONLY caches and temp files — never conversations,
 *   memories, notes or the model without explicit user action.
 * - Storage capacity is a physical fact; the app can only help you use it.
 */
class StorageManager(private val context: Context) {

    data class Usage(
        val modelBytes: Long,
        val conversationsBytes: Long,
        val memoriesBytes: Long,
        val notesBytes: Long,
        val reportsBytes: Long,
        val taskHistoryBytes: Long,
        val cacheBytes: Long,
        val totalBytes: Long,
    )

    fun usage(): Usage {
        fun dirSize(f: File?): Long = f?.walkBottomUp()?.filter { it.isFile }?.sumOf { it.length() } ?: 0L
        val files = context.filesDir
        val cache = context.cacheDir
        val model = dirSize(File(files, "models"))
        val conv = File(files, "conversations.json").length().takeIf { it >= 0 } ?: 0L
        val mem = File(files, "memories.json").length().takeIf { it >= 0 } ?: 0L
        val u = Usage(
            modelBytes = model,
            conversationsBytes = if (File(files, "conversations.json").exists()) conv else 0L,
            memoriesBytes = if (File(files, "memories.json").exists()) mem else 0L,
            notesBytes = dirSize(File(files, "notes")),
            reportsBytes = dirSize(File(files, "reports")),
            taskHistoryBytes = if (File(files, "tasks.json").exists()) File(files, "tasks.json").length() else 0L,
            cacheBytes = dirSize(cache),
            totalBytes = 0L,
        )
        return u.copy(totalBytes = model + u.conversationsBytes + u.memoriesBytes + u.notesBytes +
                u.reportsBytes + u.taskHistoryBytes + u.cacheBytes)
    }

    /** Free space on the app's storage volume. */
    fun availableBytes(): Long = try {
        StatFs(context.filesDir.absolutePath).availableBytes
    } catch (t: Throwable) {
        0L
    }

    /** Safe cleanup: caches and temp files ONLY. Returns bytes freed. */
    fun cleanup(): Long {
        var freed = 0L
        context.cacheDir?.walkBottomUp()?.forEach { f ->
            if (f.isFile) { val l = f.length(); if (f.delete()) freed += l }
        }
        // half-written stores from crashes
        for (name in listOf("conversations.json.tmp", "memories.json.tmp", "tasks.json.tmp", "settings.json.tmp")) {
            val f = File(context.filesDir, name)
            if (f.exists() && f.delete()) freed += f.length()
        }
        return freed
    }

    /** Export a backup bundle of user data (never the model — it ships inside the APK). */
    fun exportBackup(target: File): Boolean = try {
        val files = context.filesDir
        target.parentFile?.mkdirs()
        java.util.zip.ZipOutputStream(target.outputStream().buffered()).use { zip ->
            for (name in listOf("conversations.json", "memories.json", "tasks.json", "settings.json")) {
                val f = File(files, name)
                if (f.exists()) {
                    zip.putNextEntry(java.util.zip.ZipEntry(name))
                    f.inputStream().use { it.copyTo(zip) }
                    zip.closeEntry()
                }
            }
        }
        true
    } catch (t: Throwable) {
        false
    }
}
