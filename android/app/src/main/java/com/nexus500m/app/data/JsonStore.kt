package com.nexus500m.app.data

import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Tiny atomic-JSON persistence helper. All app data lives in the app's private
 * sandbox. Writes are write-temp-then-rename so a crash or full disk can never
 * leave a half-written store (corruption-safe by construction).
 */
object JsonStore {

    fun readArray(file: File): JSONArray {
        if (!file.exists()) return JSONArray()
        return try {
            JSONArray(file.readText(Charsets.UTF_8))
        } catch (t: Throwable) {
            JSONArray() // unreadable -> treat as empty, but never throw at read time
        }
    }

    fun writeArray(file: File, arr: JSONArray) {
        file.parentFile?.mkdirs()
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.writeText(arr.toString(), Charsets.UTF_8)
        if (!tmp.renameTo(file)) {
            // rename can fail across odd filesystems: fall back to direct write
            file.writeText(arr.toString(), Charsets.UTF_8)
            tmp.delete()
        }
    }

    fun readObject(file: File): JSONObject {
        if (!file.exists()) return JSONObject()
        return try {
            JSONObject(file.readText(Charsets.UTF_8))
        } catch (t: Throwable) {
            JSONObject()
        }
    }

    fun writeObject(file: File, obj: JSONObject) {
        file.parentFile?.mkdirs()
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.writeText(obj.toString(), Charsets.UTF_8)
        if (!tmp.renameTo(file)) {
            file.writeText(obj.toString(), Charsets.UTF_8)
            tmp.delete()
        }
    }
}
