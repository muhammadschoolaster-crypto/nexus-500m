package com.nexus.assistant

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

class ChatAdapter : RecyclerView.Adapter<ChatAdapter.VH>() {

    enum class Role { USER, ASSISTANT }

    data class Item(val role: Role, val text: String)

    private val items = ArrayList<Item>()

    fun add(role: Role, text: String) {
        items.add(Item(role, text))
        notifyItemInserted(items.size - 1)
    }

    fun updateLast(text: String) {
        if (items.isEmpty()) return
        items[items.size - 1] = Item(items[items.size - 1].role, text)
        notifyItemChanged(items.size - 1)
    }

    fun setAll(list: List<Item>) {
        items.clear()
        items.addAll(list)
        notifyDataSetChanged()
    }

    override fun getItemViewType(position: Int): Int = items[position].role.ordinal

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val layout = if (viewType == Role.USER.ordinal)
            R.layout.item_message_user else R.layout.item_message_assistant
        val v = LayoutInflater.from(parent.context).inflate(layout, parent, false)
        return VH(v)
    }

    override fun getItemCount(): Int = items.size

    class VH(v: View) : RecyclerView.ViewHolder(v) {
        val text: TextView = v.findViewById(R.id.messageText)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        holder.text.text = items[position].text
    }
}
