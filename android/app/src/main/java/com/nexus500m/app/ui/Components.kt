package com.nexus500m.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Markdown-lite rendering: fenced ```code``` blocks become monospace chips
 * with horizontal scrolling; everything else renders as plain body text.
 * (Full markdown rendering is a future enhancement — kept honest.)
 */
object MarkdownLite {

    fun parse(text: String): List<Section> {
        val sections = ArrayList<Section>()
        val lines = text.lines()
        val buf = StringBuilder()
        val code = StringBuilder()
        var inCode = false
        for (line in lines) {
            if (line.trim().startsWith("```")) {
                if (inCode) {
                    sections.add(Section.Code(code.toString().trimEnd()))
                    code.clear()
                    inCode = false
                } else {
                    if (buf.isNotBlank()) sections.add(Section.Text(buf.toString().trim()))
                    buf.clear()
                    inCode = true
                }
            } else if (inCode) {
                code.appendLine(line)
            } else {
                buf.appendLine(line)
            }
        }
        if (inCode && code.isNotBlank()) sections.add(Section.Code(code.toString()))
        if (buf.isNotBlank()) sections.add(Section.Text(buf.toString().trim()))
        return sections
    }

    sealed class Section {
        data class Text(val text: String) : Section()
        data class Code(val code: String) : Section()
    }
}

@Composable
fun MarkdownText(text: String, modifier: Modifier = Modifier) {
    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(6.dp)) {
        for (section in MarkdownLite.parse(text)) {
            when (section) {
                is MarkdownLite.Section.Text -> Text(
                    section.text,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                is MarkdownLite.Section.Code -> CodeBlock(section.code)
            }
        }
    }
}

@Composable
fun CodeBlock(code: String) {
    val scroll = rememberScrollState()
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f),
                shape = RoundedCornerShape(10.dp),
            )
            .horizontalScroll(scroll)
            .padding(10.dp),
    ) {
        Text(
            code,
            fontFamily = FontFamily.Monospace,
            fontSize = 12.sp,
            color = MaterialTheme.colorScheme.onSurface,
        )
    }
}
