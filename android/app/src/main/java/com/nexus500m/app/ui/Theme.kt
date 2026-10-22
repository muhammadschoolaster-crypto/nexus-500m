package com.nexus500m.app.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val Sky = Color(0xFF38BDF8)
private val Navy = Color(0xFF0F172A)

private val DarkScheme = darkColorScheme(
    primary = Sky,
    onPrimary = Navy,
    background = Color(0xFF0B1220),
    surface = Color(0xFF111A2C),
    surfaceVariant = Color(0xFF1B2740),
    secondary = Color(0xFF93C5FD),
)

private val LightScheme = lightColorScheme(
    primary = Color(0xFF0284C7),
    background = Color(0xFFF8FAFC),
    surface = Color(0xFFFFFFFF),
    surfaceVariant = Color(0xFFE2E8F0),
)

@Composable
fun NexusTheme(darkTheme: Boolean = isSystemInDarkTheme(), content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkScheme else LightScheme,
        content = content,
    )
}
