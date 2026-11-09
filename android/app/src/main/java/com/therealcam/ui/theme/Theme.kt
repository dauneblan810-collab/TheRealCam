package com.therealcam.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

object RealColors {
    val Accent = Color(0xFFFFC400)
    val AccentDim = Color(0x66FFC400)
    val Ok = Color(0xFF4CD964)
    val Warn = Color(0xFFFF9F0A)
    val Bad = Color(0xFFFF453A)
    val Scrim = Color(0x99000000)
    val ScrimLight = Color(0x55000000)
    val Surface = Color(0xFF15181D)
    val SurfaceHigh = Color(0xFF1F232A)
    val TextDim = Color(0xFFB0B4BA)
}

private val scheme = darkColorScheme(
    primary = RealColors.Accent,
    onPrimary = Color.Black,
    secondary = Color(0xFF8AB4F8),
    background = Color.Black,
    onBackground = Color.White,
    surface = RealColors.Surface,
    onSurface = Color.White,
    surfaceVariant = RealColors.SurfaceHigh,
    onSurfaceVariant = RealColors.TextDim,
    error = RealColors.Bad,
)

@Composable
fun RealTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = scheme, typography = Typography(), content = content)
}
