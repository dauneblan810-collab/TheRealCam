package com.therealcam.ui.common

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.therealcam.data.SceneKeys
import com.therealcam.ui.theme.RealColors

/** Create or edit a category: a name and one of the built-in scene icons. */
@Composable
fun CategoryDialog(
    title: String,
    initialName: String = "",
    initialIcon: String? = null,
    onDismiss: () -> Unit,
    onConfirm: (name: String, icon: String) -> Unit,
) {
    var name by remember { mutableStateOf(initialName) }
    var icon by remember { mutableStateOf(initialIcon) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = {
            Column {
                OutlinedTextField(
                    value = name, onValueChange = { name = it.take(40) }, singleLine = true,
                    label = { Text("Название") },
                    placeholder = { Text(icon?.let { SceneKeys.entry(it).label } ?: "Например: Звёзды на даче") },
                    modifier = Modifier.fillMaxWidth(),
                )
                Spacer(Modifier.height(10.dp))
                Text(
                    icon?.let { "Значок: ${SceneKeys.entry(it).label} · профиль «${SceneKeys.profileOf(it).label}»" } ?: "Выберите значок",
                    fontSize = 12.sp, color = RealColors.TextDim,
                )
                Spacer(Modifier.height(6.dp))
                LazyVerticalGrid(
                    columns = GridCells.Fixed(6),
                    modifier = Modifier.heightIn(max = 260.dp),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    items(SceneKeys.all, key = { it.key }) { e ->
                        val sel = e.key == icon
                        Box(
                            Modifier
                                .aspectRatio(1f)
                                .clip(RoundedCornerShape(10.dp))
                                .background(if (sel) RealColors.Accent else RealColors.SurfaceHigh)
                                .clickable { icon = e.key },
                            contentAlignment = Alignment.Center,
                        ) {
                            Icon(SceneIcons.of(e.key), e.label, tint = if (sel) Color.Black else Color.White, modifier = Modifier.size(22.dp).padding(1.dp))
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(enabled = icon != null, onClick = {
                val ic = icon ?: return@TextButton
                onConfirm(name.trim().ifEmpty { SceneKeys.entry(ic).label }, ic)
            }) { Text("Сохранить") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Отмена") } },
    )
}
