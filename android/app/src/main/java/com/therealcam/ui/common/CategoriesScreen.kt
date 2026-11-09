package com.therealcam.ui.common

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.therealcam.data.SceneKeys
import com.therealcam.data.db.CategoryEntity
import com.therealcam.ui.theme.RealColors

private val Suggestions = listOf("stars", "moon", "clouds", "sunset", "city_night", "landscape", "portrait", "pet", "food", "document")

/** List of the user's categories with create / rename / change icon / delete. */
@Composable
fun CategoriesScreen(
    categories: List<CategoryEntity>,
    onBack: () -> Unit,
    onCreate: (String, String) -> Unit,
    onUpdate: (CategoryEntity) -> Unit,
    onDelete: (String) -> Unit,
) {
    var editing by remember { mutableStateOf<CategoryEntity?>(null) }
    var creating by remember { mutableStateOf(false) }
    var deleting by remember { mutableStateOf<CategoryEntity?>(null) }
    Column(Modifier.fillMaxSize().background(Color.Black).statusBarsPadding()) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(4.dp)) {
            IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Назад", tint = Color.White) }
            Text("Категории", fontSize = 20.sp, fontWeight = FontWeight.Bold, color = Color.White, modifier = Modifier.weight(1f))
            IconButton(onClick = { creating = true }) { Icon(Icons.Filled.Add, "Новая", tint = RealColors.Accent) }
        }
        Text(
            "Категория — это то, что вы снимаете. Значок задаёт профиль съёмки (например, «звёзды» — сведение без выбросов, " +
                "«луна» — отбор самых резких кадров). ИИ учится различать ваши категории.",
            color = RealColors.TextDim, fontSize = 12.sp, modifier = Modifier.padding(horizontal = 16.dp),
        )
        val unused = Suggestions.filter { key -> categories.none { it.icon == key } }
        if (unused.isNotEmpty()) {
            Text("Быстро добавить:", color = RealColors.TextDim, fontSize = 12.sp, modifier = Modifier.padding(start = 16.dp, top = 12.dp))
            LazyRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 16.dp, vertical = 6.dp),
            ) {
                items(unused) { key ->
                    val e = SceneKeys.entry(key)
                    Row(
                        Modifier.clip(RoundedCornerShape(16.dp)).background(RealColors.SurfaceHigh)
                            .clickable { onCreate(e.label, key) }.padding(horizontal = 10.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(SceneIcons.of(key), null, tint = Color.White, modifier = Modifier.size(16.dp))
                        Spacer(Modifier.size(4.dp))
                        Text(e.label, fontSize = 12.sp, color = Color.White)
                    }
                }
            }
        }
        LazyColumn(Modifier.fillMaxSize().padding(top = 8.dp)) {
            items(categories, key = { it.id }) { c ->
                Row(
                    Modifier.fillMaxWidth().clickable { editing = c }.padding(horizontal = 16.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Box(Modifier.size(40.dp).clip(CircleShape).background(RealColors.SurfaceHigh), contentAlignment = Alignment.Center) {
                        Icon(SceneIcons.of(c.icon), null, tint = Color.White)
                    }
                    Spacer(Modifier.size(12.dp))
                    Column(Modifier.weight(1f)) {
                        Text(c.name, color = Color.White)
                        Text("Профиль: ${SceneKeys.profileOf(c.icon).label}", color = RealColors.TextDim, fontSize = 12.sp)
                    }
                    IconButton(onClick = { editing = c }) { Icon(Icons.Filled.Edit, "Изменить", tint = RealColors.TextDim) }
                    IconButton(onClick = { deleting = c }) { Icon(Icons.Filled.Delete, "Удалить", tint = RealColors.TextDim) }
                }
            }
        }
    }
    if (creating) CategoryDialog("Новая категория", onDismiss = { creating = false }) { n, i ->
        creating = false
        onCreate(n, i)
    }
    editing?.let { c ->
        CategoryDialog("Категория", c.name, c.icon, onDismiss = { editing = null }) { n, i ->
            editing = null
            onUpdate(c.copy(name = n, icon = i))
        }
    }
    deleting?.let { c ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("Удалить «${c.name}»?") },
            text = { Text("Снимки этой категории потеряют её и выйдут из датасета, пока вы не выберете им другую категорию.") },
            confirmButton = { TextButton(onClick = { deleting = null; onDelete(c.id) }) { Text("Удалить", color = RealColors.Bad) } },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("Отмена") } },
        )
    }
}
