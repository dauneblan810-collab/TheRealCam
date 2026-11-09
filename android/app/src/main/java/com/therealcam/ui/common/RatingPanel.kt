package com.therealcam.ui.common

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.therealcam.data.db.CaptureEntity
import com.therealcam.data.db.CategoryEntity
import com.therealcam.ui.theme.RealColors

/** One rating criterion: label + options (value to caption). */
data class Criterion(val title: String, val options: List<Pair<Int, String>>, val get: (CaptureEntity) -> Int?, val set: (CaptureEntity, Int) -> CaptureEntity)

val Criteria: List<Criterion> = listOf(
    Criterion("Яркость", listOf(-1 to "Темно", 0 to "Норм", 1 to "Пересвет"), { it.fbExposure }, { e, v -> e.copy(fbExposure = v) }),
    Criterion("Резкость", listOf(-1 to "Мыло", 0 to "Норм", 1 to "Чётко"), { it.fbSharpness }, { e, v -> e.copy(fbSharpness = v) }),
    Criterion("Шум", listOf(-1 to "Шумно", 0 to "Терпимо", 1 to "Чисто"), { it.fbNoise }, { e, v -> e.copy(fbNoise = v) }),
    Criterion("Цвет", listOf(-1 to "Холодно", 0 to "Естеств.", 1 to "Тепло"), { it.fbColor }, { e, v -> e.copy(fbColor = v) }),
    Criterion("Смаз", listOf(0 to "Нет", 1 to "Есть"), { it.fbMotionBlur }, { e, v -> e.copy(fbMotionBlur = v) }),
)

@Composable
fun RatingRow(value: Int?, onChange: (Int) -> Unit, modifier: Modifier = Modifier) {
    Row(modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        for (i in 1..10) {
            val sel = value == i
            Box(
                Modifier
                    .size(30.dp)
                    .clip(CircleShape)
                    .background(if (sel) RealColors.Accent else RealColors.SurfaceHigh)
                    .clickable { onChange(i) },
                contentAlignment = Alignment.Center,
            ) {
                Text("$i", color = if (sel) Color.Black else Color.White, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
            }
        }
    }
}

@Composable
fun SegmentedChoice(options: List<Pair<Int, String>>, value: Int?, onChange: (Int) -> Unit, modifier: Modifier = Modifier) {
    Row(
        modifier
            .clip(RoundedCornerShape(10.dp))
            .background(RealColors.SurfaceHigh),
    ) {
        options.forEach { (v, label) ->
            val sel = v == value
            Box(
                Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(10.dp))
                    .background(if (sel) RealColors.Accent else Color.Transparent)
                    .clickable { onChange(v) }
                    .padding(vertical = 6.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(label, color = if (sel) Color.Black else Color.White, fontSize = 12.sp, textAlign = TextAlign.Center, maxLines = 1)
            }
        }
    }
}

@Composable
fun CategoryPicker(
    categories: List<CategoryEntity>,
    selectedId: String?,
    onSelect: (CategoryEntity) -> Unit,
    onAdd: (() -> Unit)?,
    modifier: Modifier = Modifier,
) {
    LazyRow(modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        items(categories, key = { it.id }) { c ->
            val sel = c.id == selectedId
            Column(
                Modifier
                    .width(64.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .clickable { onSelect(c) }
                    .padding(vertical = 4.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Box(
                    Modifier
                        .size(40.dp)
                        .clip(CircleShape)
                        .background(if (sel) RealColors.Accent else RealColors.SurfaceHigh)
                        .border(1.dp, if (sel) RealColors.Accent else Color.Transparent, CircleShape),
                    contentAlignment = Alignment.Center,
                ) {
                    Icon(SceneIcons.of(c.icon), c.name, tint = if (sel) Color.Black else Color.White, modifier = Modifier.size(22.dp))
                }
                Text(c.name, fontSize = 10.sp, maxLines = 1, color = if (sel) RealColors.Accent else RealColors.TextDim)
            }
        }
        if (onAdd != null) item {
            Column(
                Modifier
                    .width(64.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .clickable { onAdd() }
                    .padding(vertical = 4.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Box(Modifier.size(40.dp).clip(CircleShape).border(1.dp, RealColors.TextDim, CircleShape), contentAlignment = Alignment.Center) {
                    Icon(Icons.Filled.Add, "Новая категория", tint = Color.White)
                }
                Text("Новая", fontSize = 10.sp, color = RealColors.TextDim)
            }
        }
    }
}

/** Full rating editor: 1–10, every criterion and the category. Used by the camera overlay and the gallery. */
@Composable
fun RatingPanel(
    capture: CaptureEntity,
    categories: List<CategoryEntity>,
    onChange: ((CaptureEntity) -> CaptureEntity) -> Unit,
    onAddCategory: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier.fillMaxWidth()) {
        Text("Оценка", color = RealColors.TextDim, fontSize = 12.sp)
        Spacer(Modifier.height(4.dp))
        RatingRow(capture.rating, { v -> onChange { it.copy(rating = v) } })
        Spacer(Modifier.height(8.dp))
        Criteria.forEach { c ->
            Row(Modifier.fillMaxWidth().padding(vertical = 3.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(c.title, Modifier.width(76.dp), fontSize = 12.sp, color = if (c.get(capture) == null) RealColors.Warn else RealColors.TextDim)
                SegmentedChoice(c.options, c.get(capture), { v -> onChange { e -> c.set(e, v) } }, Modifier.weight(1f))
            }
        }
        Spacer(Modifier.height(6.dp))
        Text(
            "Что снято" + if (capture.categoryId == null) " — выберите категорию" else "",
            color = if (capture.categoryId == null) RealColors.Warn else RealColors.TextDim, fontSize = 12.sp,
        )
        Spacer(Modifier.height(4.dp))
        CategoryPicker(categories, capture.categoryId, { cat -> onChange { it.copy(categoryId = cat.id) } }, onAddCategory)
    }
}

fun completenessText(e: CaptureEntity): String = when {
    !e.hasMeta -> "Без данных датасета"
    e.inDataset -> "В датасете"
    e.isRatingComplete -> "Оценено"
    else -> {
        val missing = buildList {
            if (e.rating == null) add("оценка")
            Criteria.filter { it.get(e) == null }.forEach { add(it.title.lowercase()) }
            if (e.categoryId == null) add("категория")
        }
        "Не хватает: " + missing.joinToString(", ")
    }
}
