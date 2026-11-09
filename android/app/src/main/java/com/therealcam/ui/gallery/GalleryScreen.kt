package com.therealcam.ui.gallery

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.BurstMode
import androidx.compose.material.icons.filled.Category
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.therealcam.data.db.CaptureEntity
import com.therealcam.dataset.MIN_DATASET_SAMPLES
import com.therealcam.ui.common.SceneIcons
import com.therealcam.ui.theme.RealColors

@Composable
fun GalleryGrid(
    vm: GalleryViewModel,
    onOpen: (List<String>, Int) -> Unit,
    onCategories: () -> Unit,
    onSettings: () -> Unit,
) {
    val items by vm.items.collectAsState()
    val filter by vm.filter.collectAsState()
    val categories by vm.categories.collectAsState()
    val catFilter by vm.categoryFilter.collectAsState()
    val datasetCount by vm.datasetCount.collectAsState()
    val pendingCount by vm.pendingCount.collectAsState()

    Column(Modifier.fillMaxSize().background(Color.Black).statusBarsPadding()) {
        Row(Modifier.fillMaxWidth().padding(start = 16.dp, end = 4.dp, top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Real Галерея", fontSize = 22.sp, fontWeight = FontWeight.Bold, color = Color.White)
                Text(
                    "Датасет: $datasetCount" + (if (datasetCount < MIN_DATASET_SAMPLES) " / $MIN_DATASET_SAMPLES" else " ✓") +
                        if (pendingCount > 0) " · ждут оценки: $pendingCount" else "",
                    fontSize = 12.sp, color = RealColors.TextDim,
                )
            }
            IconButton(onClick = onCategories) { Icon(Icons.Filled.Category, "Категории", tint = Color.White) }
            IconButton(onClick = onSettings) { Icon(Icons.Filled.Settings, "Настройки", tint = Color.White) }
        }
        if (datasetCount < MIN_DATASET_SAMPLES) {
            LinearProgressIndicator(
                progress = { datasetCount / MIN_DATASET_SAMPLES.toFloat() },
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp),
                color = RealColors.Accent,
            )
        }
        LazyRow(
            Modifier.fillMaxWidth().padding(vertical = 6.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 12.dp),
        ) {
            items(GalleryFilter.entries) { f ->
                FilterChip(f.label, f == filter) { vm.setFilter(f) }
            }
            items(categories, key = { it.id }) { c ->
                val sel = c.id == catFilter
                Box(
                    Modifier.size(32.dp).clip(CircleShape).background(if (sel) RealColors.Accent else RealColors.SurfaceHigh)
                        .clickable { vm.setCategoryFilter(c.id) },
                    contentAlignment = Alignment.Center,
                ) {
                    Icon(SceneIcons.of(c.icon), c.name, tint = if (sel) Color.Black else Color.White, modifier = Modifier.size(18.dp))
                }
            }
        }
        val list = items
        when {
            list == null -> Unit
            list.isEmpty() -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(
                    if (filter == GalleryFilter.ALL && catFilter == null) "Здесь появятся снимки TheRealCam" else "Ничего не найдено",
                    color = RealColors.TextDim,
                )
            }
            else -> LazyVerticalGrid(
                columns = GridCells.Adaptive(110.dp),
                modifier = Modifier.fillMaxSize(),
                horizontalArrangement = Arrangement.spacedBy(2.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                items(list, key = { it.id }) { e ->
                    Thumb(e) { onOpen(list.map { it.id }, list.indexOf(e)) }
                }
            }
        }
    }
}

@Composable
fun FilterChip(text: String, selected: Boolean, onClick: () -> Unit) {
    Text(
        text, fontSize = 13.sp, color = if (selected) Color.Black else Color.White,
        modifier = Modifier.clip(RoundedCornerShape(16.dp))
            .background(if (selected) RealColors.Accent else RealColors.SurfaceHigh)
            .clickable(onClick = onClick).padding(horizontal = 12.dp, vertical = 7.dp),
    )
}

@Composable
private fun Thumb(e: CaptureEntity, onClick: () -> Unit) {
    val ctx = LocalContext.current
    Box(Modifier.aspectRatio(1f).background(RealColors.Surface).clickable(onClick = onClick)) {
        AsyncImage(
            model = ImageRequest.Builder(ctx).data(e.imageUri).size(360).crossfade(true).build(),
            contentDescription = null,
            contentScale = ContentScale.Crop,
            modifier = Modifier.fillMaxSize(),
        )
        if (e.seriesId != null) {
            Row(
                Modifier.align(Alignment.BottomStart).padding(4.dp).clip(RoundedCornerShape(6.dp))
                    .background(RealColors.Scrim).padding(horizontal = 4.dp, vertical = 1.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(Icons.Filled.BurstMode, null, tint = Color.White, modifier = Modifier.size(12.dp))
                Spacer(Modifier.size(2.dp))
                Text("${e.seriesCount}", fontSize = 10.sp, color = Color.White)
            }
        }
        when {
            e.inDataset -> Icon(Icons.Filled.CheckCircle, "В датасете", tint = RealColors.Ok,
                modifier = Modifier.align(Alignment.TopEnd).padding(4.dp).size(16.dp))
            e.hasMeta -> Box(Modifier.align(Alignment.TopEnd).padding(6.dp).size(9.dp).clip(CircleShape).background(RealColors.Warn))
        }
        e.rating?.let {
            Text(
                "$it", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = Color.Black,
                modifier = Modifier.align(Alignment.TopStart).padding(4.dp).clip(RoundedCornerShape(6.dp))
                    .background(RealColors.Accent).padding(horizontal = 5.dp),
            )
        }
    }
}
