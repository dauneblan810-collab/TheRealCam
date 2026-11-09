package com.therealcam.ui.gallery

import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.therealcam.camera.ExposureScale
import com.therealcam.data.Profile
import com.therealcam.data.db.CaptureEntity
import com.therealcam.data.db.Roles
import com.therealcam.ui.common.CategoryDialog
import com.therealcam.ui.common.RatingPanel
import com.therealcam.ui.common.completenessText
import com.therealcam.ui.theme.RealColors
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun DetailScreen(
    vm: GalleryViewModel,
    ids: List<String>,
    start: Int,
    onBack: () -> Unit,
    onOpenFrames: (List<String>, Int) -> Unit,
) {
    if (ids.isEmpty()) {
        onBack()
        return
    }
    val pager = rememberPagerState(initialPage = start.coerceIn(0, ids.lastIndex)) { ids.size }
    var zoomed by remember { mutableStateOf(false) }
    val currentId = ids[pager.currentPage.coerceIn(0, ids.lastIndex)]
    val current by remember(currentId) { vm.observe(currentId) }.collectAsState(initial = null)
    val categories by vm.categories.collectAsState()
    var expanded by remember { mutableStateOf<Boolean?>(null) }
    var info by remember { mutableStateOf(false) }
    var confirmDelete by remember { mutableStateOf(false) }
    var newCategory by remember { mutableStateOf(false) }
    val ctx = LocalContext.current

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        HorizontalPager(state = pager, userScrollEnabled = !zoomed, modifier = Modifier.fillMaxSize(), key = { ids[it] }) { page ->
            val e by remember(ids[page]) { vm.observe(ids[page]) }.collectAsState(initial = null)
            e?.let { ZoomableImage(it.imageUri, onZoom = { z -> if (page == pager.currentPage) zoomed = z }) }
        }

        Row(
            Modifier.fillMaxWidth().background(RealColors.ScrimLight).statusBarsPadding().padding(4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Назад", tint = Color.White) }
            Column(Modifier.weight(1f)) {
                current?.let { e ->
                    Text(
                        when (e.role) {
                            Roles.FRAME -> "Кадр ${e.frameIndex + 1} из ${e.seriesCount}"
                            Roles.RESULT -> "Серия: ${e.seriesCount} кадров · ${Profile.of(e.profile).label}"
                            else -> SimpleDateFormat("d MMM, HH:mm", Locale("ru")).format(Date(e.createdAt))
                        },
                        color = Color.White, fontSize = 15.sp, fontWeight = FontWeight.SemiBold,
                    )
                    Text(shotLine(e), color = RealColors.TextDim, fontSize = 11.sp)
                }
            }
            IconButton(onClick = { info = true }) { Icon(Icons.Filled.Info, "Информация", tint = Color.White) }
            IconButton(onClick = { current?.let { share(ctx, it) } }) { Icon(Icons.Filled.Share, "Поделиться", tint = Color.White) }
            IconButton(onClick = { confirmDelete = true }) { Icon(Icons.Filled.Delete, "Удалить", tint = Color.White) }
        }

        current?.let { e ->
            val open = expanded ?: (e.hasMeta && !e.isRatingComplete)
            Column(
                Modifier.align(Alignment.BottomCenter).fillMaxWidth()
                    .clip(RoundedCornerShape(topStart = 18.dp, topEnd = 18.dp))
                    .background(RealColors.Surface.copy(alpha = 0.96f)).navigationBarsPadding().padding(horizontal = 14.dp, vertical = 8.dp),
            ) {
                if (e.role == Roles.RESULT && e.seriesId != null) {
                    SeriesStrip(vm, e.seriesId) { frameIds, i -> onOpenFrames(frameIds, i) }
                    Spacer(Modifier.height(6.dp))
                }
                Row(
                    Modifier.fillMaxWidth().clickable { expanded = !open },
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    e.rating?.let {
                        Text("$it", fontWeight = FontWeight.Bold, color = Color.Black,
                            modifier = Modifier.clip(RoundedCornerShape(6.dp)).background(RealColors.Accent).padding(horizontal = 7.dp, vertical = 2.dp))
                        Spacer(Modifier.size(8.dp))
                    }
                    Text(completenessText(e), color = if (e.inDataset) RealColors.Ok else RealColors.TextDim, fontSize = 12.sp, modifier = Modifier.weight(1f))
                    Icon(if (open) Icons.Filled.ExpandMore else Icons.Filled.ExpandLess, null, tint = Color.White)
                }
                if (open) {
                    Column(Modifier.heightIn(max = 430.dp).verticalScroll(rememberScrollState())) {
                        Spacer(Modifier.height(6.dp))
                        if (!e.hasMeta) {
                            Text(
                                "Снимок сделан без режима сбора датасета: оценка сохранится, но в обучение он не попадёт.",
                                color = RealColors.TextDim, fontSize = 11.sp,
                            )
                            Spacer(Modifier.height(6.dp))
                        }
                        RatingPanel(e, categories, onChange = { change -> vm.rate(e.id, change) }, onAddCategory = { newCategory = true })
                    }
                }
            }
        }
    }

    if (info) current?.let { e -> InfoDialog(vm, e) { info = false } }
    if (confirmDelete) current?.let { e ->
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text("Удалить снимок?") },
            text = {
                Text(
                    if (e.role == Roles.RESULT) "Удалятся итог серии и все её кадры, а также их данные в датасете."
                    else "Снимок и его данные в датасете будут удалены.",
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    confirmDelete = false
                    vm.delete(e)
                    if (ids.size <= 1) onBack()
                }) { Text("Удалить", color = RealColors.Bad) }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Отмена") } },
        )
    }
    if (newCategory) {
        CategoryDialog("Новая категория", onDismiss = { newCategory = false }) { name, icon ->
            newCategory = false
            vm.createCategory(name, icon, assignTo = currentId)
        }
    }
}

private fun shotLine(e: CaptureEntity): String = listOfNotNull(
    e.exposureNs?.let { ExposureScale.formatShutter(it) },
    e.iso?.let { "ISO$it" },
    e.focusDiopters?.let { if (it <= 0.01f) "∞" else "%.2fм".format(1f / it) },
    when (e.mode) { "manual" -> "ручной"; "ai" -> "AI"; else -> "авто" },
).joinToString(" · ")

@Composable
private fun SeriesStrip(vm: GalleryViewModel, seriesId: String, onOpen: (List<String>, Int) -> Unit) {
    val members by remember(seriesId) { vm.series(seriesId) }.collectAsState(initial = emptyList())
    val frames = members.filter { it.role == Roles.FRAME }.sortedBy { it.frameIndex }
    if (frames.isEmpty()) return
    val ctx = LocalContext.current
    LazyRow(horizontalArrangement = Arrangement.spacedBy(4.dp), modifier = Modifier.fillMaxWidth()) {
        itemsIndexed(frames, key = { _, f -> f.id }) { i, f ->
            Box(
                Modifier.size(52.dp).clip(RoundedCornerShape(6.dp))
                    .border(1.dp, if (f.inDataset) RealColors.Ok else Color.Transparent, RoundedCornerShape(6.dp))
                    .clickable { onOpen(frames.map { it.id }, i) },
            ) {
                AsyncImage(ImageRequest.Builder(ctx).data(f.imageUri).size(160).build(), null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
                Text("${f.frameIndex + 1}", fontSize = 9.sp, color = Color.White,
                    modifier = Modifier.align(Alignment.BottomEnd).background(RealColors.Scrim).padding(horizontal = 3.dp))
            }
        }
    }
}

@Composable
private fun ZoomableImage(uri: String, onZoom: (Boolean) -> Unit) {
    var scale by remember { mutableFloatStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }
    val ctx = LocalContext.current
    Box(
        Modifier.fillMaxSize()
            .pointerInput(Unit) {
                detectTapGestures(onDoubleTap = {
                    if (scale > 1f) {
                        scale = 1f
                        offset = Offset.Zero
                    } else scale = 2.5f
                    onZoom(scale > 1f)
                })
            }
            .pointerInput(Unit) {
                awaitEachGesture {
                    awaitFirstDown(requireUnconsumed = false)
                    do {
                        val event = awaitPointerEvent()
                        // Leave single-finger swipes to the pager unless the image is zoomed in.
                        if (event.changes.size > 1 || scale > 1f) {
                            scale = (scale * event.calculateZoom()).coerceIn(1f, 6f)
                            offset = if (scale <= 1f) Offset.Zero else offset + event.calculatePan()
                            onZoom(scale > 1f)
                            event.changes.forEach { if (it.positionChanged()) it.consume() }
                        }
                    } while (event.changes.any { it.pressed })
                }
            },
        contentAlignment = Alignment.Center,
    ) {
        AsyncImage(
            model = ImageRequest.Builder(ctx).data(uri).size(2048).build(),
            contentDescription = null,
            contentScale = ContentScale.Fit,
            modifier = Modifier.fillMaxSize().graphicsLayer(scaleX = scale, scaleY = scale, translationX = offset.x, translationY = offset.y),
        )
    }
}

@Composable
private fun InfoDialog(vm: GalleryViewModel, e: CaptureEntity, onDismiss: () -> Unit) {
    val meta = remember(e.id) { vm.meta(e.id) }
    val lines = buildList {
        add("Дата" to SimpleDateFormat("d MMMM yyyy, HH:mm:ss", Locale("ru")).format(Date(e.createdAt)))
        add("Размер" to "${e.width}×${e.height}")
        e.exposureNs?.let { add("Выдержка" to ExposureScale.formatShutter(it)) }
        e.iso?.let { add("ISO" to "$it") }
        e.focusDiopters?.let { add("Фокус" to if (it <= 0.01f) "∞" else "%.2f м".format(1f / it)) }
        add("Режим" to when (e.mode) { "manual" -> "Ручной"; "ai" -> "AI"; else -> "Авто" })
        e.profile?.let { add("Профиль" to Profile.of(it).label) }
        if (e.lat != null && e.lon != null) add("Место" to "%.5f, %.5f".format(e.lat, e.lon))
        meta?.environment?.lux?.let { add("Освещённость" to "%.1f лк".format(it)) }
        meta?.environment?.sunAltDeg?.let { add("Солнце" to "%.0f° над горизонтом".format(it)) }
        meta?.motion?.stats?.estBlurPx?.let { add("Оценка смаза" to "%.1f px".format(it)) }
        meta?.ai?.let { ai -> add("AI" to "${ai.modelName}: ${ai.chosenClass}" + if (ai.userOverride) " (выбрано вручную)" else "") }
        meta?.series?.frames?.let { fr ->
            val used = fr.count { it.accepted }
            add("Сведение" to "$used из ${fr.size} кадров")
            fr.filter { !it.accepted }.forEach { f ->
                add("  кадр ${f.index + 1}" to when (f.rejectReason) {
                    "motion" -> "отброшен: телефон двигался"
                    "brightness_outlier" -> "отброшен: засветка"
                    "not_sharpest" -> "не из лучших по резкости"
                    "align_failed" -> "не удалось выровнять"
                    "metering" -> "замерочный кадр"
                    else -> f.rejectReason ?: "не использован"
                })
            }
        }
        add("Датасет" to completenessText(e))
    }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Информация") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                lines.forEach { (k, v) ->
                    Row(Modifier.padding(vertical = 2.dp)) {
                        Text(k, color = RealColors.TextDim, fontSize = 13.sp, modifier = Modifier.weight(0.4f))
                        Text(v, color = Color.White, fontSize = 13.sp, modifier = Modifier.weight(0.6f))
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("OK") } },
    )
}

private fun share(ctx: android.content.Context, e: CaptureEntity) {
    val uri = Uri.parse(e.imageUri)
    val shareUri = if (uri.scheme == "file") {
        runCatching { FileProvider.getUriForFile(ctx, "${ctx.packageName}.files", File(uri.path!!)) }.getOrNull() ?: return
    } else uri
    val send = Intent(Intent.ACTION_SEND).apply {
        type = "image/jpeg"
        putExtra(Intent.EXTRA_STREAM, shareUri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    ctx.startActivity(Intent.createChooser(send, "Поделиться"))
}
