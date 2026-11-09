package com.therealcam.ui.camera

import android.graphics.SurfaceTexture
import android.view.TextureView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FlashOff
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material.icons.filled.FlipCameraAndroid
import androidx.compose.material.icons.filled.Grid3x3
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Timer
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import coil.compose.AsyncImage
import com.therealcam.ai.ScenePreview
import com.therealcam.camera.CameraMode
import com.therealcam.camera.ExposureScale
import com.therealcam.capture.ShotRunner
import com.therealcam.dataset.MIN_DATASET_SAMPLES
import com.therealcam.ui.common.CategoryDialog
import com.therealcam.ui.common.RatingPanel
import com.therealcam.ui.common.SceneIcons
import com.therealcam.ui.common.completenessText
import com.therealcam.ui.theme.RealColors
import kotlinx.coroutines.delay
import kotlin.math.abs
import kotlin.math.roundToInt

@Composable
fun CameraScreen(
    vm: CameraViewModel,
    onGallery: (captureId: String?) -> Unit,
    onSettings: () -> Unit,
) {
    val settings by vm.settings.collectAsState()
    val mode by vm.mode.collectAsState()
    val caps by vm.camera.caps.collectAsState()
    val motion by vm.motion.collectAsState()
    val busy by vm.busy.collectAsState()
    val countdown by vm.countdown.collectAsState()
    val ratingFor by vm.ratingFor.collectAsState()
    val message by vm.message.collectAsState()
    val latest by vm.latest.collectAsState()
    val datasetCount by vm.datasetCount.collectAsState()
    val model by vm.model.collectAsState()
    val ai by vm.ai.collectAsState()
    val torch by vm.torch.collectAsState()
    val rot = rotationFor(motion.deviceOrientation)

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        Column(Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding()) {
            // ---------------------------------------------------------------- top bar
            Row(
                Modifier.fillMaxWidth().height(48.dp).padding(horizontal = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                TopIcon(Icons.Filled.Settings, "Настройки", rot) { onSettings() }
                TopIcon(Icons.Filled.Timer, "Таймер", rot, badge = settings.timerSeconds.takeIf { it > 0 }?.let { "${it}с" }) {
                    vm.setTimer(when (settings.timerSeconds) { 0 -> 3; 3 -> 10; else -> 0 })
                }
                TopIcon(Icons.Filled.Grid3x3, "Сетка", rot, active = settings.grid) { vm.setGrid(!settings.grid) }
                if (caps?.flash == true) {
                    TopIcon(if (torch) Icons.Filled.FlashOn else Icons.Filled.FlashOff, "Фонарик", rot, active = torch) { vm.toggleTorch() }
                }
                Spacer(Modifier.weight(1f))
                if (settings.datasetMode) {
                    Row(
                        Modifier.clip(RoundedCornerShape(10.dp)).background(RealColors.SurfaceHigh).padding(horizontal = 8.dp, vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Box(Modifier.size(7.dp).clip(CircleShape).background(RealColors.Bad))
                        Spacer(Modifier.size(5.dp))
                        Text(
                            if (datasetCount >= MIN_DATASET_SAMPLES) "$datasetCount" else "$datasetCount/$MIN_DATASET_SAMPLES",
                            fontSize = 11.sp, color = Color.White,
                        )
                    }
                }
                if (model != null) {
                    Spacer(Modifier.size(6.dp))
                    Icon(Icons.Filled.Psychology, "Модель загружена", tint = RealColors.Accent, modifier = Modifier.size(20.dp).rotate(rot))
                }
            }

            // ---------------------------------------------------------------- preview
            Box(Modifier.fillMaxWidth().aspectRatio(3f / 4f)) {
                PreviewView(vm, Modifier.fillMaxSize())
                if (settings.grid) GridOverlay()
                if (settings.level) LevelOverlay(motion.rollDeg, motion.pitchDeg, motion.deviceOrientation)
                FocusTapLayer(enabled = busy == null) { u, v -> vm.focusAt(u, v) }

                val needTripod = when (mode) {
                    CameraMode.AI -> ai?.plan?.needsTripod == true
                    CameraMode.MANUAL -> vm.shot.collectAsState().value.let { s ->
                        (s.exposureNs ?: 0L) >= 66_000_000L || (s.frames > 1 && (s.exposureNs ?: 0L) * s.frames >= 1_000_000_000L)
                    }
                    CameraMode.AUTO -> false
                }
                if (needTripod) {
                    Box(
                        Modifier.align(Alignment.TopEnd).padding(10.dp).size(40.dp).clip(CircleShape)
                            .background(RealColors.Scrim)
                            .border(2.dp, if (motion.steady) RealColors.Ok else RealColors.Warn, CircleShape),
                        contentAlignment = Alignment.Center,
                    ) {
                        Icon(SceneIcons.Tripod, "Зафиксируйте телефон", tint = if (motion.steady) RealColors.Ok else RealColors.Warn,
                            modifier = Modifier.size(24.dp).rotate(rot))
                    }
                }
                countdown?.let {
                    Text("$it", fontSize = 96.sp, fontWeight = FontWeight.Light, color = Color.White,
                        modifier = Modifier.align(Alignment.Center).rotate(rot))
                }
                busy?.let { BusyOverlay(it, Modifier.align(Alignment.BottomCenter)) }
                message?.let { m ->
                    LaunchedEffect(m) {
                        delay(3500)
                        vm.dismissMessage()
                    }
                    Text(
                        m, color = Color.White, fontSize = 13.sp,
                        modifier = Modifier.align(Alignment.TopCenter).padding(12.dp).clip(RoundedCornerShape(10.dp))
                            .background(RealColors.Scrim).padding(horizontal = 12.dp, vertical = 8.dp),
                    )
                }
            }

            // ---------------------------------------------------------------- controls
            Column(Modifier.fillMaxWidth().weight(1f), verticalArrangement = Arrangement.SpaceEvenly) {
                when (mode) {
                    CameraMode.AUTO -> {
                        EvRow(vm.autoEv.collectAsState().value, vm::setAutoEv)
                        HintRow(vm.categories.collectAsState().value, vm.categoryHint.collectAsState().value, rot, vm::setCategoryHint)
                    }
                    CameraMode.MANUAL -> {
                        ManualControls(
                            caps, vm.shot.collectAsState().value, vm.meterEv.collectAsState().value,
                            vm::setIso, vm::setShutter, vm::updateShot, vm::setFrames,
                        )
                        HintRow(vm.categories.collectAsState().value, vm.categoryHint.collectAsState().value, rot, vm::setCategoryHint)
                    }
                    CameraMode.AI -> AiControls(model, vm.modelError.collectAsState().value, ai, datasetCount, rot, vm::pickAiClass)
                }
                ModeSwitcher(mode, enabled = busy == null, onMode = vm::setMode)
                Row(
                    Modifier.fillMaxWidth().padding(horizontal = 28.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Box(
                        Modifier.size(52.dp).clip(CircleShape).background(RealColors.SurfaceHigh)
                            .clickable { onGallery(latest?.id) },
                    ) {
                        latest?.let {
                            AsyncImage(model = it.imageUri, contentDescription = "Последний снимок", contentScale = ContentScale.Crop,
                                modifier = Modifier.fillMaxSize().rotate(rot))
                        }
                    }
                    Shutter(busy, countdown != null, onClick = {
                        when {
                            busy != null -> vm.requestStop()
                            countdown != null -> vm.cancelShot()
                            else -> vm.onShutter()
                        }
                    })
                    Box(
                        Modifier.size(52.dp).clip(CircleShape).background(RealColors.SurfaceHigh)
                            .clickable(enabled = busy == null) { vm.switchCamera() },
                        contentAlignment = Alignment.Center,
                    ) {
                        Icon(Icons.Filled.FlipCameraAndroid, "Сменить камеру", tint = Color.White, modifier = Modifier.rotate(rot))
                    }
                }
            }
        }

        ratingFor?.let { id -> RatingOverlay(vm, id, onDetails = { onGallery(id) }) }
    }
}

@Composable
private fun TopIcon(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    desc: String,
    rot: Float,
    active: Boolean = false,
    badge: String? = null,
    onClick: () -> Unit,
) {
    Box(Modifier.size(44.dp).clip(CircleShape).clickable(onClick = onClick), contentAlignment = Alignment.Center) {
        Icon(icon, desc, tint = if (active || badge != null) RealColors.Accent else Color.White, modifier = Modifier.size(22.dp).rotate(rot))
        badge?.let {
            Text(it, fontSize = 9.sp, color = RealColors.Accent, modifier = Modifier.align(Alignment.BottomEnd))
        }
    }
}

@Composable
private fun PreviewView(vm: CameraViewModel, modifier: Modifier) {
    val motion by vm.motion.collectAsState()
    var view by remember { mutableStateOf<TextureView?>(null) }
    val orientation = motion.deviceOrientation
    DisposableEffect(view, orientation) {
        val v = view
        vm.sceneGrabber = if (v != null) ({ ScenePreview.grab(v, orientation) }) else null
        onDispose { vm.sceneGrabber = null }
    }
    AndroidView(
        modifier = modifier,
        factory = { ctx ->
            TextureView(ctx).apply {
                surfaceTextureListener = object : TextureView.SurfaceTextureListener {
                    override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, h: Int) = vm.onSurface(st)
                    override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, h: Int) = Unit
                    override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean {
                        vm.onSurface(null)
                        return true
                    }
                    override fun onSurfaceTextureUpdated(st: SurfaceTexture) = Unit
                }
                view = this
            }
        },
    )
}

@Composable
private fun GridOverlay() {
    Canvas(Modifier.fillMaxSize()) {
        val c = Color.White.copy(alpha = 0.35f)
        for (i in 1..2) {
            drawLine(c, Offset(size.width * i / 3f, 0f), Offset(size.width * i / 3f, size.height), 1.5f)
            drawLine(c, Offset(0f, size.height * i / 3f), Offset(size.width, size.height * i / 3f), 1.5f)
        }
    }
}

@Composable
private fun LevelOverlay(rollDeg: Double, pitchDeg: Double, orientation: Int) {
    if (!rollDeg.isFinite() || !pitchDeg.isFinite() || abs(pitchDeg) > 60) return
    var tilt = (rollDeg + orientation) % 360.0
    if (tilt > 180) tilt -= 360.0
    if (tilt < -180) tilt += 360.0
    if (abs(tilt) > 25) return
    val level = abs(tilt) < 1.0
    Canvas(Modifier.fillMaxSize()) {
        val half = size.width * 0.18f
        rotate(tilt.toFloat(), pivot = center) {
            drawLine(if (level) RealColors.Ok else Color.White.copy(alpha = 0.8f),
                Offset(center.x - half, center.y), Offset(center.x + half, center.y), if (level) 4f else 2.5f)
        }
    }
}

@Composable
private fun FocusTapLayer(enabled: Boolean, onTap: (Float, Float) -> Unit) {
    var tap by remember { mutableStateOf<Offset?>(null) }
    LaunchedEffect(tap) {
        if (tap != null) {
            delay(1200)
            tap = null
        }
    }
    val density = LocalDensity.current
    BoxWithConstraints(
        Modifier.fillMaxSize().pointerInput(enabled) {
            if (enabled) detectTapGestures { o ->
                tap = o
                onTap(o.x / size.width, o.y / size.height)
            }
        },
    ) {
        tap?.let { o ->
            val r = with(density) { 32.dp.toPx() }
            Box(
                Modifier.offset { IntOffset((o.x - r).roundToInt(), (o.y - r).roundToInt()) }
                    .size(64.dp).border(2.dp, RealColors.Accent, CircleShape),
            )
        }
    }
}

@Composable
private fun BusyOverlay(p: ShotRunner.Progress, modifier: Modifier) {
    Column(
        modifier.padding(12.dp).fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(RealColors.Scrim).padding(10.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        val (text, progress) = when (p) {
            is ShotRunner.Progress.Capturing -> {
                val exp = p.exposureNs?.let { " · " + ExposureScale.formatShutter(it) } ?: ""
                (if (p.total > 1) "Кадр ${p.frame} / ${p.total}$exp" else "Съёмка$exp") to
                    (if (p.total > 1) (p.frame - 1f) / p.total else null)
            }
            is ShotRunner.Progress.Merging -> "Сведение кадров ${(p.fraction * 100).roundToInt()}%" to p.fraction
            ShotRunner.Progress.Saving -> "Сохранение…" to null
        }
        Text(text, color = Color.White, fontSize = 13.sp)
        if (progress != null) {
            Spacer(Modifier.height(6.dp))
            LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth(), color = RealColors.Accent)
        }
    }
}

@Composable
private fun Shutter(busy: ShotRunner.Progress?, counting: Boolean, onClick: () -> Unit) {
    Box(
        Modifier.size(78.dp).clip(CircleShape).border(4.dp, Color.White, CircleShape)
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Box(
            Modifier.size(62.dp).clip(CircleShape)
                .background(if (busy != null) RealColors.SurfaceHigh else if (counting) RealColors.Bad else Color.White),
        )
        if (busy != null) CircularProgressIndicator(color = RealColors.Accent, modifier = Modifier.size(70.dp), strokeWidth = 3.dp)
        if (busy is ShotRunner.Progress.Capturing && busy.total > 1) {
            Box(Modifier.size(22.dp).clip(RoundedCornerShape(4.dp)).background(RealColors.Bad))
        }
    }
}

@Composable
private fun RatingOverlay(vm: CameraViewModel, id: String, onDetails: () -> Unit) {
    val capture by remember(id) { vm.observeCapture(id) }.collectAsState(initial = null)
    val categories by vm.categories.collectAsState()
    var newCategory by remember { mutableStateOf(false) }
    val e = capture ?: return
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.BottomCenter) {
        Column(
            Modifier.fillMaxWidth().clip(RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp))
                .background(RealColors.Surface).navigationBarsPadding().padding(14.dp)
                .verticalScroll(rememberScrollState()),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                AsyncImage(model = e.imageUri, contentDescription = null, contentScale = ContentScale.Crop,
                    modifier = Modifier.size(44.dp).clip(RoundedCornerShape(8.dp)))
                Spacer(Modifier.size(10.dp))
                Column(Modifier.weight(1f)) {
                    Text("Как получилось?", color = Color.White, fontWeight = FontWeight.SemiBold)
                    Text(completenessText(e), color = if (e.inDataset) RealColors.Ok else RealColors.TextDim, fontSize = 11.sp)
                }
                TextButton(onClick = onDetails) { Text("В галерее") }
            }
            Spacer(Modifier.height(6.dp))
            RatingPanel(e, categories, onChange = { change -> vm.rate(id, change) }, onAddCategory = { newCategory = true })
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                TextButton(onClick = { vm.dismissRating() }) { Text(if (e.isRatingComplete) "Готово" else "Позже") }
            }
        }
    }
    if (newCategory) {
        CategoryDialog(
            title = "Новая категория",
            onDismiss = { newCategory = false },
            onConfirm = { name, icon ->
                newCategory = false
                vm.createCategory(name, icon, assignTo = id)
            },
        )
    }
}
