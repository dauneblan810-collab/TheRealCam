package com.therealcam.ui.camera

import androidx.compose.animation.core.animateFloatAsState
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
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.therealcam.ai.engine.TrcmModel
import com.therealcam.camera.CameraCaps
import com.therealcam.camera.CameraMode
import com.therealcam.camera.ExposureScale
import com.therealcam.camera.FocusMode
import com.therealcam.camera.ShotSettings
import com.therealcam.camera.WbPreset
import com.therealcam.data.StackMode
import com.therealcam.data.db.CategoryEntity
import com.therealcam.dataset.MIN_DATASET_SAMPLES
import com.therealcam.ui.common.SceneIcons
import com.therealcam.ui.theme.RealColors
import kotlin.math.roundToInt

@Composable
fun rotationFor(deviceOrientation: Int): Float {
    val target = when (deviceOrientation) {
        90 -> -90f
        270 -> 90f
        180 -> 180f
        else -> 0f
    }
    val v by animateFloatAsState(target, label = "iconRotation")
    return v
}

@Composable
fun ModeSwitcher(mode: CameraMode, enabled: Boolean, onMode: (CameraMode) -> Unit) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
        CameraMode.entries.forEach { m ->
            val sel = m == mode
            Text(
                m.label,
                color = if (sel) RealColors.Accent else Color.White.copy(alpha = if (enabled) 0.75f else 0.35f),
                fontWeight = if (sel) FontWeight.Bold else FontWeight.Medium,
                fontSize = 14.sp,
                modifier = Modifier
                    .clip(RoundedCornerShape(16.dp))
                    .clickable(enabled = enabled) { onMode(m) }
                    .padding(horizontal = 16.dp, vertical = 8.dp),
            )
        }
    }
}

@Composable
private fun Chip(text: String, selected: Boolean, modifier: Modifier = Modifier, enabled: Boolean = true, onClick: () -> Unit) {
    Box(
        modifier
            .clip(RoundedCornerShape(14.dp))
            .background(if (selected) RealColors.Accent else RealColors.SurfaceHigh.copy(alpha = 0.9f))
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 10.dp, vertical = 6.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            text, fontSize = 12.sp, maxLines = 1,
            color = when {
                !enabled -> Color.White.copy(alpha = 0.35f)
                selected -> Color.Black
                else -> Color.White
            },
        )
    }
}

/** Horizontal list of values; the selected one is scrolled into view. */
@Composable
private fun <T> ValueRow(values: List<T>, selected: T, label: (T) -> String, onPick: (T) -> Unit) {
    val state = rememberLazyListState()
    val idx = values.indexOf(selected)
    LaunchedEffect(idx) { if (idx >= 0) state.animateScrollToItem((idx - 2).coerceAtLeast(0)) }
    LazyRow(state = state, horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.fillMaxWidth()) {
        itemsIndexed(values) { _, v -> Chip(label(v), v == selected) { onPick(v) } }
    }
}

enum class Dial(val title: String) { ISO("ISO"), SHUTTER("Выдержка"), FOCUS("Фокус"), WB("ББ"), EV("EV"), FRAMES("Кадры") }

@Composable
fun ManualControls(
    caps: CameraCaps?,
    shot: ShotSettings,
    meterEv: Double?,
    onIso: (Int?) -> Unit,
    onShutter: (Long?) -> Unit,
    onShot: ((ShotSettings) -> ShotSettings) -> Unit,
    onFrames: (Int) -> Unit,
) {
    var dial by remember { mutableStateOf<Dial?>(null) }
    val manualOk = caps?.manualSensor == true
    Column(Modifier.fillMaxWidth().padding(horizontal = 10.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            fun v(d: Dial): String = when (d) {
                Dial.ISO -> shot.iso?.toString() ?: "A"
                Dial.SHUTTER -> ExposureScale.formatShutter(shot.exposureNs)
                Dial.FOCUS -> when (shot.focus) {
                    FocusMode.AUTO -> "AF"
                    FocusMode.INFINITY -> "∞"
                    FocusMode.MANUAL -> if (shot.focusDiopters <= 0.01f) "∞" else "%.1fм".format(1f / shot.focusDiopters)
                }
                Dial.WB -> shot.wb.label
                Dial.EV -> if (shot.manualExposure) "—" else "%+.1f".format(shot.ev)
                Dial.FRAMES -> if (shot.frames <= 1) "1" else "${shot.frames}"
            }
            Dial.entries.forEach { d ->
                val enabled = when (d) {
                    Dial.ISO, Dial.SHUTTER -> manualOk
                    Dial.FOCUS -> caps?.canManualFocus == true
                    Dial.EV -> !shot.manualExposure
                    else -> true
                }
                Column(
                    Modifier
                        .weight(1f)
                        .clip(RoundedCornerShape(10.dp))
                        .background(if (dial == d) RealColors.Accent.copy(alpha = 0.25f) else Color.Transparent)
                        .clickable(enabled = enabled) { dial = if (dial == d) null else d }
                        .padding(vertical = 4.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text(d.title, fontSize = 10.sp, color = if (enabled) RealColors.TextDim else RealColors.TextDim.copy(alpha = 0.4f), maxLines = 1)
                    Text(v(d), fontSize = 13.sp, fontWeight = FontWeight.SemiBold, maxLines = 1,
                        color = if (enabled) Color.White else Color.White.copy(alpha = 0.35f))
                }
            }
        }
        Spacer(Modifier.height(6.dp))
        Box(Modifier.fillMaxWidth().height(36.dp), contentAlignment = Alignment.CenterStart) {
            when (dial) {
                Dial.ISO -> ValueRow(listOf<Int?>(null) + ExposureScale.isoValues(caps), shot.iso, { it?.toString() ?: "Авто" }, onIso)
                Dial.SHUTTER -> ValueRow(listOf<Long?>(null) + ExposureScale.shutterValuesNs(caps), shot.exposureNs,
                    { if (it == null) "Авто" else ExposureScale.formatShutter(it) }, onShutter)
                Dial.FOCUS -> FocusDial(caps, shot, onShot)
                Dial.WB -> ValueRow(WbPreset.entries.toList(), shot.wb, { it.label }) { wb -> onShot { it.copy(wb = wb) } }
                Dial.EV -> {
                    val steps = (-9..9).map { it / 3f }
                    ValueRow(steps, steps.minBy { kotlin.math.abs(it - shot.ev) }, { "%+.1f".format(it) }) { ev -> onShot { it.copy(ev = ev) } }
                }
                Dial.FRAMES -> FramesDial(shot, onFrames, onShot)
                null -> Text(
                    meterEv?.let { "Экспозамер: %+.1f EV".format(it) + if (!manualOk) " · ручная экспозиция не поддерживается камерой" else "" }
                        ?: if (!manualOk) "Камера не поддерживает ручную экспозицию" else "Выберите параметр",
                    color = RealColors.TextDim, fontSize = 12.sp,
                )
            }
        }
    }
}

@Composable
private fun FocusDial(caps: CameraCaps?, shot: ShotSettings, onShot: ((ShotSettings) -> ShotSettings) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Chip("AF", shot.focus == FocusMode.AUTO) { onShot { it.copy(focus = FocusMode.AUTO) } }
        Spacer(Modifier.width(6.dp))
        Chip("∞", shot.focus == FocusMode.INFINITY) { onShot { it.copy(focus = FocusMode.INFINITY, focusDiopters = 0f) } }
        Spacer(Modifier.width(6.dp))
        val maxD = caps?.minFocusDistance ?: 0f
        if (maxD > 0f) {
            Slider(
                value = if (shot.focus == FocusMode.MANUAL) shot.focusDiopters else 0f,
                onValueChange = { d -> onShot { it.copy(focus = FocusMode.MANUAL, focusDiopters = d) } },
                valueRange = 0f..maxD,
                modifier = Modifier.weight(1f),
                colors = SliderDefaults.colors(thumbColor = RealColors.Accent, activeTrackColor = RealColors.Accent),
            )
        }
    }
}

@Composable
private fun FramesDial(shot: ShotSettings, onFrames: (Int) -> Unit, onShot: ((ShotSettings) -> ShotSettings) -> Unit) {
    val counts = listOf(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
    val modes = listOf(
        StackMode.MEAN to "Среднее", StackMode.SIGMA_CLIP to "Без выбросов", StackMode.LUCKY to "Лучшие",
        StackMode.LIGHTEN to "Следы", StackMode.BEST_OF to "Самый резкий",
    )
    LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.fillMaxWidth()) {
        itemsIndexed(counts) { _, n -> Chip("$n", n == shot.frames) { onFrames(n) } }
        if (shot.frames > 1) itemsIndexed(modes) { _, (m, label) ->
            Chip(label, m == shot.stackMode) { onShot { it.copy(stackMode = m) } }
        }
    }
}

/** Auto/manual: optional "what am I shooting" hint, stored with the photo as its category. */
@Composable
fun HintRow(categories: List<CategoryEntity>, hint: String?, orientation: Float, onHint: (String) -> Unit) {
    if (categories.isEmpty()) return
    LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.fillMaxWidth().padding(horizontal = 10.dp)) {
        itemsIndexed(categories) { _, c ->
            val sel = c.id == hint
            Box(
                Modifier
                    .size(34.dp)
                    .clip(CircleShape)
                    .background(if (sel) RealColors.Accent else RealColors.SurfaceHigh)
                    .clickable { onHint(c.id) },
                contentAlignment = Alignment.Center,
            ) {
                Icon(SceneIcons.of(c.icon), c.name, tint = if (sel) Color.Black else Color.White, modifier = Modifier.size(18.dp).rotate(orientation))
            }
        }
    }
}

@Composable
fun EvRow(ev: Float, onEv: (Float) -> Unit) {
    Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp), verticalAlignment = Alignment.CenterVertically) {
        Text("EV", color = RealColors.TextDim, fontSize = 12.sp)
        Slider(
            value = ev, onValueChange = { onEv((it * 3).roundToInt() / 3f) }, valueRange = -3f..3f,
            modifier = Modifier.weight(1f).padding(horizontal = 8.dp),
            colors = SliderDefaults.colors(thumbColor = RealColors.Accent, activeTrackColor = RealColors.Accent),
        )
        Text("%+.1f".format(ev), color = Color.White, fontSize = 12.sp, modifier = Modifier.width(36.dp))
    }
}

@Composable
fun AiControls(
    model: TrcmModel?,
    modelError: String?,
    ai: AiState?,
    datasetCount: Int,
    orientation: Float,
    onPick: (Int) -> Unit,
) {
    Column(Modifier.fillMaxWidth().padding(horizontal = 12.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        if (model == null) {
            Text(
                modelError ?: if (datasetCount < MIN_DATASET_SAMPLES)
                    "Нет модели. Датасет: $datasetCount / $MIN_DATASET_SAMPLES оценённых снимков"
                else "Нет модели. Датасет готов ($datasetCount) — обучите модель на ПК и загрузите её в настройках",
                color = RealColors.TextDim, fontSize = 12.sp,
            )
            return
        }
        if (ai == null) {
            Text("Смотрю на сцену…", color = RealColors.TextDim, fontSize = 12.sp)
            return
        }
        // Up to three guesses; the chosen one is highlighted. Tap to tell the AI what you shoot.
        val shown = ai.ranking.take(3).filter { it == ai.ranking.first() || ai.probs[it] >= 0.08f }.toMutableList()
        if (ai.chosen !in shown) shown.add(ai.chosen)
        Row(horizontalArrangement = Arrangement.spacedBy(14.dp), verticalAlignment = Alignment.CenterVertically) {
            shown.forEach { k ->
                val c = model.classes[k]
                val sel = k == ai.chosen
                Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.clickable { onPick(k) }) {
                    Box(
                        Modifier
                            .size(if (sel) 46.dp else 38.dp)
                            .clip(CircleShape)
                            .background(if (sel) RealColors.Accent else RealColors.SurfaceHigh)
                            .border(if (sel && ai.userOverride) 2.dp else 0.dp, Color.White, CircleShape),
                        contentAlignment = Alignment.Center,
                    ) {
                        Icon(SceneIcons.of(c.icon), c.name, tint = if (sel) Color.Black else Color.White,
                            modifier = Modifier.size(if (sel) 26.dp else 20.dp).rotate(orientation))
                    }
                    Text("${(ai.probs[k] * 100).roundToInt()}%", fontSize = 10.sp, color = if (sel) RealColors.Accent else RealColors.TextDim)
                }
            }
        }
        ai.plan?.let { p ->
            Spacer(Modifier.height(2.dp))
            Text("${p.cls.name} · ${p.label}", color = Color.White, fontSize = 12.sp, maxLines = 1)
        }
    }
}
