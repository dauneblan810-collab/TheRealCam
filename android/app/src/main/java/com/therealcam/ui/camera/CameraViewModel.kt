package com.therealcam.ui.camera

import android.app.Application
import android.graphics.SurfaceTexture
import android.media.MediaActionSound
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.therealcam.ai.CapturePlan
import com.therealcam.ai.CapturePlanner
import com.therealcam.ai.SceneFeatures
import com.therealcam.ai.ScenePreview
import com.therealcam.ai.engine.ModelOutput
import com.therealcam.ai.engine.TrcmModel
import com.therealcam.appGraph
import com.therealcam.camera.CameraController
import com.therealcam.camera.CameraMode
import com.therealcam.camera.ExposureScale
import com.therealcam.camera.FocusMode
import com.therealcam.camera.ShotSettings
import com.therealcam.capture.ShotRunner
import com.therealcam.data.AppSettings
import com.therealcam.data.Profile
import com.therealcam.data.SceneKeys
import com.therealcam.data.StackMode
import com.therealcam.data.db.CaptureEntity
import com.therealcam.data.db.CategoryEntity
import com.therealcam.dataset.AiInfo
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.TimeZone
import kotlin.math.ln

/** AI suggestions for the current scene. */
data class AiState(
    val probs: FloatArray,
    val ranking: List<Int>,
    val chosen: Int,
    val userOverride: Boolean,
    val plan: CapturePlan?,
)

class CameraViewModel(app: Application) : AndroidViewModel(app) {
    private val g = app.appGraph
    val camera = CameraController(app)
    private val runner = ShotRunner(g, camera)
    private val shutterSound = MediaActionSound().apply { load(MediaActionSound.SHUTTER_CLICK) }

    val settings: StateFlow<AppSettings> = g.settings.settings.stateIn(viewModelScope, SharingStarted.Eagerly, AppSettings())
    val categories: StateFlow<List<CategoryEntity>> = g.categories.all.stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())
    val latest: StateFlow<CaptureEntity?> = g.captures.latest.stateIn(viewModelScope, SharingStarted.Eagerly, null)
    val datasetCount: StateFlow<Int> = g.captures.datasetCount.stateIn(viewModelScope, SharingStarted.Eagerly, 0)
    val model: StateFlow<TrcmModel?> = g.models.active
    val modelError: StateFlow<String?> = g.models.activeError
    val motion = g.sensors.state
    val location = g.location.location

    private val _mode = MutableStateFlow(CameraMode.AUTO)
    val mode = _mode.asStateFlow()
    private val _shot = MutableStateFlow(ShotSettings())
    val shot = _shot.asStateFlow()
    private val _autoEv = MutableStateFlow(0f)
    val autoEv = _autoEv.asStateFlow()
    private val _torch = MutableStateFlow(false)
    val torch = _torch.asStateFlow()
    private val _categoryHint = MutableStateFlow<String?>(null)
    val categoryHint = _categoryHint.asStateFlow()
    private val _front = MutableStateFlow(false)
    val front = _front.asStateFlow()
    private val _busy = MutableStateFlow<ShotRunner.Progress?>(null)
    val busy = _busy.asStateFlow()
    private val _countdown = MutableStateFlow<Int?>(null)
    val countdown = _countdown.asStateFlow()
    private val _ratingFor = MutableStateFlow<String?>(null)
    val ratingFor = _ratingFor.asStateFlow()
    private val _message = MutableStateFlow<String?>(null)
    val message = _message.asStateFlow()
    private val _ai = MutableStateFlow<AiState?>(null)
    val ai = _ai.asStateFlow()
    /** Exposure meter of the last analysed preview, EV relative to mid-grey. */
    private val _meterEv = MutableStateFlow<Double?>(null)
    val meterEv = _meterEv.asStateFlow()

    /** Provided by the UI while the preview view exists (main thread only). */
    var sceneGrabber: (() -> ScenePreview?)? = null

    private var texture: SurfaceTexture? = null
    private var resumed = false
    private var analysisJob: Job? = null
    private var shotJob: Job? = null
    private var smoothed: FloatArray? = null
    private var smoothedFor: TrcmModel? = null

    init {
        viewModelScope.launch {
            val s = g.settings.settings.first()
            _mode.value = CameraMode.of(s.lastMode)
            _categoryHint.value = s.lastCategoryHint
            applyPreviewSettings()
        }
    }

    // ------------------------------------------------------------ lifecycle

    fun onSurface(t: SurfaceTexture?) {
        texture = t
        if (t == null) viewModelScope.launch { camera.close() } else openIfReady()
    }

    fun onResume() {
        if (resumed) return
        resumed = true
        g.sensors.acquire()
        if (settings.value.geotag) g.location.start()
        openIfReady()
        startAnalysis()
    }

    fun onPause() {
        if (!resumed) return
        resumed = false
        analysisJob?.cancel()
        g.sensors.release()
        g.location.stop()
        viewModelScope.launch { camera.close() }
    }

    fun observeCapture(id: String) = g.captures.observe(id)

    fun startLocation() {
        if (resumed && settings.value.geotag) g.location.start()
    }

    private fun openIfReady() {
        val t = texture ?: return
        if (!resumed) return
        viewModelScope.launch {
            try {
                val (back, frontId) = camera.cameraIds()
                val id = (if (_front.value) frontId else back) ?: back ?: frontId ?: run {
                    _message.value = "Камера не найдена"
                    return@launch
                }
                camera.open(id, t)
                applyPreviewSettings()
            } catch (e: Exception) {
                _message.value = "Не удалось открыть камеру: ${e.message}"
            }
        }
    }

    override fun onCleared() {
        camera.release()
        shutterSound.release()
    }

    // ------------------------------------------------------------ controls

    fun setMode(m: CameraMode) {
        if (_mode.value == m) return
        _mode.value = m
        _ai.value = null
        smoothed = null
        viewModelScope.launch { g.settings.setLastMode(m.key) }
        applyPreviewSettings()
    }

    fun switchCamera() {
        _front.value = !_front.value
        _shot.value = ShotSettings(frames = _shot.value.frames, stackMode = _shot.value.stackMode)
        openIfReady()
    }

    fun setAutoEv(ev: Float) {
        _autoEv.value = ev
        applyPreviewSettings()
    }

    fun toggleTorch() {
        _torch.value = !_torch.value
        applyPreviewSettings()
    }

    fun setCategoryHint(id: String?) {
        _categoryHint.value = if (_categoryHint.value == id) null else id
        viewModelScope.launch { g.settings.setLastCategoryHint(_categoryHint.value) }
        val cat = categories.value.firstOrNull { it.id == _categoryHint.value }
        // A hint with a multi-frame profile proposes that profile's merge mode for manual series.
        if (cat != null && _shot.value.frames > 1) {
            val p = SceneKeys.profileOf(cat.icon)
            if (p.stackMode != StackMode.NONE) updateShot { it.copy(stackMode = p.stackMode) }
        }
    }

    fun updateShot(change: (ShotSettings) -> ShotSettings) {
        _shot.value = change(_shot.value)
        applyPreviewSettings()
    }

    /** Manual ISO; picking one while shutter is auto fixes the shutter at the current metered value. */
    fun setIso(iso: Int?) = updateShot { s ->
        if (iso == null) s.copy(iso = null, exposureNs = null)
        else s.copy(iso = iso, exposureNs = s.exposureNs ?: meteredShutter())
    }

    fun setShutter(ns: Long?) = updateShot { s ->
        if (ns == null) s.copy(iso = null, exposureNs = null)
        else s.copy(exposureNs = ns, iso = s.iso ?: meteredIso())
    }

    private fun meteredShutter(): Long {
        val caps = camera.caps.value
        val p = camera.preview.value?.exposureNs ?: 16_666_666L
        return ExposureScale.nearest(ExposureScale.shutterValuesNs(caps), p)
    }

    private fun meteredIso(): Int {
        val caps = camera.caps.value
        val p = camera.preview.value?.iso ?: 100
        return ExposureScale.nearestIso(ExposureScale.isoValues(caps), p)
    }

    fun setFrames(n: Int) = updateShot { s ->
        val mode = when {
            n <= 1 -> StackMode.NONE
            s.stackMode != StackMode.NONE -> s.stackMode
            else -> hintProfile().stackMode.takeIf { it != StackMode.NONE } ?: StackMode.MEAN
        }
        s.copy(frames = n.coerceIn(1, 64), stackMode = mode)
    }

    private fun hintProfile(): Profile =
        categories.value.firstOrNull { it.id == _categoryHint.value }?.let { SceneKeys.profileOf(it.icon) } ?: Profile.DEFAULT

    fun focusAt(u: Float, v: Float) {
        if (_mode.value == CameraMode.MANUAL && _shot.value.focus != FocusMode.AUTO) return
        camera.focusAt(u, v)
    }

    fun pickAiClass(k: Int) {
        val st = _ai.value ?: return
        val m = model.value ?: return
        val caps = camera.caps.value ?: return
        val override = !(st.userOverride && st.chosen == k)
        _ai.value = st.copy(chosen = k, userOverride = override, plan = CapturePlanner.plan(m, lastOutput ?: return, k, caps))
    }

    fun dismissMessage() {
        _message.value = null
    }

    fun dismissRating() {
        _ratingFor.value = null
    }

    fun rate(id: String, change: (CaptureEntity) -> CaptureEntity) {
        viewModelScope.launch { g.captures.rate(id, change) }
    }

    fun createCategory(name: String, icon: String, assignTo: String?) {
        viewModelScope.launch {
            val c = g.categories.create(name, icon)
            if (assignTo != null) g.captures.rate(assignTo) { it.copy(categoryId = c.id) }
        }
    }

    private fun previewSettings(): ShotSettings = when (_mode.value) {
        CameraMode.AUTO -> ShotSettings(ev = _autoEv.value, torch = _torch.value)
        CameraMode.MANUAL -> _shot.value.copy(torch = _torch.value)
        CameraMode.AI -> ShotSettings(torch = _torch.value)
    }

    private fun applyPreviewSettings() = camera.setShot(previewSettings())

    // ------------------------------------------------------------ analysis (meter + AI)

    private var lastOutput: ModelOutput? = null

    private fun featureContext(): SceneFeatures.Context {
        val p = camera.preview.value
        val m = motion.value
        val loc = if (settings.value.geotag) location.value else null
        val now = System.currentTimeMillis()
        return SceneFeatures.Context(
            previewExposureNs = p?.exposureNs, previewIso = p?.iso, aperture = p?.aperture, lux = m.lux,
            pitchDeg = m.pitchDeg, rollDeg = m.rollDeg, gyroRms = m.gyroRms, timeMs = now,
            tzOffsetMs = TimeZone.getDefault().getOffset(now), lat = loc?.latitude, lon = loc?.longitude,
            focusDiopters = p?.focusDiopters, zoom = p?.zoomRatio, front = camera.caps.value?.isFront == true,
        )
    }

    private fun startAnalysis() {
        analysisJob?.cancel()
        analysisJob = viewModelScope.launch {
            while (isActive) {
                delay(if (_mode.value == CameraMode.AI) 450L else 800L)
                if (_busy.value != null || _mode.value == CameraMode.AUTO) continue
                val caps = camera.caps.value ?: continue
                val scene = sceneGrabber?.invoke() ?: continue
                val ctx = featureContext()
                val m = model.value
                val result = withContext(Dispatchers.Default) {
                    val stats = SceneFeatures.imageStats(scene.px, scene.w, scene.h)
                    val meter = ln(maxOf(stats.meanLin, 1e-5) / 0.18) / ln(2.0)
                    val out = if (_mode.value == CameraMode.AI && m != null) {
                        runCatching { m.run(scene.thumb, SceneFeatures.compute(scene.px, scene.w, scene.h, ctx)) }.getOrNull()
                    } else null
                    meter to out
                }
                _meterEv.value = result.first
                val out = result.second
                if (out == null || m == null || _mode.value != CameraMode.AI) continue
                lastOutput = out
                val prev = smoothed?.takeIf { smoothedFor === m && it.size == out.probs.size }
                val sm = if (prev == null) out.probs.copyOf() else FloatArray(out.probs.size) { 0.6f * prev[it] + 0.4f * out.probs[it] }
                smoothed = sm
                smoothedFor = m
                val ranking = sm.indices.sortedByDescending { sm[it] }
                val st = _ai.value
                val chosen = if (st != null && st.userOverride && st.chosen in sm.indices) st.chosen else ranking.first()
                _ai.value = AiState(sm, ranking, chosen, st?.userOverride == true && st.chosen == chosen, CapturePlanner.plan(m, out, chosen, caps))
            }
        }
    }

    // ------------------------------------------------------------ shutter

    fun onShutter() {
        if (_busy.value != null || _countdown.value != null) return
        val caps = camera.caps.value ?: return
        _ratingFor.value = null
        shotJob = viewModelScope.launch {
            val app = settings.value
            for (s in app.timerSeconds downTo 1) {
                _countdown.value = s
                delay(1000)
            }
            _countdown.value = null
            val scene = sceneGrabber?.invoke()
            val ctx = featureContext()
            val features = scene?.let { s -> withContext(Dispatchers.Default) { SceneFeatures.compute(s.px, s.w, s.h, ctx) } }
                ?: SceneFeatures.NAMES.associateWith { null }
            val plan = buildPlan(app)
            _busy.value = ShotRunner.Progress.Capturing(0, plan.settings.frames, plan.settings.exposureNs)
            try {
                val env = ShotRunner.Env(caps, camera.preview.value, scene, features, motion.value, location.value, app)
                stopRequested = false
                shutterSound.play(MediaActionSound.SHUTTER_CLICK)
                val e = withContext(Dispatchers.Default) { runner.run(plan, env, stop = { stopRequested }) { _busy.value = it } }
                if (app.datasetMode && app.askRating) _ratingFor.value = e.id
            } catch (t: Throwable) {
                if (t !is kotlinx.coroutines.CancellationException) _message.value = "Не удалось снять: ${t.message}"
            } finally {
                _busy.value = null
                applyPreviewSettings()
            }
        }
    }

    @Volatile private var stopRequested = false

    /** Second press during a series: stop after the current frame and merge what was taken. */
    fun requestStop() {
        val b = _busy.value
        if (b is ShotRunner.Progress.Capturing && b.total > 1 && !stopRequested) {
            stopRequested = true
            _message.value = "Останавливаю серию после текущего кадра…"
        }
    }

    fun cancelShot() {
        if (_countdown.value != null) {
            shotJob?.cancel()
            _countdown.value = null
        }
    }

    private fun buildPlan(app: AppSettings): ShotRunner.Plan {
        val hint = _categoryHint.value?.takeIf { id -> categories.value.any { it.id == id } }
        return when (_mode.value) {
            CameraMode.AUTO -> ShotRunner.Plan(
                CameraMode.AUTO, ShotSettings(ev = _autoEv.value, torch = _torch.value), hintProfile(), hint, null, app.timerSeconds,
            )
            CameraMode.MANUAL -> ShotRunner.Plan(
                CameraMode.MANUAL, _shot.value.copy(torch = _torch.value), hintProfile(), hint, null, app.timerSeconds,
            )
            CameraMode.AI -> {
                val m = model.value
                val st = _ai.value
                val plan = st?.plan
                if (m == null || st == null || plan == null) {
                    ShotRunner.Plan(CameraMode.AI, ShotSettings(torch = _torch.value), Profile.DEFAULT, hint, null, app.timerSeconds)
                } else {
                    val userCategory = categories.value.firstOrNull { it.id == plan.cls.id }?.id
                    val info = AiInfo(
                        modelName = m.name, modelSha256 = m.sha256,
                        probs = m.classes.indices.associate { m.classes[it].id to st.probs[it].toDouble() },
                        chosenClass = plan.cls.id, userOverride = st.userOverride, params = plan.rawParams,
                    )
                    ShotRunner.Plan(CameraMode.AI, plan.settings.copy(torch = _torch.value), plan.profile, userCategory ?: hint, info, app.timerSeconds)
                }
            }
        }
    }

    fun setTimer(seconds: Int) = viewModelScope.launch { g.settings.setTimerSeconds(seconds) }
    fun setGrid(on: Boolean) = viewModelScope.launch { g.settings.setGrid(on) }
}
