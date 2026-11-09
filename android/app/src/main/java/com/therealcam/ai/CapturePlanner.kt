package com.therealcam.ai

import com.therealcam.ai.engine.ModelClass
import com.therealcam.ai.engine.ModelOutput
import com.therealcam.ai.engine.TrcmModel
import com.therealcam.camera.CameraCaps
import com.therealcam.camera.ExposureScale
import com.therealcam.camera.FocusMode
import com.therealcam.camera.ShotSettings
import com.therealcam.data.Profile
import com.therealcam.data.SceneKeys
import com.therealcam.data.StackMode
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.roundToLong

/** What the AI intends to shoot for one scene class. */
data class CapturePlan(
    val classIndex: Int,
    val cls: ModelClass,
    val profile: Profile,
    val settings: ShotSettings,
    val needsTripod: Boolean,
    val label: String,
    val rawParams: Map<String, Double>,
)

/** Turns model outputs into camera settings, clamped to what this phone can do. */
object CapturePlanner {
    /** Handheld exposures longer than this blur; the tripod icon is shown. */
    const val HANDHELD_MAX_S = 1.0 / 15.0

    fun plan(model: TrcmModel, out: ModelOutput, k: Int, caps: CameraCaps): CapturePlan {
        val cls = model.classes[k]
        val profile = SceneKeys.profileOf(cls.icon)
        val p = out.paramsOf(k)

        var frames = p["log2_frames"]?.let { 2.0.pow(it).roundToInt() } ?: profile.defaultFrames
        frames = frames.coerceIn(1, profile.maxFrames)
        var exposureNs = p["log2_exposure_s"]?.let { (2.0.pow(it) * 1e9).roundToLong() }
        var iso = p["log2_iso"]?.let { 2.0.pow(it).roundToInt() }
        if (!caps.manualSensor || exposureNs == null || iso == null) {
            exposureNs = null
            iso = null
        } else {
            caps.exposureRange?.let { r ->
                if (exposureNs!! > r.upper && frames < profile.maxFrames) {
                    // The sensor cannot expose that long: keep the total light by adding frames.
                    val factor = exposureNs!!.toDouble() / r.upper
                    frames = (frames * factor).roundToInt().coerceIn(frames, profile.maxFrames)
                }
                exposureNs = exposureNs!!.coerceIn(r.lower, r.upper)
            }
            caps.isoRange?.let { r -> iso = iso!!.coerceIn(r.lower, r.upper) }
        }
        val focusInf = p["focus_infinity"]?.let { 1.0 / (1.0 + exp(-it)) >= 0.5 } ?: profile.focusInfinity
        val ev = (p["ev_bias"]?.toFloat() ?: 0f).coerceIn(caps.aeCompRange.lower * caps.aeCompStep.toFloat(), caps.aeCompRange.upper * caps.aeCompStep.toFloat())
        val stack = if (frames > 1) profile.stackMode else StackMode.NONE
        val settings = ShotSettings(
            iso = iso, exposureNs = exposureNs,
            focus = if (focusInf && caps.canManualFocus) FocusMode.INFINITY else FocusMode.AUTO,
            ev = if (exposureNs == null) ev else 0f,
            frames = frames, stackMode = stack,
        )
        val perFrameS = (exposureNs ?: 0L) / 1e9
        val needsTripod = perFrameS >= HANDHELD_MAX_S || (frames > 1 && perFrameS * frames >= 1.0)
        val label = buildString {
            if (frames > 1) append("$frames×")
            append(if (exposureNs != null) ExposureScale.formatShutter(exposureNs) else "авто")
            iso?.let { append(" · ISO$it") }
            if (settings.focus == FocusMode.INFINITY) append(" · ∞")
            if (exposureNs == null && ev != 0f) append(" · EV%+.1f".format(ev))
        }
        return CapturePlan(k, cls, profile, settings, needsTripod, label, p)
    }
}
