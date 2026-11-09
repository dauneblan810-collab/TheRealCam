package com.therealcam.camera

import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureResult
import android.location.Location
import com.therealcam.data.StackMode
import kotlin.math.abs
import kotlin.math.roundToLong

enum class CameraMode(val key: String, val label: String) {
    AUTO("auto", "АВТО"),
    MANUAL("manual", "РУЧНОЙ"),
    AI("ai", "AI");

    companion object {
        fun of(key: String?) = entries.firstOrNull { it.key == key } ?: AUTO
    }
}

enum class FocusMode(val key: String) { AUTO("auto"), INFINITY("infinity"), MANUAL("manual") }

enum class WbPreset(val key: String, val label: String, val awbMode: Int) {
    AUTO("auto", "Авто", CameraMetadata.CONTROL_AWB_MODE_AUTO),
    DAYLIGHT("daylight", "Солнце", CameraMetadata.CONTROL_AWB_MODE_DAYLIGHT),
    CLOUDY("cloudy", "Облачно", CameraMetadata.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT),
    SHADE("shade", "Тень", CameraMetadata.CONTROL_AWB_MODE_SHADE),
    TWILIGHT("twilight", "Сумерки", CameraMetadata.CONTROL_AWB_MODE_TWILIGHT),
    FLUORESCENT("fluorescent", "Лампа дн.", CameraMetadata.CONTROL_AWB_MODE_FLUORESCENT),
    INCANDESCENT("incandescent", "Лампа нак.", CameraMetadata.CONTROL_AWB_MODE_INCANDESCENT),
}

/**
 * What the next shot should use. `iso`/`exposureNs` both null = auto exposure (with [ev]).
 * When the user sets either one in manual mode, both become manual (Camera2 has no priority modes).
 */
data class ShotSettings(
    val iso: Int? = null,
    val exposureNs: Long? = null,
    val focus: FocusMode = FocusMode.AUTO,
    val focusDiopters: Float = 0f,
    val wb: WbPreset = WbPreset.AUTO,
    val ev: Float = 0f,
    val torch: Boolean = false,
    val frames: Int = 1,
    val stackMode: StackMode = StackMode.NONE,
) {
    val manualExposure get() = iso != null && exposureNs != null
}

/** Latest preview frame metadata (what auto exposure / focus currently see). */
data class PreviewSnapshot(
    val exposureNs: Long?,
    val iso: Int?,
    val aperture: Float?,
    val focusDiopters: Float?,
    val aeState: Int?,
    val afState: Int?,
    val timestampNs: Long,
    val zoomRatio: Float?,
) {
    val aeStateName get() = aeStateName(aeState)
    val afStateName get() = afStateName(afState)
}

/** One frame to capture. */
data class StillRequest(
    val settings: ShotSettings,
    val jpegOrientation: Int,
    val jpegQuality: Int,
    val location: Location?,
)

/** Metadata of a captured frame, read from TotalCaptureResult. */
data class FrameMetadata(
    val exposureNs: Long?,
    val iso: Int?,
    val aperture: Float?,
    val focalLengthMm: Float?,
    val focusDiopters: Float?,
    val aeMode: String?,
    val afMode: String?,
    val awbMode: String?,
    val wbGains: List<Float>?,
    val evCompensation: Float?,
    val flash: String?,
    val ois: String?,
    val zoomRatio: Float?,
    val frameDurationNs: Long?,
    val noiseReduction: String?,
    val sensorTimestampNs: Long?,
)

class CapturedFrame(
    val jpeg: ByteArray,
    val meta: FrameMetadata,
    /** Exposure window in elapsedRealtimeNanos. */
    val exposureStartRt: Long,
    val exposureEndRt: Long,
)

object ExposureScale {
    val isoStops = listOf(50, 64, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3200, 4000, 5000, 6400, 8000, 10000, 12800)

    /** Shutter speeds in seconds. */
    val shutterStops: List<Double> = listOf(
        1 / 8000.0, 1 / 4000.0, 1 / 2000.0, 1 / 1000.0, 1 / 500.0, 1 / 250.0, 1 / 125.0, 1 / 60.0, 1 / 30.0,
        1 / 15.0, 1 / 8.0, 1 / 4.0, 1 / 2.0, 0.8, 1.0, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0, 15.0, 20.0, 25.0, 30.0,
    )

    fun isoValues(caps: CameraCaps?): List<Int> {
        val r = caps?.isoRange ?: return isoStops.filter { it in 100..3200 }
        val v = isoStops.filter { it in r.lower..r.upper }
        return v.ifEmpty { listOf(r.lower, r.upper).distinct() }
    }

    fun shutterValuesNs(caps: CameraCaps?): List<Long> {
        val r = caps?.exposureRange
        val all = shutterStops.map { (it * 1e9).roundToLong() }
        if (r == null) return all.filter { it <= 1_000_000_000L }
        val v = all.filter { it in r.lower..r.upper }
        return v.ifEmpty { listOf(r.lower, r.upper).distinct() }
    }

    fun nearest(values: List<Long>, target: Long): Long = values.minBy { abs(it - target) }
    fun nearestIso(values: List<Int>, target: Int): Int = values.minBy { abs(it - target) }

    fun formatShutter(ns: Long?): String {
        if (ns == null) return "A"
        val s = ns / 1e9
        return when {
            s >= 0.75 -> if (s >= 10 || s == s.toLong().toDouble()) "${s.toLong()}с" else String.format("%.1fс", s)
            else -> "1/${(1.0 / s).let { if (it > 10) it.roundToLong() else (it * 10).roundToLong() / 10.0 }}"
                .replace(".0", "")
        }
    }
}

fun aeStateName(s: Int?): String? = when (s) {
    null -> null
    CaptureResult.CONTROL_AE_STATE_INACTIVE -> "inactive"
    CaptureResult.CONTROL_AE_STATE_SEARCHING -> "searching"
    CaptureResult.CONTROL_AE_STATE_CONVERGED -> "converged"
    CaptureResult.CONTROL_AE_STATE_LOCKED -> "locked"
    CaptureResult.CONTROL_AE_STATE_FLASH_REQUIRED -> "flash_required"
    CaptureResult.CONTROL_AE_STATE_PRECAPTURE -> "precapture"
    else -> "unknown"
}

fun afStateName(s: Int?): String? = when (s) {
    null -> null
    CaptureResult.CONTROL_AF_STATE_INACTIVE -> "inactive"
    CaptureResult.CONTROL_AF_STATE_PASSIVE_SCAN -> "passive_scan"
    CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED -> "passive_focused"
    CaptureResult.CONTROL_AF_STATE_ACTIVE_SCAN -> "active_scan"
    CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED -> "focused_locked"
    CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED -> "not_focused_locked"
    CaptureResult.CONTROL_AF_STATE_PASSIVE_UNFOCUSED -> "passive_unfocused"
    else -> "unknown"
}

fun aeModeName(m: Int?) = when (m) {
    null -> null
    CameraMetadata.CONTROL_AE_MODE_OFF -> "off"
    CameraMetadata.CONTROL_AE_MODE_ON -> "on"
    CameraMetadata.CONTROL_AE_MODE_ON_AUTO_FLASH -> "on_auto_flash"
    CameraMetadata.CONTROL_AE_MODE_ON_ALWAYS_FLASH -> "on_always_flash"
    else -> "other_$m"
}

fun afModeName(m: Int?) = when (m) {
    null -> null
    CameraMetadata.CONTROL_AF_MODE_OFF -> "off"
    CameraMetadata.CONTROL_AF_MODE_AUTO -> "auto"
    CameraMetadata.CONTROL_AF_MODE_MACRO -> "macro"
    CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE -> "continuous_picture"
    CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO -> "continuous_video"
    else -> "other_$m"
}

fun awbModeName(m: Int?) = when (m) {
    null -> null
    else -> WbPreset.entries.firstOrNull { it.awbMode == m }?.key ?: if (m == CameraMetadata.CONTROL_AWB_MODE_OFF) "off" else "other_$m"
}

fun nrModeName(m: Int?) = when (m) {
    null -> null
    CameraMetadata.NOISE_REDUCTION_MODE_OFF -> "off"
    CameraMetadata.NOISE_REDUCTION_MODE_FAST -> "fast"
    CameraMetadata.NOISE_REDUCTION_MODE_HIGH_QUALITY -> "high_quality"
    CameraMetadata.NOISE_REDUCTION_MODE_MINIMAL -> "minimal"
    CameraMetadata.NOISE_REDUCTION_MODE_ZERO_SHUTTER_LAG -> "zsl"
    else -> "other_$m"
}
