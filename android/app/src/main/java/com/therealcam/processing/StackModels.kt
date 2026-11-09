package com.therealcam.processing

import com.therealcam.data.StackMode
import java.io.File

/** One captured frame of a series. All frames of a series share pixel size and EXIF orientation. */
data class StackFrame(
    val file: File,
    val index: Int,
    val exposureNs: Long?,
    val iso: Int?,
    /** Gyro RMS |ω| during this frame's exposure, rad/s; null (or NaN) = unknown, never rejected. */
    val motionRms: Double?,
)

data class StackOptions(
    val mode: StackMode,
    val align: Boolean,
    val rejectOutlierFrames: Boolean,
    /** Frames above this are rejected as "motion" (only with [rejectOutlierFrames], never for LIGHTEN). */
    val maxMotionRadS: Double = 0.03,
    /** LUCKY keeps this fraction of the sharpest accepted frames (at least 1). */
    val luckyKeepFraction: Float = 0.4f,
    val sigma: Float = 2.5f,
    val jpegQuality: Int = 95,
)

data class FrameReport(
    val index: Int,
    val accepted: Boolean,
    /** null | "motion" | "brightness_outlier" | "not_sharpest" | "align_failed" */
    val rejectReason: String?,
    /** Applied shift in full-resolution pixels: frame pixel (x, y) lands on reference pixel (x + dx, y + dy). */
    val dx: Int,
    val dy: Int,
    /** Variance of the Laplacian of the downscaled luminance (relative, compare within a series). */
    val sharpness: Double,
    /** Mean luminance 0..255 of the downscaled frame. */
    val meanLuma: Double,
)

data class StackResult(
    val output: File,
    val width: Int,
    val height: Int,
    val frames: List<FrameReport>,
    val usedFrames: Int,
)

object RejectReason {
    const val MOTION = "motion"
    const val BRIGHTNESS_OUTLIER = "brightness_outlier"
    const val NOT_SHARPEST = "not_sharpest"
    /** Also used for frames that cannot be decoded or have a different pixel size. */
    const val ALIGN_FAILED = "align_failed"
}
