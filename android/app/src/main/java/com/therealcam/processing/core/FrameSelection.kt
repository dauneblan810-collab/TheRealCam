package com.therealcam.processing.core

import com.therealcam.data.StackMode
import com.therealcam.processing.RejectReason
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.roundToInt

/** Per-frame measurements from the downscaled analysis copy. */
internal class FrameStats(
    val meanLuma: Double,
    val sharpness: Double,
    val motionRms: Double?,
    /** False if the frame could not be decoded or has a different size. */
    val usable: Boolean,
)

internal class FrameSelection(
    /** Reject reason per frame position, null = accepted. */
    val reasons: Array<String?>,
    /** Position of the reference (sharpest accepted) frame; always accepted. */
    val reference: Int,
) {
    val accepted: List<Int> get() = reasons.indices.filter { reasons[it] == null }
}

internal object FrameSelector {
    /** |robust z| of the mean luma above which a frame is a brightness outlier (Iglewicz–Hoaglin). */
    const val BRIGHTNESS_Z = 3.5
    /** Scale floor for the robust z: max(1 luma level, 3% of the median luma). */
    const val BRIGHTNESS_FLOOR_ABS = 1.0
    const val BRIGHTNESS_FLOOR_REL = 0.03

    fun select(
        stats: List<FrameStats>,
        mode: StackMode,
        rejectOutliers: Boolean,
        maxMotionRadS: Double,
        luckyKeepFraction: Float,
    ): FrameSelection {
        val n = stats.size
        require(n > 0) { "No frames" }
        val reasons = arrayOfNulls<String>(n)
        for (i in 0 until n) if (!stats[i].usable) reasons[i] = RejectReason.ALIGN_FAILED

        if (rejectOutliers && mode != StackMode.LIGHTEN) {
            for (i in 0 until n) {
                val m = stats[i].motionRms ?: continue
                if (reasons[i] == null && m > maxMotionRadS) reasons[i] = RejectReason.MOTION
            }
        }

        val usable = (0 until n).filter { stats[it].usable }
        if (rejectOutliers && usable.size >= 3) {
            val lumas = DoubleArray(usable.size) { stats[usable[it]].meanLuma }
            val floor = max(BRIGHTNESS_FLOOR_ABS, BRIGHTNESS_FLOOR_REL * RobustStats.median(lumas))
            val z = RobustStats.robustZ(lumas, floor)
            usable.forEachIndexed { k, i ->
                if (reasons[i] == null && abs(z[k]) > BRIGHTNESS_Z) reasons[i] = RejectReason.BRIGHTNESS_OUTLIER
            }
        }

        if (reasons.none { it == null }) {
            // Never reject everything: keep the sharpest usable frame.
            reasons[sharpest(stats, usable.ifEmpty { (0 until n).toList() })] = null
        }

        val accepted = reasons.indices.filter { reasons[it] == null }
        val reference = sharpest(stats, accepted)

        when (mode) {
            StackMode.LUCKY -> {
                val fraction = if (luckyKeepFraction.isFinite()) luckyKeepFraction.coerceIn(0f, 1f) else 0.4f
                val keep = (fraction * accepted.size).roundToInt().coerceIn(1, accepted.size)
                byDescendingSharpness(stats, accepted).drop(keep).forEach { reasons[it] = RejectReason.NOT_SHARPEST }
            }
            StackMode.BEST_OF -> accepted.filter { it != reference }.forEach { reasons[it] = RejectReason.NOT_SHARPEST }
            else -> Unit
        }
        return FrameSelection(reasons, reference)
    }

    private fun byDescendingSharpness(stats: List<FrameStats>, positions: List<Int>): List<Int> =
        positions.sortedWith(compareByDescending<Int> { stats[it].sharpness }.thenBy { it })

    private fun sharpest(stats: List<FrameStats>, positions: List<Int>): Int = byDescendingSharpness(stats, positions).first()
}
