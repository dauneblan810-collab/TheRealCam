package com.therealcam.processing

import com.therealcam.data.StackMode
import com.therealcam.processing.core.FrameSelector
import com.therealcam.processing.core.FrameStats
import com.therealcam.processing.core.RobustStats
import com.therealcam.processing.core.analysisSampleSize
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random

class FrameSelectionTest {

    private fun stats(
        means: List<Double>,
        sharp: List<Double> = means.indices.map { 10.0 + it },
        motion: List<Double?> = means.map { null },
        usable: List<Boolean> = means.map { true },
    ) = means.indices.map { FrameStats(means[it], sharp[it], motion[it], usable[it]) }

    private fun select(s: List<FrameStats>, mode: StackMode = StackMode.SIGMA_CLIP, reject: Boolean = true, lucky: Float = 0.4f) =
        FrameSelector.select(s, mode, reject, maxMotionRadS = 0.03, luckyKeepFraction = lucky)

    @Test
    fun headlightFrameIsBrightnessOutlier() {
        val sel = select(
            stats(
                means = listOf(20.1, 20.3, 19.9, 20.0, 45.0, 20.2, 20.1, 19.8),
                sharp = listOf(5.0, 6.0, 7.0, 8.0, 50.0, 4.0, 3.0, 2.0),
            ),
        )
        assertEquals(RejectReason.BRIGHTNESS_OUTLIER, sel.reasons[4])
        assertEquals(listOf(0, 1, 2, 3, 5, 6, 7), sel.accepted)
        assertEquals(3, sel.reference) // sharpest accepted, not the bright frame
    }

    @Test
    fun gradualBrighteningIsNotAnOutlier() {
        val sel = select(stats(means = (0 until 12).map { 20.0 + it }))
        assertEquals(12, sel.accepted.size)
    }

    @Test
    fun motionRejectedUnlessLightenOrDisabled() {
        val s = stats(means = List(5) { 30.0 }, motion = listOf(0.01, 0.2, null, 0.031, 0.0))
        val sel = select(s)
        assertEquals(RejectReason.MOTION, sel.reasons[1])
        assertEquals(RejectReason.MOTION, sel.reasons[3])
        assertNull(sel.reasons[2]) // unknown motion is not a reason to reject
        assertEquals(5, select(s, mode = StackMode.LIGHTEN).accepted.size)
        assertEquals(5, select(s, reject = false).accepted.size)
    }

    @Test
    fun neverRejectsAllFrames() {
        val allShaky = select(
            stats(means = List(6) { 30.0 }, sharp = listOf(1.0, 9.0, 3.0, 4.0, 2.0, 5.0), motion = List(6) { 0.5 }),
        )
        assertEquals(listOf(1), allShaky.accepted)
        assertEquals(1, allShaky.reference)

        // Every frame rejected for some reason (unusable, motion, brightness): the sharpest usable one survives.
        val mixed = select(
            stats(
                means = listOf(30.0, 30.2, 29.9, 30.1, 90.0),
                sharp = listOf(50.0, 4.0, 7.0, 6.0, 99.0),
                motion = listOf(null, 0.5, 0.5, 0.5, null),
                usable = listOf(false, true, true, true, true),
            ),
        )
        assertEquals(listOf(4), mixed.accepted)
        assertEquals(RejectReason.ALIGN_FAILED, mixed.reasons[0])

        for (mode in StackMode.entries) {
            assertTrue(select(stats(List(3) { 10.0 }, motion = List(3) { 1.0 }), mode = mode).accepted.isNotEmpty())
        }
    }

    @Test
    fun luckyKeepsSharpestFraction_bestOfKeepsOne() {
        val sharp = listOf(3.0, 9.0, 1.0, 7.0, 5.0, 10.0, 2.0, 8.0, 4.0, 6.0)
        val s = stats(means = List(10) { 50.0 }, sharp = sharp)
        val lucky = select(s, mode = StackMode.LUCKY, lucky = 0.3f)
        assertEquals(listOf(1, 5, 7), lucky.accepted)
        assertEquals(5, lucky.reference)
        assertTrue((0 until 10).filter { it !in lucky.accepted }.all { lucky.reasons[it] == RejectReason.NOT_SHARPEST })

        assertEquals(listOf(5), select(s, mode = StackMode.LUCKY, lucky = 0.01f).accepted)

        val best = select(s, mode = StackMode.BEST_OF)
        assertEquals(listOf(5), best.accepted)
        assertEquals(RejectReason.NOT_SHARPEST, best.reasons[1])
    }

    @Test
    fun kthSmallestMatchesSorting() {
        val rnd = Random(3)
        repeat(500) {
            val n = 1 + rnd.nextInt(70)
            val a = IntArray(n) { rnd.nextInt(256) }
            val sorted = a.sortedArray()
            val k = rnd.nextInt(n)
            assertEquals(sorted[k], RobustStats.kthSmallest(a.copyOf(), n, k))
        }
    }

    @Test
    fun analysisSampleSizeKeepsLongSideNear1000() {
        assertEquals(4, analysisSampleSize(4000, 3000))
        assertEquals(4, analysisSampleSize(4032, 3024))
        assertEquals(8, analysisSampleSize(8160, 6120))
        assertEquals(2, analysisSampleSize(1920, 1080))
        assertEquals(1, analysisSampleSize(640, 480))
    }
}
