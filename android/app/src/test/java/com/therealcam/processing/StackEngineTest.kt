package com.therealcam.processing

import com.therealcam.data.StackMode
import com.therealcam.processing.Synth.gray
import com.therealcam.processing.core.Layer
import com.therealcam.processing.core.PlannedOutput
import com.therealcam.processing.core.StackEngine
import com.therealcam.processing.core.StackPlan
import com.therealcam.processing.core.packRgb
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

class StackEngineTest {

    private suspend fun planOf(
        frames: List<IntArray?>, w: Int, h: Int, options: StackOptions,
        motions: List<Double?> = frames.map { 0.005 },
    ): StackPlan = StackEngine().plan(frames.indices.map { Synth.frame(it, motions[it]) }, options, MemoryFrameSource(frames, w, h))

    private suspend fun mergeOf(
        layers: List<Layer>, frames: List<IntArray?>, w: Int, h: Int, mode: StackMode,
        engine: StackEngine = StackEngine(), onProgress: (Float) -> Unit = {},
    ): IntArray {
        val out = IntArray(w * h)
        engine.merge(
            w, h, mode, layers, MemoryFrameSource(frames, w, h), 2.5f,
            { y0, rows, px -> System.arraycopy(px, 0, out, y0 * w, rows * w) }, onProgress,
        )
        return out
    }

    private fun layersOf(plan: StackPlan): List<Layer> = (plan.output as PlannedOutput.Merge).layers

    @Test
    fun alignmentRecoversKnownShiftsExactly_includingNegativeAndOdd() = runBlocking {
        val w = 512
        val h = 384
        val stars = Synth.starField(seed = 31, count = 200, w = w, h = h, minAmp = 40.0, maxAmp = 200.0)
        val shifts = listOf(0 to 0, 5 to -3, -7 to 2, 13 to 9, -11 to -6, 0 to 1, 3 to -12)
        val rnd = Random(9)
        val frames = shifts.map { (sx, sy) -> Synth.toArgb(Synth.render(w, h, stars, 20.0, sx, sy, noise = 5.0, rnd = rnd)) }
        val plan = planOf(frames, w, h, StackOptions(StackMode.SIGMA_CLIP, align = true, rejectOutlierFrames = true))
        val (rx, ry) = shifts[plan.reference]
        for ((i, r) in plan.reports.withIndex()) {
            assertTrue("frame $i rejected: ${r.rejectReason}", r.accepted)
            assertEquals("dx of frame $i", rx - shifts[i].first, r.dx)
            assertEquals("dy of frame $i", ry - shifts[i].second, r.dy)
        }
        assertTrue(plan.aligned)
        assertEquals(shifts.size, plan.usedFrames)
    }

    @Test
    fun headlightFrameIsRejectedAsBrightnessOutlier() = runBlocking {
        val w = 256
        val h = 192
        val stars = Synth.starField(seed = 2, count = 80, w = w, h = h)
        val rnd = Random(3)
        val frames = List(8) { k ->
            val c = Synth.render(w, h, stars, 25.0, noise = 5.0, rnd = rnd)
            if (k == 3) {
                // Headlights: the whole frame floods, brightest towards one corner.
                for (y in 0 until h) for (x in 0 until w) c[y * w + x] += 80.0 + 100.0 * (x + y) / (w + h)
            }
            Synth.toArgb(c)
        }
        val plan = planOf(frames, w, h, StackOptions(StackMode.SIGMA_CLIP, align = true, rejectOutlierFrames = true))
        assertEquals(RejectReason.BRIGHTNESS_OUTLIER, plan.reports[3].rejectReason)
        assertTrue(plan.reports.filterIndexed { i, _ -> i != 3 }.all { it.accepted })
        assertEquals(7, plan.usedFrames)

        val base = Synth.toArgb(Synth.render(w, h, stars, 25.0))
        val out = mergeOf(layersOf(plan), frames, w, h, StackMode.SIGMA_CLIP)
        val meanDiff = (0 until w * h).sumOf { gray(out[it]) - gray(base[it]) }.toDouble() / (w * h)
        assertTrue("headlight light leaked into the stack: $meanDiff", abs(meanDiff) < 1.0)
    }

    @Test
    fun lightenKeepsMovingDotTrail() = runBlocking {
        val w = 160
        val h = 96
        val rnd = Random(4)
        val frames = List(6) { k ->
            Synth.toArgb(Synth.render(w, h, listOf(Blob(20.0 + 20 * k, 48.0, 220.0, 1.3)), 12.0, noise = 1.5, rnd = rnd))
        }
        val plan = planOf(frames, w, h, StackOptions(StackMode.LIGHTEN, align = false, rejectOutlierFrames = false))
        val layers = layersOf(plan)
        assertEquals(6, layers.size)
        assertTrue(layers.all { it.dx == 0 && it.dy == 0 })

        val trail = mergeOf(layers, frames, w, h, StackMode.LIGHTEN)
        val mean = mergeOf(layers, frames, w, h, StackMode.MEAN)
        for (k in 0 until 6) {
            val i = 48 * w + 20 + 20 * k
            assertTrue("trail point $k = ${gray(trail[i])}", gray(trail[i]) >= 200)
            assertTrue("mean dilutes the dot: ${gray(mean[i])}", gray(mean[i]) <= 12 + 220 / 6 + 6)
        }
        val farFromDots = (0 until w * h).filter { abs(it / w - 48) > 8 }.map { gray(trail[it]) }.sorted()
        assertTrue("background stays dark", farFromDots[farFromDots.size / 2] <= 20)
    }

    @Test
    fun luckyAndBestOfPickSharpestFrames() = runBlocking {
        val w = 256
        val h = 192
        val sharp = Synth.render(w, h, Synth.texture(seed = 41, count = 300, w = w, h = h), 120.0)
        val radii = listOf(4, 0, 7, 2, 9, 1, 5, 3, 8, 6)
        val rnd = Random(5)
        val frames = radii.map { r ->
            val c = Synth.boxBlur(sharp, w, h, r)
            for (i in c.indices) c[i] += rnd.nextGaussian()
            Synth.toArgb(c)
        }
        val sharpest = radii.indexOf(0)

        val best = planOf(frames, w, h, StackOptions(StackMode.BEST_OF, align = false, rejectOutlierFrames = false))
        assertEquals(PlannedOutput.Copy(sharpest), best.output)
        assertEquals(listOf(sharpest), best.reports.indices.filter { best.reports[it].accepted })
        assertTrue(best.reports.filter { !it.accepted }.all { it.rejectReason == RejectReason.NOT_SHARPEST })
        val sharpestThree = best.reports.indices.sortedByDescending { best.reports[it].sharpness }.take(3)
        assertEquals(listOf(0, 1, 2), sharpestThree.map { radii[it] })

        val lucky = planOf(
            frames, w, h,
            StackOptions(StackMode.LUCKY, align = true, rejectOutlierFrames = true, luckyKeepFraction = 0.3f),
        )
        val kept = lucky.reports.indices.filter { lucky.reports[it].accepted }
        assertEquals(listOf(0, 1, 2).map { radii.indexOf(it) }.sorted(), kept)
        assertEquals(sharpest, lucky.reference)
        val layers = layersOf(lucky)
        assertEquals(sharpest, layers.first().position)
        assertTrue(layers.all { it.dx == 0 && it.dy == 0 })
        assertTrue(
            lucky.reports.indices.filter { it !in kept }.all { lucky.reports[it].rejectReason == RejectReason.NOT_SHARPEST },
        )
    }

    @Test
    fun neverRejectsAllFrames() = runBlocking {
        val w = 128
        val h = 96
        val scene = Synth.texture(seed = 6, count = 120, w = w, h = h)
        val rnd = Random(6)
        val frames = List(5) { Synth.toArgb(Synth.render(w, h, scene, 100.0, noise = 3.0, rnd = rnd)) }
        val plan = planOf(
            frames, w, h, StackOptions(StackMode.SIGMA_CLIP, align = true, rejectOutlierFrames = true),
            motions = List(5) { 0.3 },
        )
        val accepted = plan.reports.indices.filter { plan.reports[it].accepted }
        assertEquals(1, accepted.size)
        assertEquals(PlannedOutput.Copy(accepted.single()), plan.output)
        assertEquals(1, plan.usedFrames)
        assertTrue(plan.reports.filter { !it.accepted }.all { it.rejectReason == RejectReason.MOTION })
    }

    @Test
    fun edgePixelsAverageOnlyTheFramesThatCoverThem() = runBlocking {
        val w = 40
        val h = 50 // not a multiple of the 16-row strips
        val shifts = listOf(0 to 0, 3 to 0, 0 to -2, -4 to 5)
        val layers = shifts.mapIndexed { k, (dx, dy) -> Layer(k, dx, dy) }
        val colored = List(4) { k ->
            IntArray(w * h) { i ->
                val x = i % w
                val y = i / w
                packRgb((x * 7 + k * 31) % 256, (y * 11 + k * 17) % 256, (x * 3 + y * 5 + k * 53) % 256)
            }
        }
        // Values within 2 levels of each other: sigma clipping keeps all, so it must equal the plain mean.
        val nearlyEqual = List(4) { k ->
            IntArray(w * h) { i ->
                val v = 100 + (k + i % w + i / w) % 3
                packRgb(v, v, v)
            }
        }
        val engine = StackEngine(workers = 2, stripRowsOverride = 16)
        for (mode in listOf(StackMode.MEAN, StackMode.LIGHTEN, StackMode.SIGMA_CLIP)) {
            val frames = if (mode == StackMode.SIGMA_CLIP) nearlyEqual else colored
            val out = mergeOf(layers, frames, w, h, mode, engine)
            for (y in 0 until h) for (x in 0 until w) {
                val covering = shifts.indices.filter { k -> x - shifts[k].first in 0 until w && y - shifts[k].second in 0 until h }
                val values = covering.map { k -> frames[k][(y - shifts[k].second) * w + (x - shifts[k].first)] }
                val expected = if (mode == StackMode.LIGHTEN) {
                    packRgb(values.maxOf { (it ushr 16) and 0xFF }, values.maxOf { (it ushr 8) and 0xFF }, values.maxOf { it and 0xFF })
                } else {
                    val n = values.size
                    packRgb(
                        (values.sumOf { (it ushr 16) and 0xFF } + n / 2) / n,
                        (values.sumOf { (it ushr 8) and 0xFF } + n / 2) / n,
                        (values.sumOf { it and 0xFF } + n / 2) / n,
                    )
                }
                assertEquals("$mode at ($x,$y) from frames $covering", expected, out[y * w + x])
            }
        }
        // Spot checks of the coverage itself.
        fun coverageAt(x: Int, y: Int) = shifts.indices.filter { k -> x - shifts[k].first in 0 until w && y - shifts[k].second in 0 until h }
        assertEquals(listOf(0, 2, 3), coverageAt(0, 25))
        assertEquals(listOf(0, 1), coverageAt(w - 1, h - 1))
        assertEquals(listOf(0, 1, 2), coverageAt(w - 1, 2))
    }

    @Test
    fun undecodableFrameIsRejected_noneAndSingleFrameAreCopies() = runBlocking {
        val w = 96
        val h = 64
        val img = Synth.toArgb(Synth.render(w, h, Synth.texture(seed = 8, count = 60, w = w, h = h), 90.0))
        val frames = listOf(img, img, null, img)
        val plan = planOf(frames, w, h, StackOptions(StackMode.MEAN, align = false, rejectOutlierFrames = false))
        assertEquals(RejectReason.ALIGN_FAILED, plan.reports[2].rejectReason)
        assertEquals(3, plan.usedFrames)

        val none = planOf(frames, w, h, StackOptions(StackMode.NONE, align = true, rejectOutlierFrames = true))
        assertEquals(PlannedOutput.Copy(0), none.output)
        assertTrue(none.reports.all { it.accepted && it.rejectReason == null })

        val single = planOf(listOf(img), w, h, StackOptions(StackMode.SIGMA_CLIP, align = true, rejectOutlierFrames = true))
        assertEquals(PlannedOutput.Copy(0), single.output)
        assertEquals(1, single.usedFrames)
        assertTrue(single.reports.single().meanLuma > 0.0)
    }

    @Test
    fun integration_satelliteHeadlightsShakeAndShifts() = runBlocking {
        val w = 384
        val h = 288
        val stars = Synth.starField(seed = 51, count = 160, w = w, h = h, minAmp = 50.0, maxAmp = 200.0)
        val shifts = listOf(0 to 0, 4 to -2, -3 to 5, 7 to 1, -6 to -4, 2 to 8, -1 to -7, 9 to -5, -8 to 3, 5 to 6)
        val satellite = 2
        val headlights = 6
        val shaky = 8
        val streak = Synth.lineMask(w, h, 0.0, 30.0, w - 1.0, 220.0, halfWidth = 1.0)
        val rnd = Random(12)
        val frames = shifts.mapIndexed { k, (sx, sy) ->
            val c = Synth.render(w, h, stars, 25.0, sx, sy, noise = 5.0, rnd = rnd)
            if (k == satellite) Synth.add(c, streak, 180.0)
            if (k == headlights) for (i in c.indices) c[i] += 80.0
            Synth.toArgb(c)
        }
        val motions = shifts.indices.map { if (it == shaky) 0.08 else 0.005 }
        val progress = ArrayList<Float>()
        val engine = StackEngine()
        val plan = engine.plan(
            frames.indices.map { Synth.frame(it, motions[it]) },
            StackOptions(StackMode.SIGMA_CLIP, align = true, rejectOutlierFrames = true),
            MemoryFrameSource(frames, w, h),
        ) { progress += it }
        assertTrue(progress.zipWithNext().all { (a, b) -> b > a } && progress.last() == 1f)

        assertEquals(RejectReason.BRIGHTNESS_OUTLIER, plan.reports[headlights].rejectReason)
        assertEquals(RejectReason.MOTION, plan.reports[shaky].rejectReason)
        val (rx, ry) = shifts[plan.reference]
        for (i in shifts.indices) {
            if (i == headlights || i == shaky) continue
            val r = plan.reports[i]
            assertNull("frame $i", r.rejectReason)
            assertEquals("dx of frame $i", rx - shifts[i].first, r.dx)
            assertEquals("dy of frame $i", ry - shifts[i].second, r.dy)
        }
        val layers = layersOf(plan)
        assertEquals(8, layers.size)

        val out = mergeOf(layers, frames, w, h, StackMode.SIGMA_CLIP)
        val base = Synth.toArgb(Synth.render(w, h, stars, 25.0, rx, ry))
        val x0 = layers.maxOf { max(0, it.dx) }
        val x1 = w + layers.minOf { min(0, it.dx) }
        val y0 = layers.maxOf { max(0, it.dy) }
        val y1 = h + layers.minOf { min(0, it.dy) }
        fun inside(x: Int, y: Int) = x in x0 until x1 && y in y0 until y1

        // Streak (drawn in the satellite frame's coordinates) is clipped away.
        val sat = layers.first { it.position == satellite }
        var residue = 0
        for (i in streak) {
            val x = i % w + sat.dx
            val y = i / w + sat.dy
            if (!inside(x, y)) continue
            residue = max(residue, abs(gray(out[y * w + x]) - gray(base[y * w + x])))
        }
        assertTrue("streak residue $residue", residue <= 12)

        // Everywhere covered by all frames the stack matches the true scene with ~1/sqrt(8) of the noise.
        var s2 = 0.0
        var count = 0
        for (y in y0 until y1) for (x in x0 until x1) {
            val d = (gray(out[y * w + x]) - gray(base[y * w + x])).toDouble()
            s2 += d * d
            count++
        }
        val rms = sqrt(s2 / count)
        assertTrue("rms error $rms", rms < 2.6)

        var baseSum = 0.0
        var outSum = 0.0
        for (s in stars) {
            val x = (s.x + rx).roundToInt()
            val y = (s.y + ry).roundToInt()
            if (!inside(x, y)) continue
            baseSum += gray(base[y * w + x])
            outSum += gray(out[y * w + x])
        }
        assertTrue("stars brightness ratio ${outSum / baseSum}", abs(outSum / baseSum - 1.0) < 0.03)
    }

    @Test
    fun mergeProgressIsMonotonicAndComplete() = runBlocking {
        val w = 64
        val h = 70
        val frames = List(3) { k -> IntArray(w * h) { packRgb(10 * k, 20, 30) } }
        val progress = ArrayList<Float>()
        mergeOf(List(3) { Layer(it, 0, 0) }, frames, w, h, StackMode.MEAN, StackEngine(stripRowsOverride = 16)) { progress += it }
        assertEquals(5, progress.size)
        assertTrue(progress.zipWithNext().all { (a, b) -> b > a })
        assertEquals(1f, progress.last())
    }
}
