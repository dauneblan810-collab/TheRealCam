package com.therealcam.processing

import com.therealcam.data.StackMode
import com.therealcam.processing.Synth.gray
import com.therealcam.processing.core.Layer
import com.therealcam.processing.core.StackEngine
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sqrt

class SigmaClipTest {
    private val w = 192
    private val h = 144
    private val n = 12

    @Test
    fun removesSatelliteStreak_keepsStars_noiseDropsLikeSqrtN() = runBlocking {
        val stars = Synth.starField(seed = 1, count = 60, w = w, h = h, minAmp = 120.0, maxAmp = 200.0)
        val base = Synth.toArgb(Synth.render(w, h, stars, background = 30.0))
        val streak = Synth.lineMask(w, h, 0.0, 20.0, w - 1.0, 120.0, halfWidth = 1.0)
        val rnd = Random(7)
        val frames = List(n) { k ->
            val c = Synth.render(w, h, stars, background = 30.0, noise = 8.0, rnd = rnd)
            if (k == 5) Synth.add(c, streak, 200.0)
            Synth.toArgb(c)
        }
        val out = IntArray(w * h)
        StackEngine(stripRowsOverride = 32).merge(
            w, h, StackMode.SIGMA_CLIP, List(n) { Layer(it, 0, 0) }, MemoryFrameSource(frames, w, h), 2.5f,
            { y0, rows, px -> System.arraycopy(px, 0, out, y0 * w, rows * w) },
        )

        // The streak is an outlier in one frame only: it must be gone.
        var residue = 0
        for (i in streak) {
            assertTrue(gray(frames[5][i]) >= minOf(255, gray(base[i]) + 120))
            residue = max(residue, abs(gray(out[i]) - gray(base[i])))
        }
        assertTrue("streak residue $residue", residue <= 12)

        // Stars are in every frame: they keep their brightness.
        var baseSum = 0.0
        var outSum = 0.0
        val streakSet = streak.toHashSet()
        for (s in stars) {
            val x = s.x.roundToInt()
            val y = s.y.roundToInt()
            if (x !in 2 until w - 2 || y !in 2 until h - 2) continue
            val i = y * w + x
            if (i in streakSet) continue
            assertTrue("star at ($x,$y): ${gray(out[i])} vs ${gray(base[i])}", abs(gray(out[i]) - gray(base[i])) <= 12)
            baseSum += gray(base[i])
            outSum += gray(out[i])
        }
        assertTrue("stars brightness ratio ${outSum / baseSum}", abs(outSum / baseSum - 1.0) < 0.03)

        // Background noise falls roughly like 1/sqrt(N).
        val nearStreak = Synth.lineMask(w, h, 0.0, 20.0, w - 1.0, 120.0, halfWidth = 4.0).toHashSet()
        val background = (0 until w * h).filter { gray(base[it]) <= 30 && it !in nearStreak }.toIntArray()
        assertTrue(background.size > w * h / 2)
        val single = Synth.std(frames[0], base, background)
        val stacked = Synth.std(out, base, background)
        val ratio = single / stacked
        assertTrue("noise ratio $ratio (single $single, stacked $stacked)", ratio in 0.8 * sqrt(n.toDouble())..1.2 * sqrt(n.toDouble()))
    }

    @Test
    fun meanKeepsStreakAtOneNth_sigmaClipDoesNot() = runBlocking {
        // Contrast check: plain averaging cannot remove the outlier, it only dilutes it.
        val streak = Synth.lineMask(w, h, 10.0, 0.0, 10.0, h - 1.0, halfWidth = 0.6)
        val frames = List(6) { k ->
            val c = DoubleArray(w * h) { 40.0 + (k % 3) }
            if (k == 2) Synth.add(c, streak, 180.0)
            Synth.toArgb(c)
        }
        val layers = List(6) { Layer(it, 0, 0) }
        val source = MemoryFrameSource(frames, w, h)
        val mean = IntArray(w * h)
        val clipped = IntArray(w * h)
        val engine = StackEngine(stripRowsOverride = 40)
        engine.merge(w, h, StackMode.MEAN, layers, source, 2.5f, { y0, rows, px -> System.arraycopy(px, 0, mean, y0 * w, rows * w) })
        engine.merge(w, h, StackMode.SIGMA_CLIP, layers, source, 2.5f, { y0, rows, px -> System.arraycopy(px, 0, clipped, y0 * w, rows * w) })
        for (i in streak) {
            assertTrue(gray(mean[i]) >= 40 + 180 / 6 - 2)
            assertTrue(gray(clipped[i]) <= 42)
        }
    }
}
