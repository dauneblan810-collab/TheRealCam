package com.therealcam.processing

import com.therealcam.processing.core.AlignConfig
import com.therealcam.processing.core.AlignVerdict
import com.therealcam.processing.core.FullResRefiner
import com.therealcam.processing.core.TranslationSearcher
import com.therealcam.processing.core.judgeAlignment
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random

class TranslationSearchTest {
    /** Content displacements (sx, sy); the applied shift that registers the frame is (-sx, -sy). */
    private val shifts = listOf(0 to 0, 7 to -3, -12 to 5, -1 to -9, 15 to 11, -16 to -16, 3 to 14, -9 to 0)

    @Test
    fun recoversKnownShiftsExactly_texture_multiLevelPyramid() {
        val w = 256
        val h = 192
        val scene = Synth.texture(seed = 3, count = 400, w = w, h = h)
        val searcher = TranslationSearcher(AlignConfig(minCoarseSide = 32))
        assertEquals(2, searcher.levelCount(w, h))
        val ref = searcher.prepare(Synth.plane(Synth.toArgb(Synth.render(w, h, scene, 120.0)), w, h))
        for ((sx, sy) in shifts) {
            val img = Synth.toArgb(Synth.render(w, h, scene, 120.0, sx, sy))
            val t = searcher.search(ref, searcher.prepare(Synth.plane(img, w, h)), maxShift = 24)
            assertEquals("dx for content shift ($sx,$sy): $t", -sx, t.dx)
            assertEquals("dy for content shift ($sx,$sy): $t", -sy, t.dy)
            assertTrue("weak match $t", t.z > 20)
        }
    }

    @Test
    fun recoversKnownShiftsExactly_noisyStarField_defaultConfig() {
        val w = 512
        val h = 384
        val stars = Synth.starField(seed = 11, count = 150, w = w, h = h, minAmp = 40.0, maxAmp = 200.0)
        val searcher = TranslationSearcher()
        assertEquals(2, searcher.levelCount(w, h))
        val rnd = Random(5)
        val ref = searcher.prepare(Synth.plane(Synth.toArgb(Synth.render(w, h, stars, 20.0, noise = 6.0, rnd = rnd)), w, h))
        for ((sx, sy) in shifts) {
            val img = Synth.toArgb(Synth.render(w, h, stars, 20.0, sx, sy, noise = 6.0, rnd = rnd))
            val t = searcher.search(ref, searcher.prepare(Synth.plane(img, w, h)))
            assertEquals("dx for content shift ($sx,$sy): $t", -sx, t.dx)
            assertEquals("dy for content shift ($sx,$sy): $t", -sy, t.dy)
        }
    }

    @Test
    fun fullResRefinerFindsExactShiftFromOffGuess() {
        val w = 320
        val h = 240
        val stars = Synth.starField(seed = 21, count = 120, w = w, h = h)
        val sx = -6
        val sy = 4
        val ref = Synth.plane(Synth.toArgb(Synth.render(w, h, stars, 25.0, noise = 4.0, rnd = Random(1))), w, h).highPass(5)
        val frame = Synth.plane(Synth.toArgb(Synth.render(w, h, stars, 25.0, sx, sy, noise = 4.0, rnd = Random(2))), w, h).highPass(5)
        // Guess is 3 px off in x and 2 px off in y; the true applied shift is (6, -4).
        val r = FullResRefiner.refine(ref, 0, 0, frame, 0, 0, cdx = 3, cdy = -2, radius = 4, margin = 6)
        assertNotNull(r)
        assertEquals(6, r!!.dx)
        assertEquals(-4, r.dy)
        assertTrue(!r.onBorder && r.z > 5)
    }

    @Test
    fun judgeAlignment_failsOnlyPoorMatchesInStructuredSeries() {
        val cfg = AlignConfig()
        val structured = judgeAlignment(
            ncc = doubleArrayOf(0.8, 0.75, 0.2, 0.82, 0.01),
            z = doubleArrayOf(150.0, 140.0, 40.0, 155.0, 2.0),
            cfg = cfg,
        )
        assertArrayEquals(
            arrayOf(AlignVerdict.OK, AlignVerdict.OK, AlignVerdict.FAILED, AlignVerdict.OK, AlignVerdict.FAILED),
            structured,
        )
        // Featureless series: nothing to align against, frames stay unshifted instead of failing.
        val flat = judgeAlignment(doubleArrayOf(0.01, 0.0, 0.02), doubleArrayOf(1.0, 0.0, 2.0), cfg)
        assertArrayEquals(Array(3) { AlignVerdict.NO_STRUCTURE }, flat)
    }
}
