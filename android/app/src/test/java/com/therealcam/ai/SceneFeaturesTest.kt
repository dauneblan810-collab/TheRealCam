package com.therealcam.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SceneFeaturesTest {
    private val ctx = SceneFeatures.Context(
        previewExposureNs = 33_000_000L, previewIso = 800, aperture = 1.8f, lux = 2.0,
        pitchDeg = 40.0, rollDeg = 0.0, gyroRms = 0.003, timeMs = 1_762_698_153_512L, tzOffsetMs = 3 * 3600_000,
        lat = 55.75, lon = 37.61, focusDiopters = 0f, zoom = 1f, front = false,
    )

    private fun gray(v: Int, w: Int = 192, h: Int = 256) = IntArray(w * h) { (0xFF shl 24) or (v shl 16) or (v shl 8) or v }

    @Test
    fun producesAllNamedFeatures() {
        val f = SceneFeatures.compute(gray(40), 192, 256, ctx)
        assertEquals(SceneFeatures.NAMES, f.keys.toList())
        assertEquals(40 / 255.0, f["luma_mean"]!!, 1e-3)
        assertEquals(0.0, f["luma_std"]!!, 1e-6)
        assertTrue(f["sun_alt_deg"]!! in -90.0..90.0)
        assertEquals(1.0, f["r_g"]!!, 1e-3)
    }

    @Test
    fun missingContextGivesNulls() {
        val f = SceneFeatures.compute(gray(10), 192, 256, ctx.copy(previewIso = null, lat = null, lon = null, lux = null, gyroRms = Double.NaN))
        assertNull(f["scene_ev"])
        assertNull(f["sun_alt_deg"])
        assertNull(f["lux_log10"])
        assertNull(f["motion_log10"])
    }

    @Test
    fun starsAreCountedAsPointSources() {
        val px = gray(5)
        val rnd = kotlin.random.Random(1)
        repeat(60) {
            val x = 2 + rnd.nextInt(188)
            val y = 2 + rnd.nextInt(252)
            px[y * 192 + x] = 0xFFFFFFFF.toInt()
        }
        val f = SceneFeatures.compute(px, 192, 256, ctx)
        assertTrue("point_sources=${f["point_sources"]}", f["point_sources"]!! > 5.0)
        assertTrue(SceneFeatures.compute(gray(5), 192, 256, ctx)["point_sources"]!! == 0.0)
    }

    @Test
    fun brighterExposureMeansLowerSceneEv() {
        val dark = SceneFeatures.compute(gray(60), 192, 256, ctx)["scene_ev"]!!
        val longer = SceneFeatures.compute(gray(60), 192, 256, ctx.copy(previewExposureNs = 66_000_000L))["scene_ev"]!!
        assertEquals(1.0, dark - longer, 1e-6)
    }

    @Test
    fun thumbAndRotation() {
        val t = SceneFeatures.thumb64(gray(100), 192, 256)
        assertEquals(64 * 64, t.size)
        assertEquals(100, t[0] and 0xFF)
        val px = IntArray(6) { it }
        val (r, w, h) = SceneFeatures.rotateCw(px, 3, 2, 90)
        assertEquals(2, w)
        assertEquals(3, h)
        // [0 1 2 / 3 4 5] rotated clockwise -> [3 0 / 4 1 / 5 2]
        assertEquals(listOf(3, 0, 4, 1, 5, 2), r.toList())
        val (r2, _, _) = SceneFeatures.rotateCw(px, 3, 2, 270)
        assertEquals(listOf(2, 5, 1, 4, 0, 3), r2.toList())
    }
}
