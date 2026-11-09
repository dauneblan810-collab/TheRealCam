package com.therealcam.ai

import com.therealcam.sensors.Astro
import java.time.Instant
import java.time.ZoneOffset
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Scene features, schema 1 (docs/MODEL_FORMAT.md §4). This is the ONLY implementation: values are
 * stored in meta.json and the trainer reads them from there. Pure Kotlin, unit-testable on the JVM.
 */
object SceneFeatures {
    const val SCHEMA = 1
    const val PREVIEW_LONG = 256
    const val PREVIEW_SHORT = 192
    const val THUMB = 64

    val NAMES: List<String> = listOf(
        "scene_ev", "preview_exp_log2", "preview_iso_log2",
        "luma_mean", "luma_std", "luma_p05", "luma_p50", "luma_p95",
        "clip_hi", "clip_lo", "bright_frac", "sharpness", "point_sources",
        "r_g", "b_g", "saturation", "top_luma", "bottom_luma", "center_luma",
        "lux_log10", "pitch_deg", "roll_deg", "motion_log10",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        "sun_alt_deg", "moon_alt_deg", "moon_illum",
        "focus_diopters", "zoom", "front",
    )

    data class Context(
        val previewExposureNs: Long?,
        val previewIso: Int?,
        val aperture: Float?,
        val lux: Double?,
        val pitchDeg: Double?,
        val rollDeg: Double?,
        val gyroRms: Double?,
        val timeMs: Long,
        val tzOffsetMs: Int,
        val lat: Double?,
        val lon: Double?,
        val focusDiopters: Float?,
        val zoom: Float?,
        val front: Boolean,
    )

    private fun ok(v: Double?): Double? = v?.takeIf { it.isFinite() }

    /** All features; missing ones are null. [px] is the upright preview (ARGB), [w]×[h]. */
    fun compute(px: IntArray, w: Int, h: Int, c: Context): Map<String, Double?> {
        val out = LinkedHashMap<String, Double?>()
        val s = imageStats(px, w, h)
        val t = c.previewExposureNs?.takeIf { it > 0 }?.let { it / 1e9 }
        val iso = c.previewIso?.takeIf { it > 0 }?.toDouble()
        val n = (c.aperture?.takeIf { it > 0f } ?: 1.8f).toDouble()
        out["scene_ev"] = if (t != null && iso != null) {
            log2(n * n / t) - log2(iso / 100.0) + log2(max(s.meanLin, 1e-4) / 0.18)
        } else null
        out["preview_exp_log2"] = t?.let { log2(it) }
        out["preview_iso_log2"] = iso?.let { log2(it) }
        out["luma_mean"] = s.mean
        out["luma_std"] = s.std
        out["luma_p05"] = s.p05
        out["luma_p50"] = s.p50
        out["luma_p95"] = s.p95
        out["clip_hi"] = s.clipHi
        out["clip_lo"] = s.clipLo
        out["bright_frac"] = s.brightFrac
        out["sharpness"] = s.sharpness
        out["point_sources"] = s.pointSources
        out["r_g"] = s.rg
        out["b_g"] = s.bg
        out["saturation"] = s.saturation
        out["top_luma"] = s.top
        out["bottom_luma"] = s.bottom
        out["center_luma"] = s.center
        out["lux_log10"] = ok(c.lux)?.let { log10(1.0 + max(0.0, it)) }
        out["pitch_deg"] = ok(c.pitchDeg)
        out["roll_deg"] = ok(c.rollDeg)
        out["motion_log10"] = ok(c.gyroRms)?.let { log10(1e-4 + it) }
        val local = Instant.ofEpochMilli(c.timeMs).atOffset(ZoneOffset.ofTotalSeconds(c.tzOffsetMs / 1000))
        val hour = local.hour + local.minute / 60.0 + local.second / 3600.0
        out["hour_sin"] = sin(2 * PI * hour / 24.0)
        out["hour_cos"] = cos(2 * PI * hour / 24.0)
        val doy = local.dayOfYear.toDouble()
        out["doy_sin"] = sin(2 * PI * (doy - 1) / 365.25)
        out["doy_cos"] = cos(2 * PI * (doy - 1) / 365.25)
        val lat = ok(c.lat)
        val lon = ok(c.lon)
        if (lat != null && lon != null) {
            out["sun_alt_deg"] = Astro.sun(c.timeMs, lat, lon).altDeg
            out["moon_alt_deg"] = Astro.moon(c.timeMs, lat, lon).altDeg
        } else {
            out["sun_alt_deg"] = null
            out["moon_alt_deg"] = null
        }
        out["moon_illum"] = Astro.moonIllumination(c.timeMs)
        out["focus_diopters"] = c.focusDiopters?.toDouble()?.let { ok(it) }
        out["zoom"] = (c.zoom ?: 1f).toDouble()
        out["front"] = if (c.front) 1.0 else 0.0
        check(out.keys.toList() == NAMES)
        return out
    }

    class ImageStats(
        val mean: Double, val std: Double, val meanLin: Double,
        val p05: Double, val p50: Double, val p95: Double,
        val clipHi: Double, val clipLo: Double, val brightFrac: Double,
        val sharpness: Double, val pointSources: Double,
        val rg: Double, val bg: Double, val saturation: Double,
        val top: Double, val bottom: Double, val center: Double,
    )

    fun luma(p: Int): Float {
        val r = (p shr 16 and 0xFF) / 255f
        val g = (p shr 8 and 0xFF) / 255f
        val b = (p and 0xFF) / 255f
        return 0.2126f * r + 0.7152f * g + 0.0722f * b
    }

    fun imageStats(px: IntArray, w: Int, h: Int): ImageStats {
        val n = w * h
        require(px.size >= n && n > 0)
        val y = FloatArray(n)
        val hist = IntArray(256)
        var sum = 0.0
        var sum2 = 0.0
        var sumLin = 0.0
        var sr = 0.0
        var sg = 0.0
        var sb = 0.0
        var sat = 0.0
        var hi = 0
        var lo = 0
        var bright = 0
        for (i in 0 until n) {
            val p = px[i]
            val r = (p shr 16 and 0xFF)
            val g = (p shr 8 and 0xFF)
            val b = (p and 0xFF)
            val l = luma(p)
            y[i] = l
            sum += l
            sum2 += l.toDouble() * l
            sumLin += l.toDouble().pow(2.2)
            sr += r; sg += g; sb += b
            val mx = max(r, max(g, b)) / 255.0
            val mn = minOf(r, g, b) / 255.0
            sat += (mx - mn) / (mx + 1e-4)
            hist[(l * 255f).toInt().coerceIn(0, 255)]++
            if (l >= 250f / 255f) hi++
            if (l <= 5f / 255f) lo++
            if (l >= 0.7f) bright++
        }
        val mean = sum / n
        val std = sqrt(max(0.0, sum2 / n - mean * mean))
        fun pct(p: Double): Double {
            val target = p * n
            var acc = 0
            for (b in 0 until 256) {
                acc += hist[b]
                if (acc >= target) return b / 255.0
            }
            return 1.0
        }
        val p50 = pct(0.50)

        var lapSum = 0.0
        var lapSum2 = 0.0
        var lapN = 0
        var peaks = 0
        val peakMin = max(p50 + 0.12, 0.2).toFloat()
        for (yy in 1 until h - 1) {
            for (xx in 1 until w - 1) {
                val i = yy * w + xx
                val v = y[i]
                val lap = 4f * v - y[i - 1] - y[i + 1] - y[i - w] - y[i + w]
                lapSum += lap
                lapSum2 += lap.toDouble() * lap
                lapN++
                if (v >= peakMin &&
                    v > y[i - 1] && v > y[i + 1] && v > y[i - w] && v > y[i + w] &&
                    v > y[i - w - 1] && v > y[i - w + 1] && v > y[i + w - 1] && v > y[i + w + 1]
                ) peaks++
            }
        }
        val lapVar = if (lapN > 0) max(0.0, lapSum2 / lapN - (lapSum / lapN).let { it * it }) else 0.0

        fun regionMean(x0: Int, y0: Int, x1: Int, y1: Int): Double {
            var s = 0.0
            var c = 0
            for (yy in y0 until y1) for (xx in x0 until x1) { s += y[yy * w + xx]; c++ }
            return if (c > 0) s / c else 0.0
        }
        val third = h / 3
        return ImageStats(
            mean = mean, std = std, meanLin = sumLin / n,
            p05 = pct(0.05), p50 = p50, p95 = pct(0.95),
            clipHi = hi.toDouble() / n, clipLo = lo.toDouble() / n, brightFrac = bright.toDouble() / n,
            sharpness = log10(1e-6 + lapVar),
            pointSources = peaks * 10_000.0 / n,
            rg = (sr / n) / (sg / n + 1e-4 * 255) , bg = (sb / n) / (sg / n + 1e-4 * 255),
            saturation = sat / n,
            top = regionMean(0, 0, w, third),
            bottom = regionMean(0, h - third, w, h),
            center = regionMean(w / 3, h / 3, w - w / 3, h - h / 3),
        )
    }

    private fun log2(v: Double) = ln(v) / ln(2.0)

    /** Average of [f]×[f] blocks (exact integer box filter). Output (w/f)×(h/f). */
    fun boxDownscale(px: IntArray, w: Int, h: Int, f: Int): IntArray {
        val ow = w / f
        val oh = h / f
        val out = IntArray(ow * oh)
        val area = f * f
        for (oy in 0 until oh) for (ox in 0 until ow) {
            var r = 0
            var g = 0
            var b = 0
            for (dy in 0 until f) {
                val row = (oy * f + dy) * w + ox * f
                for (dx in 0 until f) {
                    val p = px[row + dx]
                    r += p shr 16 and 0xFF; g += p shr 8 and 0xFF; b += p and 0xFF
                }
            }
            out[oy * ow + ox] = (0xFF shl 24) or ((r + area / 2) / area shl 16) or ((g + area / 2) / area shl 8) or ((b + area / 2) / area)
        }
        return out
    }

    /** Rotate clockwise by 0/90/180/270. Returns pixels and new width/height. */
    fun rotateCw(px: IntArray, w: Int, h: Int, deg: Int): Triple<IntArray, Int, Int> {
        return when (((deg % 360) + 360) % 360) {
            90 -> { val o = IntArray(w * h); for (y in 0 until h) for (x in 0 until w) o[x * h + (h - 1 - y)] = px[y * w + x]; Triple(o, h, w) }
            180 -> { val o = IntArray(w * h); for (i in 0 until w * h) o[w * h - 1 - i] = px[i]; Triple(o, w, h) }
            270 -> { val o = IntArray(w * h); for (y in 0 until h) for (x in 0 until w) o[(w - 1 - x) * h + y] = px[y * w + x]; Triple(o, h, w) }
            else -> Triple(px, w, h)
        }
    }

    /** Model input: central square of the upright preview, box-filtered to 64×64 (docs/DATASET_FORMAT.md). */
    fun thumb64(px: IntArray, w: Int, h: Int): IntArray {
        val side = minOf(w, h)
        val x0 = (w - side) / 2
        val y0 = (h - side) / 2
        val crop = IntArray(side * side)
        for (y in 0 until side) System.arraycopy(px, (y0 + y) * w + x0, crop, y * side, side)
        val f = side / THUMB
        val cs = f * THUMB
        val exact = if (cs == side) crop else IntArray(cs * cs).also { o -> for (y in 0 until cs) System.arraycopy(crop, y * side, o, y * cs, cs) }
        return boxDownscale(exact, cs, cs, f)
    }
}
