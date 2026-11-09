package com.therealcam.processing.core

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Full-resolution pixels of one frame for the current output strip, already positioned:
 * output pixel (x, y) is covered iff outX0 <= x < outX0 + width and outY0 <= y < outY0 + height,
 * and its value is pixels[(y - outY0) * stride + (x - outX0)].
 */
internal class FrameStrip(capacity: Int = 0) {
    var pixels = IntArray(capacity)
        private set
    var stride = 0
    var outX0 = 0
    var outY0 = 0
    var width = 0
    /** 0 = the frame covers nothing in this strip. */
    var height = 0

    fun ensureCapacity(n: Int) {
        if (pixels.size < n) pixels = IntArray(n)
    }

    /** Positions raw frame pixels (frame rows starting at [srcY0], full width) shifted by (dx, dy). */
    fun place(srcY0: Int, rows: Int, frameWidth: Int, dx: Int, dy: Int) {
        stride = frameWidth
        width = frameWidth
        height = rows
        outX0 = dx
        outY0 = srcY0 + dy
    }
}

/** Per-worker scratch, sized for the number of layers; keeps the per-pixel loops allocation-free. */
internal class MergeScratch(maxLayers: Int) {
    val px = arrayOfNulls<IntArray>(maxLayers)
    val rowBase = IntArray(maxLayers)
    val xa = IntArray(maxLayers)
    val xb = IntArray(maxLayers)
    val argb = IntArray(maxLayers)
    val lum = IntArray(maxLayers)
    val tmp = IntArray(maxLayers)
}

internal object StripKernels {
    /** Lower bound for the robust per-pixel sigma, in luma levels (8-bit data with MAD = 0). */
    const val MIN_SIGMA = 1.0f

    /**
     * Sigma-clipped mean for strip rows [rowFrom, rowTo) of an output strip starting at image row y0.
     * Outlier frames are decided per pixel on luminance (|l - median| > sigma * 1.4826 * MAD), then
     * the RGB of the remaining frames is averaged. Writes out[r * width + x].
     */
    fun sigmaClipRows(
        layers: Array<FrameStrip>, count: Int, y0: Int, rowFrom: Int, rowTo: Int, width: Int,
        out: IntArray, sigma: Float, s: MergeScratch,
    ) {
        val px = s.px
        val rowBase = s.rowBase
        val xa = s.xa
        val xb = s.xb
        val argb = s.argb
        val lum = s.lum
        val tmp = s.tmp
        for (r in rowFrom until rowTo) {
            val m = activeLayers(layers, count, y0 + r, width, s)
            val o = r * width
            for (x in 0 until width) {
                var n = 0
                for (j in 0 until m) {
                    if (x < xa[j] || x >= xb[j]) continue
                    val c = px[j]!![rowBase[j] + x]
                    argb[n] = c
                    val l = luma8(c)
                    lum[n] = l
                    tmp[n] = l
                    n++
                }
                out[o + x] = when (n) {
                    0 -> OPAQUE
                    1 -> argb[0] or OPAQUE
                    2 -> mean2(argb[0], argb[1])
                    else -> clippedMean(argb, lum, tmp, n, sigma)
                }
            }
        }
    }

    private fun clippedMean(argb: IntArray, lum: IntArray, tmp: IntArray, n: Int, sigma: Float): Int {
        val k = (n - 1) / 2
        val med = RobustStats.kthSmallest(tmp, n, k)
        for (i in 0 until n) tmp[i] = abs(lum[i] - med)
        val mad = RobustStats.kthSmallest(tmp, n, k)
        val thr = sigma * max(MAD_TO_SIGMA.toFloat() * mad, MIN_SIGMA)
        var sr = 0
        var sg = 0
        var sb = 0
        var kept = 0
        for (i in 0 until n) {
            if (abs(lum[i] - med) <= thr) {
                val c = argb[i]
                sr += (c ushr 16) and 0xFF
                sg += (c ushr 8) and 0xFF
                sb += c and 0xFF
                kept++
            }
        }
        if (kept == 0) {
            // Only reachable with sigma < 0; fall back to the plain mean.
            for (i in 0 until n) {
                val c = argb[i]
                sr += (c ushr 16) and 0xFF
                sg += (c ushr 8) and 0xFF
                sb += c and 0xFF
            }
            kept = n
        }
        val h = kept / 2
        return packRgb((sr + h) / kept, (sg + h) / kept, (sb + h) / kept)
    }

    private fun mean2(a: Int, b: Int): Int = packRgb(
        (((a ushr 16) and 0xFF) + ((b ushr 16) and 0xFF) + 1) / 2,
        (((a ushr 8) and 0xFF) + ((b ushr 8) and 0xFF) + 1) / 2,
        ((a and 0xFF) + (b and 0xFF) + 1) / 2,
    )

    /** Collects the layers covering image row y into the scratch arrays; returns their count. */
    private fun activeLayers(layers: Array<FrameStrip>, count: Int, y: Int, width: Int, s: MergeScratch): Int {
        var m = 0
        for (i in 0 until count) {
            val l = layers[i]
            val ly = y - l.outY0
            if (ly < 0 || ly >= l.height) continue
            val a = max(0, l.outX0)
            val b = min(width, l.outX0 + l.width)
            if (a >= b) continue
            s.px[m] = l.pixels
            s.rowBase[m] = ly * l.stride - l.outX0
            s.xa[m] = a
            s.xb[m] = b
            m++
        }
        return m
    }

    /** Adds one layer to per-channel sums and the coverage count (MEAN). */
    fun accumulateMean(
        l: FrameStrip, y0: Int, rowFrom: Int, rowTo: Int, width: Int,
        sumR: IntArray, sumG: IntArray, sumB: IntArray, cnt: IntArray,
    ) {
        if (l.height == 0) return
        val xa = max(0, l.outX0)
        val xb = min(width, l.outX0 + l.width)
        if (xa >= xb) return
        val p = l.pixels
        for (r in rowFrom until rowTo) {
            val ly = y0 + r - l.outY0
            if (ly < 0 || ly >= l.height) continue
            val base = ly * l.stride - l.outX0
            val o = r * width
            for (x in xa until xb) {
                val c = p[base + x]
                val i = o + x
                sumR[i] += (c ushr 16) and 0xFF
                sumG[i] += (c ushr 8) and 0xFF
                sumB[i] += c and 0xFF
                cnt[i]++
            }
        }
    }

    fun finishMean(
        rowFrom: Int, rowTo: Int, width: Int,
        sumR: IntArray, sumG: IntArray, sumB: IntArray, cnt: IntArray, out: IntArray,
    ) {
        for (i in rowFrom * width until rowTo * width) {
            val n = cnt[i]
            out[i] = if (n == 0) OPAQUE else {
                val h = n / 2
                packRgb((sumR[i] + h) / n, (sumG[i] + h) / n, (sumB[i] + h) / n)
            }
        }
    }

    /** Per-channel maximum into packed RGB accumulators (LIGHTEN); accumulators start at 0. */
    fun accumulateMax(l: FrameStrip, y0: Int, rowFrom: Int, rowTo: Int, width: Int, acc: IntArray) {
        if (l.height == 0) return
        val xa = max(0, l.outX0)
        val xb = min(width, l.outX0 + l.width)
        if (xa >= xb) return
        val p = l.pixels
        for (r in rowFrom until rowTo) {
            val ly = y0 + r - l.outY0
            if (ly < 0 || ly >= l.height) continue
            val base = ly * l.stride - l.outX0
            val o = r * width
            for (x in xa until xb) {
                val c = p[base + x]
                val a = acc[o + x]
                val rr = max((a ushr 16) and 0xFF, (c ushr 16) and 0xFF)
                val gg = max((a ushr 8) and 0xFF, (c ushr 8) and 0xFF)
                val bb = max(a and 0xFF, c and 0xFF)
                acc[o + x] = (rr shl 16) or (gg shl 8) or bb
            }
        }
    }

    fun finishMax(rowFrom: Int, rowTo: Int, width: Int, acc: IntArray, out: IntArray) {
        for (i in rowFrom * width until rowTo * width) out[i] = acc[i] or OPAQUE
    }
}
