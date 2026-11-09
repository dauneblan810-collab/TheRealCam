package com.therealcam.processing.core

import kotlin.math.max
import kotlin.math.min

internal const val OPAQUE = -0x1000000 // 0xFF000000

/** Integer BT.601 luma 0..255 of an ARGB pixel (weights sum to 256). */
@Suppress("NOTHING_TO_INLINE")
internal inline fun luma8(c: Int): Int =
    (((c ushr 16) and 0xFF) * 77 + ((c ushr 8) and 0xFF) * 150 + (c and 0xFF) * 29 + 128) ushr 8

@Suppress("NOTHING_TO_INLINE")
internal inline fun packRgb(r: Int, g: Int, b: Int): Int = OPAQUE or (r shl 16) or (g shl 8) or b

/** 8-bit luminance image; compact storage for the per-frame analysis copies. */
internal class Luma8(val width: Int, val height: Int, val data: ByteArray) {
    init {
        require(data.size >= width * height)
    }

    operator fun get(x: Int, y: Int): Int = data[y * width + x].toInt() and 0xFF

    fun toPlane(): LumaPlane {
        val out = FloatArray(width * height)
        for (i in out.indices) out[i] = (data[i].toInt() and 0xFF).toFloat()
        return LumaPlane(width, height, out)
    }

    fun mean(): Double {
        var s = 0L
        for (i in 0 until width * height) s += data[i].toInt() and 0xFF
        return s.toDouble() / (width * height).coerceAtLeast(1)
    }

    /** Variance of the 4-neighbour Laplacian over the interior: a standard focus measure. */
    fun laplacianVariance(): Double {
        if (width < 3 || height < 3) return 0.0
        var s = 0.0
        var s2 = 0.0
        val d = data
        for (y in 1 until height - 1) {
            var i = y * width + 1
            for (x in 1 until width - 1) {
                val lap = 4 * (d[i].toInt() and 0xFF) -
                    (d[i - 1].toInt() and 0xFF) - (d[i + 1].toInt() and 0xFF) -
                    (d[i - width].toInt() and 0xFF) - (d[i + width].toInt() and 0xFF)
                s += lap
                s2 += lap.toDouble() * lap
                i++
            }
        }
        val n = ((width - 2) * (height - 2)).toDouble()
        val m = s / n
        return max(0.0, s2 / n - m * m)
    }

    companion object {
        fun fromArgb(width: Int, height: Int, argb: IntArray, offset: Int = 0, stride: Int = width): Luma8 {
            val out = ByteArray(width * height)
            for (y in 0 until height) {
                var si = offset + y * stride
                var di = y * width
                for (x in 0 until width) {
                    out[di++] = luma8(argb[si++]).toByte()
                }
            }
            return Luma8(width, height, out)
        }
    }
}

/** Float luminance image used for alignment. */
internal class LumaPlane(val width: Int, val height: Int, val data: FloatArray) {
    init {
        require(data.size >= width * height)
    }

    /** 2x2 box average; odd last row/column is dropped. */
    fun downsample2(): LumaPlane {
        val w = width / 2
        val h = height / 2
        val out = FloatArray(w * h)
        for (y in 0 until h) {
            val r0 = 2 * y * width
            val r1 = r0 + width
            var o = y * w
            for (x in 0 until w) {
                val sx = 2 * x
                out[o++] = 0.25f * (data[r0 + sx] + data[r0 + sx + 1] + data[r1 + sx] + data[r1 + sx + 1])
            }
        }
        return LumaPlane(w, h, out)
    }

    /** Separable box blur with clamped edges, radius r (window 2r+1). */
    fun boxBlur(r: Int): LumaPlane {
        if (r <= 0) return LumaPlane(width, height, data.copyOf(width * height))
        val tmp = FloatArray(width * height)
        val out = FloatArray(width * height)
        val norm = 1f / (2 * r + 1)
        // Horizontal pass.
        for (y in 0 until height) {
            val row = y * width
            var acc = 0f
            for (k in -r..r) acc += data[row + k.coerceIn(0, width - 1)]
            for (x in 0 until width) {
                tmp[row + x] = acc * norm
                val add = min(width - 1, x + r + 1)
                val sub = max(0, x - r)
                acc += data[row + add] - data[row + sub]
            }
        }
        // Vertical pass.
        for (x in 0 until width) {
            var acc = 0f
            for (k in -r..r) acc += tmp[k.coerceIn(0, height - 1) * width + x]
            for (y in 0 until height) {
                out[y * width + x] = acc * norm
                val add = min(height - 1, y + r + 1)
                val sub = max(0, y - r)
                acc += tmp[add * width + x] - tmp[sub * width + x]
            }
        }
        return LumaPlane(width, height, out)
    }

    /** Image minus its box blur: removes gradients/vignetting so stars, edges and texture dominate. */
    fun highPass(r: Int): LumaPlane {
        val blur = boxBlur(r)
        val out = blur.data
        for (i in 0 until width * height) out[i] = data[i] - out[i]
        return blur
    }

    companion object {
        fun fromArgb(width: Int, height: Int, argb: IntArray, offset: Int = 0, stride: Int = width): LumaPlane {
            val out = FloatArray(width * height)
            for (y in 0 until height) {
                var si = offset + y * stride
                var di = y * width
                for (x in 0 until width) out[di++] = luma8(argb[si++]).toFloat()
            }
            return LumaPlane(width, height, out)
        }
    }
}
