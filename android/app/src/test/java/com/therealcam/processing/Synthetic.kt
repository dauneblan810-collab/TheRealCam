package com.therealcam.processing

import com.therealcam.processing.core.FrameSource
import com.therealcam.processing.core.Luma8
import com.therealcam.processing.core.LumaPlane
import com.therealcam.processing.core.OPAQUE
import com.therealcam.processing.core.luma8
import java.io.File
import java.io.IOException
import java.util.Random
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/** Gaussian blob of a continuous synthetic scene (a star when small and bright). */
internal class Blob(val x: Double, val y: Double, val amp: Double, val sigma: Double)

/** Synthetic scenes are continuous functions, so integer-shifted renders are exact copies of moved content. */
internal object Synth {

    fun starField(seed: Long, count: Int, w: Int, h: Int, minAmp: Double = 60.0, maxAmp: Double = 200.0, pad: Int = 40): List<Blob> {
        val r = Random(seed)
        return List(count) {
            Blob(
                -pad + r.nextDouble() * (w + 2 * pad), -pad + r.nextDouble() * (h + 2 * pad),
                minAmp + r.nextDouble() * (maxAmp - minAmp), 0.9 + r.nextDouble() * 0.7,
            )
        }
    }

    fun texture(seed: Long, count: Int, w: Int, h: Int, pad: Int = 40): List<Blob> {
        val r = Random(seed)
        return List(count) {
            Blob(
                -pad + r.nextDouble() * (w + 2 * pad), -pad + r.nextDouble() * (h + 2 * pad),
                -70.0 + r.nextDouble() * 140.0, 1.2 + r.nextDouble() * 5.8,
            )
        }
    }

    /** Pixel (x, y) shows scene point (x - sx, y - sy), i.e. the content is displaced by (sx, sy). */
    fun render(
        w: Int, h: Int, blobs: List<Blob>, background: Double,
        sx: Int = 0, sy: Int = 0, noise: Double = 0.0, rnd: Random? = null,
    ): DoubleArray {
        val c = DoubleArray(w * h) { background }
        for (b in blobs) {
            val cx = b.x + sx
            val cy = b.y + sy
            val rad = ceil(4 * b.sigma).toInt()
            val x0 = max(0, (cx - rad).toInt())
            val x1 = min(w - 1, (cx + rad).toInt() + 1)
            val y0 = max(0, (cy - rad).toInt())
            val y1 = min(h - 1, (cy + rad).toInt() + 1)
            val k = -1.0 / (2 * b.sigma * b.sigma)
            for (y in y0..y1) for (x in x0..x1) {
                val dx = x - cx
                val dy = y - cy
                c[y * w + x] += b.amp * exp((dx * dx + dy * dy) * k)
            }
        }
        if (noise > 0) {
            val r = rnd ?: Random(0)
            for (i in c.indices) c[i] += r.nextGaussian() * noise
        }
        return c
    }

    /** Pixel indices within [halfWidth] of the segment (x0, y0)-(x1, y1). */
    fun lineMask(w: Int, h: Int, x0: Double, y0: Double, x1: Double, y1: Double, halfWidth: Double): IntArray {
        val dx = x1 - x0
        val dy = y1 - y0
        val len2 = dx * dx + dy * dy
        val out = ArrayList<Int>()
        for (y in 0 until h) for (x in 0 until w) {
            val t = (((x - x0) * dx + (y - y0) * dy) / len2).coerceIn(0.0, 1.0)
            val px = x0 + t * dx - x
            val py = y0 + t * dy - y
            if (px * px + py * py <= halfWidth * halfWidth) out += y * w + x
        }
        return out.toIntArray()
    }

    fun add(c: DoubleArray, indices: IntArray, amp: Double) {
        for (i in indices) c[i] += amp
    }

    fun boxBlur(c: DoubleArray, w: Int, h: Int, r: Int): DoubleArray {
        if (r <= 0) return c.copyOf()
        val tmp = DoubleArray(c.size)
        val out = DoubleArray(c.size)
        val norm = 1.0 / (2 * r + 1)
        for (y in 0 until h) for (x in 0 until w) {
            var s = 0.0
            for (k in -r..r) s += c[y * w + (x + k).coerceIn(0, w - 1)]
            tmp[y * w + x] = s * norm
        }
        for (y in 0 until h) for (x in 0 until w) {
            var s = 0.0
            for (k in -r..r) s += tmp[(y + k).coerceIn(0, h - 1) * w + x]
            out[y * w + x] = s * norm
        }
        return out
    }

    /** 8-bit gray ARGB, rounded and clamped like a JPEG decode. */
    fun toArgb(c: DoubleArray): IntArray = IntArray(c.size) {
        val v = c[it].roundToInt().coerceIn(0, 255)
        OPAQUE or (v shl 16) or (v shl 8) or v
    }

    fun gray(c: Int): Int = luma8(c)

    fun plane(argb: IntArray, w: Int, h: Int): LumaPlane = LumaPlane.fromArgb(w, h, argb)

    fun std(a: IntArray, b: IntArray, mask: IntArray): Double {
        var s = 0.0
        var s2 = 0.0
        for (i in mask) {
            val d = (gray(a[i]) - gray(b[i])).toDouble()
            s += d
            s2 += d * d
        }
        val m = s / mask.size
        return sqrt(s2 / mask.size - m * m)
    }

    fun frame(index: Int, motion: Double? = 0.005, exposureNs: Long? = 1_000_000_000L) =
        StackFrame(File("synthetic_$index.jpg"), index, exposureNs, 800, motion)
}

/** In-memory frames; analysis copies are factor x factor box averages (like inSampleSize). Null = undecodable. */
internal class MemoryFrameSource(
    private val frames: List<IntArray?>,
    override val width: Int,
    override val height: Int,
    private val factor: Int = 2,
) : FrameSource {

    override fun analysisLuma(position: Int): Luma8? {
        val src = frames[position] ?: return null
        val w = width / factor
        val h = height / factor
        val out = ByteArray(w * h)
        val area = factor * factor
        for (y in 0 until h) for (x in 0 until w) {
            var s = 0
            for (j in 0 until factor) for (i in 0 until factor) s += luma8(src[(y * factor + j) * width + x * factor + i])
            out[y * w + x] = ((s + area / 2) / area).toByte()
        }
        return Luma8(w, h, out)
    }

    override fun readRegion(
        position: Int, left: Int, top: Int, width: Int, height: Int,
        dst: IntArray, dstOffset: Int, dstStride: Int,
    ) {
        val src = frames[position] ?: throw IOException("frame $position unreadable")
        require(left >= 0 && top >= 0 && left + width <= this.width && top + height <= this.height) {
            "region ($left,$top ${width}x$height) outside ${this.width}x${this.height}"
        }
        for (r in 0 until height) System.arraycopy(src, (top + r) * this.width + left, dst, dstOffset + r * dstStride, width)
    }
}
