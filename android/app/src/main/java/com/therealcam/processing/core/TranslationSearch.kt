package com.therealcam.processing.core

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

internal data class AlignConfig(
    /** Search range at analysis resolution as a fraction of the long side (at least [minMaxShift] px). */
    val maxShiftFraction: Double = 0.06,
    val minMaxShift: Int = 16,
    /** The pyramid stops before its short side would drop below this. */
    val minCoarseSide: Int = 96,
    val maxLevels: Int = 4,
    /** High-pass radius (box blur) at every pyramid level. */
    val hpRadius: Int = 3,
    /** Local maxima of the coarse NCC map that are followed down the pyramid. */
    val candidates: Int = 4,
    /** Search radius around the upsampled estimate at each finer level. */
    val levelRadius: Int = 2,
    /** Full-resolution refinement radius (raised to the analysis scale if that is larger). */
    val refineRadius: Int = 4,
    /** Side of the central full-resolution crop used for refinement. */
    val refineCrop: Int = 768,
    val refineHpRadius: Int = 5,
    /** Minimum z = ncc * sqrt(overlap / 16) for a match to count as real structure. */
    val minZ: Double = 5.0,
    /** In a series with structure, a frame matching below this fraction of the median NCC fails. */
    val relativeNcc: Double = 0.4,
)

/**
 * Applied shift (frame -> reference) in plane pixels: frame pixel (x, y) lands on (x + dx, y + dy).
 * [subDx]/[subDy] are parabolic sub-pixel corrections in [-0.5, 0.5].
 */
internal class Translation(
    val dx: Int,
    val dy: Int,
    val subDx: Double,
    val subDy: Double,
    val ncc: Double,
    val overlap: Int,
) {
    /**
     * Significance of the correlation peak. Pure noise gives |ncc| ~ 1/sqrt(n); the /16 accounts for
     * neighbouring pixels being correlated (demosaicing, JPEG), so this is deliberately conservative.
     */
    val z: Double get() = if (overlap <= 0) 0.0 else ncc * sqrt(overlap / 16.0)

    override fun toString() = "Translation(dx=$dx, dy=$dy, ncc=${"%.3f".format(ncc)}, z=${"%.1f".format(z)})"
}

/** High-passed pyramid, level 0 = finest. */
internal class AlignPyramid(val levels: List<LumaPlane>) {
    val base: LumaPlane get() = levels[0]
}

internal object Ncc {
    const val INVALID = -2.0

    /**
     * Normalized cross-correlation of zero-mean (high-passed) planes, comparing ref(x, y) with
     * frame(x - dx, y - dy). Planes are placed in a common coordinate system by their origins; [margin]
     * pixels at each plane border are ignored (filter edge effects).
     */
    fun ncc(
        ref: LumaPlane, rox: Int, roy: Int,
        frame: LumaPlane, fox: Int, foy: Int,
        dx: Int, dy: Int, margin: Int,
    ): Double {
        val x0 = max(rox + margin, fox + dx + margin)
        val x1 = min(rox + ref.width - margin, fox + frame.width + dx - margin)
        val y0 = max(roy + margin, foy + dy + margin)
        val y1 = min(roy + ref.height - margin, foy + frame.height + dy - margin)
        if (x1 - x0 < 2 || y1 - y0 < 2) return INVALID
        val rd = ref.data
        val fd = frame.data
        val len = x1 - x0
        var sab = 0.0
        var saa = 0.0
        var sbb = 0.0
        for (y in y0 until y1) {
            var ri = (y - roy) * ref.width + (x0 - rox)
            var fi = (y - dy - foy) * frame.width + (x0 - dx - fox)
            var ab = 0f
            var aa = 0f
            var bb = 0f
            for (k in 0 until len) {
                val a = rd[ri++]
                val b = fd[fi++]
                ab += a * b
                aa += a * a
                bb += b * b
            }
            sab += ab
            saa += aa
            sbb += bb
        }
        if (saa <= 0.0 || sbb <= 0.0) return 0.0
        return sab / sqrt(saa * sbb)
    }

    fun overlap(
        ref: LumaPlane, rox: Int, roy: Int,
        frame: LumaPlane, fox: Int, foy: Int,
        dx: Int, dy: Int, margin: Int,
    ): Int {
        val w = min(rox + ref.width - margin, fox + frame.width + dx - margin) - max(rox + margin, fox + dx + margin)
        val h = min(roy + ref.height - margin, foy + frame.height + dy - margin) - max(roy + margin, foy + dy + margin)
        return if (w <= 0 || h <= 0) 0 else w * h
    }
}

/** Global integer translation by coarse-to-fine NCC search on high-passed luminance. No rotation, no warping. */
internal class TranslationSearcher(val cfg: AlignConfig = AlignConfig()) {

    fun levelCount(width: Int, height: Int): Int {
        var n = 0
        var w = width
        var h = height
        while (n < cfg.maxLevels && min(w, h) / 2 >= cfg.minCoarseSide) {
            w /= 2
            h /= 2
            n++
        }
        return n
    }

    fun maxShift(width: Int, height: Int): Int =
        max(cfg.minMaxShift, (cfg.maxShiftFraction * max(width, height)).roundToInt())

    fun prepare(luma: LumaPlane): AlignPyramid {
        val n = levelCount(luma.width, luma.height)
        val out = ArrayList<LumaPlane>(n + 1)
        var cur = luma
        for (k in 0..n) {
            out += cur.highPass(cfg.hpRadius)
            if (k < n) cur = cur.downsample2()
        }
        return AlignPyramid(out)
    }

    /** Best applied shift of [frame] onto [ref] within ±[maxShift] plane pixels. */
    fun search(ref: AlignPyramid, frame: AlignPyramid, maxShift: Int = maxShift(ref.base.width, ref.base.height)): Translation {
        val top = min(ref.levels.size, frame.levels.size) - 1
        val margin = cfg.hpRadius + 1
        val factor = 1 shl top
        val r = (maxShift + factor - 1) / factor + 1
        val side = 2 * r + 1
        val map = DoubleArray(side * side)
        val rTop = ref.levels[top]
        val fTop = frame.levels[top]
        for (dy in -r..r) for (dx in -r..r) {
            map[(dy + r) * side + dx + r] = score(rTop, fTop, dx, dy, margin)
        }

        var bestDx = 0
        var bestDy = 0
        var best = Ncc.INVALID
        for (cand in localMaxima(map, side, cfg.candidates)) {
            var cx = cand % side - r
            var cy = cand / side - r
            var v = map[cand]
            for (level in top - 1 downTo 0) {
                val rl = ref.levels[level]
                val fl = frame.levels[level]
                val sx = 2 * cx
                val sy = 2 * cy
                v = Ncc.INVALID
                for (dy in sy - cfg.levelRadius..sy + cfg.levelRadius) {
                    for (dx in sx - cfg.levelRadius..sx + cfg.levelRadius) {
                        val s = score(rl, fl, dx, dy, margin)
                        if (s > v) {
                            v = s
                            cx = dx
                            cy = dy
                        }
                    }
                }
            }
            if (v > best) {
                best = v
                bestDx = cx
                bestDy = cy
            }
        }

        val r0 = ref.base
        val f0 = frame.base
        if (best <= Ncc.INVALID) return Translation(0, 0, 0.0, 0.0, 0.0, 0)
        val subX = parabolic(
            score(r0, f0, bestDx - 1, bestDy, margin), best, score(r0, f0, bestDx + 1, bestDy, margin),
        )
        val subY = parabolic(
            score(r0, f0, bestDx, bestDy - 1, margin), best, score(r0, f0, bestDx, bestDy + 1, margin),
        )
        return Translation(bestDx, bestDy, subX, subY, best, Ncc.overlap(r0, 0, 0, f0, 0, 0, bestDx, bestDy, margin))
    }

    /** NCC at one shift; INVALID if the overlap is under a quarter of the reference. */
    private fun score(ref: LumaPlane, frame: LumaPlane, dx: Int, dy: Int, margin: Int): Double {
        val overlap = Ncc.overlap(ref, 0, 0, frame, 0, 0, dx, dy, margin)
        if (overlap * 4L < ref.width.toLong() * ref.height) return Ncc.INVALID
        return Ncc.ncc(ref, 0, 0, frame, 0, 0, dx, dy, margin)
    }

    /** Indices of the [k] largest valid local maxima (8-neighbourhood) of a square map. */
    private fun localMaxima(map: DoubleArray, side: Int, k: Int): List<Int> {
        val found = ArrayList<Int>()
        for (y in 0 until side) for (x in 0 until side) {
            val v = map[y * side + x]
            if (v <= Ncc.INVALID) continue
            var isMax = true
            loop@ for (ny in max(0, y - 1)..min(side - 1, y + 1)) {
                for (nx in max(0, x - 1)..min(side - 1, x + 1)) {
                    if ((nx != x || ny != y) && map[ny * side + nx] > v) {
                        isMax = false
                        break@loop
                    }
                }
            }
            if (isMax) found += y * side + x
        }
        return found.sortedByDescending { map[it] }.take(max(1, k))
    }

    private fun parabolic(fm: Double, f0: Double, fp: Double): Double {
        if (fm <= Ncc.INVALID || fp <= Ncc.INVALID) return 0.0
        val den = fm - 2 * f0 + fp
        if (den >= 0.0) return 0.0
        return (0.5 * (fm - fp) / den).coerceIn(-0.5, 0.5)
    }
}

internal class Refinement(val dx: Int, val dy: Int, val ncc: Double, val overlap: Int, val onBorder: Boolean) {
    val z: Double get() = if (overlap <= 0) 0.0 else ncc * sqrt(overlap / 16.0)
}

internal object FullResRefiner {
    /**
     * Exhaustive search of applied shifts in [cdx ± radius] x [cdy ± radius] between high-passed
     * full-resolution patches placed at their image origins. Null if no shift had a valid overlap.
     */
    fun refine(
        refHp: LumaPlane, rox: Int, roy: Int,
        frameHp: LumaPlane, fox: Int, foy: Int,
        cdx: Int, cdy: Int, radius: Int, margin: Int,
    ): Refinement? {
        var best = Ncc.INVALID
        var bx = cdx
        var by = cdy
        for (dy in cdy - radius..cdy + radius) {
            for (dx in cdx - radius..cdx + radius) {
                val v = Ncc.ncc(refHp, rox, roy, frameHp, fox, foy, dx, dy, margin)
                if (v > best) {
                    best = v
                    bx = dx
                    by = dy
                }
            }
        }
        if (best <= Ncc.INVALID) return null
        val onBorder = abs(bx - cdx) == radius || abs(by - cdy) == radius
        return Refinement(bx, by, best, Ncc.overlap(refHp, rox, roy, frameHp, fox, foy, bx, by, margin), onBorder)
    }
}

internal enum class AlignVerdict { OK, FAILED, NO_STRUCTURE }

/**
 * Series-level decision. If the typical frame matches with real structure, frames that match far
 * worse fail. If the whole series has no structure (fog, clear sky), there is nothing to align
 * against: those frames are kept unshifted instead of being moved by noise.
 */
internal fun judgeAlignment(ncc: DoubleArray, z: DoubleArray, cfg: AlignConfig): Array<AlignVerdict> {
    if (ncc.isEmpty()) return emptyArray()
    val structured = RobustStats.median(z) >= cfg.minZ
    val medNcc = RobustStats.median(ncc)
    return Array(ncc.size) { i ->
        when {
            structured && (z[i] < cfg.minZ || ncc[i] < cfg.relativeNcc * medNcc) -> AlignVerdict.FAILED
            !structured && z[i] < cfg.minZ -> AlignVerdict.NO_STRUCTURE
            else -> AlignVerdict.OK
        }
    }
}
