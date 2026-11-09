package com.therealcam.processing.core

import com.therealcam.data.StackMode
import com.therealcam.processing.FrameReport
import com.therealcam.processing.RejectReason
import com.therealcam.processing.StackFrame
import com.therealcam.processing.StackOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** Pixel access to the frames of a series (JPEG decoders on Android, arrays in tests). */
internal interface FrameSource {
    /** Full-resolution size shared by all frames. */
    val width: Int
    val height: Int

    /** Downscaled luminance for analysis, or null if the frame cannot be used. */
    fun analysisLuma(position: Int): Luma8?

    /** Full-resolution ARGB of a rect inside the image. May be called concurrently for different frames. */
    fun readRegion(position: Int, left: Int, top: Int, width: Int, height: Int, dst: IntArray, dstOffset: Int, dstStride: Int)
}

/** Receives finished output rows [y0, y0 + rows) (row stride = image width), in order, from one coroutine. */
internal fun interface StripSink {
    fun write(y0: Int, rows: Int, argb: IntArray)
}

/** A frame placed with its applied shift (frame pixel (x, y) -> output (x + dx, y + dy)). */
internal data class Layer(val position: Int, val dx: Int, val dy: Int)

internal sealed interface PlannedOutput {
    /** Output is a byte copy of this frame's file. */
    data class Copy(val position: Int) : PlannedOutput

    /** Output is merged from these layers; the reference comes first. */
    data class Merge(val layers: List<Layer>) : PlannedOutput
}

internal class StackPlan(
    val reports: List<FrameReport>,
    /** Position of the reference frame (EXIF source). */
    val reference: Int,
    val output: PlannedOutput,
    val aligned: Boolean,
) {
    val usedFrames: Int
        get() = when (output) {
            is PlannedOutput.Copy -> 1
            is PlannedOutput.Merge -> output.layers.size
        }
}

/** Largest power-of-two sample size that keeps the long side at or above [minLongSide]. */
internal fun analysisSampleSize(width: Int, height: Int, minLongSide: Int = 800): Int {
    val long = max(width, height)
    var s = 1
    while (long / (s * 2) >= minLongSide) s *= 2
    return s
}

internal fun defaultMemoryBudget(): Long {
    val rt = Runtime.getRuntime()
    val free = rt.maxMemory() - (rt.totalMemory() - rt.freeMemory())
    return (free / 5 * 2).coerceIn(16L shl 20, 128L shl 20)
}

/**
 * Deterministic stacking: analysis -> frame selection -> translation alignment -> strip-wise merge.
 * Pure Kotlin; the Android adapter supplies decoding ([FrameSource]) and encoding ([StripSink]).
 */
internal class StackEngine(
    private val workers: Int = defaultWorkers(),
    /** Heap budget for the per-strip buffers; decides the strip height. */
    private val memoryBudgetBytes: Long = defaultMemoryBudget(),
    /** Fixed strip height (tests); 0 = derive from the budget. */
    private val stripRowsOverride: Int = 0,
    private val alignConfig: AlignConfig = AlignConfig(),
) {

    suspend fun plan(
        frames: List<StackFrame>,
        options: StackOptions,
        source: FrameSource,
        onProgress: (Float) -> Unit = {},
    ): StackPlan = withContext(Dispatchers.Default) {
        require(frames.isNotEmpty()) { "No frames to stack" }
        val n = frames.size
        val progress = Progress(onProgress)

        val lumas = arrayOfNulls<Luma8>(n)
        val means = DoubleArray(n)
        val sharpness = DoubleArray(n)
        val analysed = AtomicInteger()
        parallelFor(n, workers) { i, _ ->
            val l = source.analysisLuma(i)
            if (l != null) {
                lumas[i] = l
                means[i] = l.mean()
                sharpness[i] = l.laplacianVariance()
            }
            progress.report(0.5f * analysed.incrementAndGet() / n)
        }

        if (options.mode == StackMode.NONE || n == 1) {
            val reports = List(n) { FrameReport(frames[it].index, true, null, 0, 0, sharpness[it], means[it]) }
            return@withContext StackPlan(reports, 0, PlannedOutput.Copy(0), aligned = false)
        }

        val canonical = lumas.firstOrNull { it != null }
            ?: throw IOException("None of the $n frames could be decoded")
        val stats = List(n) { i ->
            val l = lumas[i]
            val usable = l != null && l.width == canonical.width && l.height == canonical.height
            if (!usable) lumas[i] = null
            FrameStats(means[i], sharpness[i], frames[i].motionRms, usable)
        }

        val selection = FrameSelector.select(
            stats, options.mode, options.rejectOutlierFrames, options.maxMotionRadS, options.luckyKeepFraction,
        )
        val reasons = selection.reasons
        val ref = selection.reference
        val dx = IntArray(n)
        val dy = IntArray(n)
        val merges = options.mode in MERGE_MODES
        val targets = reasons.indices.filter { reasons[it] == null && it != ref }
        val aligned = merges && options.align && targets.isNotEmpty()
        if (aligned) align(source, lumas, ref, targets, dx, dy, reasons, progress)
        progress.report(1f)

        val used = listOf(ref) + reasons.indices.filter { reasons[it] == null && it != ref }
        val output = if (!merges || used.size == 1) {
            PlannedOutput.Copy(ref)
        } else {
            PlannedOutput.Merge(used.map { Layer(it, dx[it], dy[it]) })
        }
        val reports = List(n) { i ->
            FrameReport(frames[i].index, reasons[i] == null, reasons[i], dx[i], dy[i], sharpness[i], means[i])
        }
        StackPlan(reports, ref, output, aligned)
    }

    private suspend fun align(
        source: FrameSource,
        lumas: Array<Luma8?>,
        ref: Int,
        targets: List<Int>,
        dx: IntArray,
        dy: IntArray,
        reasons: Array<String?>,
        progress: Progress,
    ) {
        val cfg = alignConfig
        val searcher = TranslationSearcher(cfg)
        val refLuma = lumas[ref]!!
        val scaleX = source.width.toDouble() / refLuma.width
        val scaleY = source.height.toDouble() / refLuma.height
        val refPyramid = searcher.prepare(refLuma.toPlane())
        val maxShift = searcher.maxShift(refLuma.width, refLuma.height)
        val radius = max(cfg.refineRadius, ceil(max(scaleX, scaleY)).toInt())
        val margin = cfg.refineHpRadius + 1
        val crop = referenceCrop(source, ref, margin)

        val m = targets.size
        val ncc = DoubleArray(m)
        val z = DoubleArray(m)
        val fx = IntArray(m)
        val fy = IntArray(m)
        val done = AtomicInteger()
        parallelFor(m, workers) { k, _ ->
            val i = targets[k]
            val t = searcher.search(refPyramid, searcher.prepare(lumas[i]!!.toPlane()), maxShift)
            ncc[k] = t.ncc
            z[k] = t.z
            var sx = ((t.dx + t.subDx) * scaleX).roundToInt()
            var sy = ((t.dy + t.subDy) * scaleY).roundToInt()
            if (crop != null) {
                val r = refine(source, i, crop, sx, sy, radius, margin)
                // A peak on the window border means the true peak may lie outside: keep the coarse estimate.
                if (r != null && !r.onBorder && r.z >= cfg.minZ) {
                    sx = r.dx
                    sy = r.dy
                }
            }
            fx[k] = sx
            fy[k] = sy
            progress.report(0.5f + 0.5f * done.incrementAndGet() / m)
        }

        val verdicts = judgeAlignment(ncc, z, cfg)
        for (k in 0 until m) {
            val i = targets[k]
            when (verdicts[k]) {
                AlignVerdict.OK -> {
                    dx[i] = fx[k]
                    dy[i] = fy[k]
                }
                AlignVerdict.FAILED -> reasons[i] = RejectReason.ALIGN_FAILED
                AlignVerdict.NO_STRUCTURE -> Unit
            }
        }
    }

    private class Crop(val hp: LumaPlane, val left: Int, val top: Int)

    private fun referenceCrop(source: FrameSource, ref: Int, margin: Int): Crop? {
        val cw = min(alignConfig.refineCrop, source.width / 2)
        val ch = min(alignConfig.refineCrop, source.height / 2)
        if (cw < 8 * margin || ch < 8 * margin) return null
        val left = (source.width - cw) / 2
        val top = (source.height - ch) / 2
        val px = IntArray(cw * ch)
        source.readRegion(ref, left, top, cw, ch, px, 0, cw)
        return Crop(LumaPlane.fromArgb(cw, ch, px).highPass(alignConfig.refineHpRadius), left, top)
    }

    private fun refine(source: FrameSource, position: Int, crop: Crop, cdx: Int, cdy: Int, radius: Int, margin: Int): Refinement? {
        val l = max(0, crop.left - cdx - radius)
        val t = max(0, crop.top - cdy - radius)
        val r = min(source.width, crop.left + crop.hp.width - cdx + radius)
        val b = min(source.height, crop.top + crop.hp.height - cdy + radius)
        if (r - l < 4 * margin || b - t < 4 * margin) return null
        val w = r - l
        val h = b - t
        val px = IntArray(w * h)
        source.readRegion(position, l, t, w, h, px, 0, w)
        val hp = LumaPlane.fromArgb(w, h, px).highPass(alignConfig.refineHpRadius)
        return FullResRefiner.refine(crop.hp, crop.left, crop.top, hp, l, t, cdx, cdy, radius, margin)
    }

    /**
     * Merges [layers] at full resolution in horizontal strips. Memory is bounded by the strip buffers:
     * SIGMA_CLIP/LUCKY hold one strip per layer, MEAN/LIGHTEN stream layers through running accumulators.
     * Output pixels covered by no layer are black (cannot happen when the zero-shift reference is included).
     */
    suspend fun merge(
        width: Int,
        height: Int,
        mode: StackMode,
        layers: List<Layer>,
        source: FrameSource,
        sigma: Float,
        sink: StripSink,
        onProgress: (Float) -> Unit = {},
    ) = withContext(Dispatchers.Default) {
        require(layers.isNotEmpty()) { "No layers to merge" }
        require(width > 0 && height > 0) { "Bad size ${width}x$height" }
        val progress = Progress(onProgress)
        val s = if (sigma.isFinite() && sigma >= 0f) sigma else 2.5f
        when (mode) {
            StackMode.SIGMA_CLIP, StackMode.LUCKY -> mergeSigmaClip(width, height, layers, source, s, sink, progress)
            StackMode.MEAN -> mergeStreaming(width, height, layers, source, mean = true, sink, progress)
            StackMode.LIGHTEN -> mergeStreaming(width, height, layers, source, mean = false, sink, progress)
            else -> throw IllegalArgumentException("$mode does not merge pixels")
        }
    }

    private suspend fun mergeSigmaClip(
        width: Int, height: Int, layers: List<Layer>, source: FrameSource, sigma: Float,
        sink: StripSink, progress: Progress,
    ) {
        val n = layers.size
        val rows = stripRows(width, height, n + 1)
        val strips = Array(n) { FrameStrip() }
        val scratch = Array(workers) { MergeScratch(n) }
        val out = IntArray(rows * width)
        var y0 = 0
        while (y0 < height) {
            val top = y0
            val h = min(rows, height - top)
            parallelFor(n, workers) { j, _ -> fill(strips[j], layers[j], top, h, width, height, source) }
            parallelRanges(h, workers) { r0, r1, w ->
                StripKernels.sigmaClipRows(strips, n, top, r0, r1, width, out, sigma, scratch[w])
            }
            sink.write(top, h, out)
            y0 += h
            progress.report(y0.toFloat() / height)
        }
    }

    private suspend fun mergeStreaming(
        width: Int, height: Int, layers: List<Layer>, source: FrameSource, mean: Boolean,
        sink: StripSink, progress: Progress,
    ) {
        val n = layers.size
        val group = min(n, max(2, workers))
        val rows = stripRows(width, height, group + if (mean) 5 else 2)
        val size = rows * width
        val strips = Array(group) { FrameStrip() }
        val out = IntArray(size)
        val sumR = if (mean) IntArray(size) else null
        val sumG = if (mean) IntArray(size) else null
        val sumB = if (mean) IntArray(size) else null
        val cnt = if (mean) IntArray(size) else null
        val acc = if (mean) null else IntArray(size)
        var y0 = 0
        while (y0 < height) {
            val top = y0
            val h = min(rows, height - top)
            val used = h * width
            if (mean) {
                sumR!!.fill(0, 0, used)
                sumG!!.fill(0, 0, used)
                sumB!!.fill(0, 0, used)
                cnt!!.fill(0, 0, used)
            } else {
                acc!!.fill(0, 0, used)
            }
            var start = 0
            while (start < n) {
                val first = start
                val c = min(group, n - first)
                parallelFor(c, workers) { j, _ -> fill(strips[j], layers[first + j], top, h, width, height, source) }
                parallelRanges(h, workers) { r0, r1, _ ->
                    for (j in 0 until c) {
                        if (mean) {
                            StripKernels.accumulateMean(strips[j], top, r0, r1, width, sumR!!, sumG!!, sumB!!, cnt!!)
                        } else {
                            StripKernels.accumulateMax(strips[j], top, r0, r1, width, acc!!)
                        }
                    }
                }
                start += c
            }
            parallelRanges(h, workers) { r0, r1, _ ->
                if (mean) {
                    StripKernels.finishMean(r0, r1, width, sumR!!, sumG!!, sumB!!, cnt!!, out)
                } else {
                    StripKernels.finishMax(r0, r1, width, acc!!, out)
                }
            }
            sink.write(top, h, out)
            y0 += h
            progress.report(y0.toFloat() / height)
        }
    }

    /** Reads the frame rows that land in output rows [y0, y0 + rows) (full width) and positions them. */
    private fun fill(strip: FrameStrip, layer: Layer, y0: Int, rows: Int, width: Int, height: Int, source: FrameSource) {
        val sy0 = max(0, y0 - layer.dy)
        val sy1 = min(height, y0 + rows - layer.dy)
        if (sy0 >= sy1 || layer.dx >= width || layer.dx <= -width) {
            strip.height = 0
            return
        }
        strip.ensureCapacity((sy1 - sy0) * width)
        source.readRegion(layer.position, 0, sy0, width, sy1 - sy0, strip.pixels, 0, width)
        strip.place(sy0, sy1 - sy0, width, layer.dx, layer.dy)
    }

    private fun stripRows(width: Int, height: Int, intsPerPixel: Int): Int {
        if (stripRowsOverride > 0) return stripRowsOverride.coerceIn(1, height)
        val bytesPerRow = 4L * width * intsPerPixel
        var rows = (memoryBudgetBytes / bytesPerRow).coerceIn(MIN_STRIP_ROWS.toLong(), MAX_STRIP_ROWS.toLong()).toInt()
        if (rows >= 32) rows -= rows % 16 // whole JPEG MCU rows
        return rows.coerceIn(1, height)
    }

    companion object {
        const val MIN_STRIP_ROWS = 16
        const val MAX_STRIP_ROWS = 256
        val MERGE_MODES = setOf(StackMode.MEAN, StackMode.SIGMA_CLIP, StackMode.LUCKY, StackMode.LIGHTEN)
    }
}
