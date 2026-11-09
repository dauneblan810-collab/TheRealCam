package com.therealcam.processing

import android.graphics.Bitmap
import com.therealcam.data.StackMode
import com.therealcam.processing.core.Layer
import com.therealcam.processing.core.PlannedOutput
import com.therealcam.processing.core.StackEngine
import com.therealcam.processing.core.StackPlan
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * Honest multi-frame stacking: frames are only selected, shifted by whole pixels and combined per
 * pixel (mean, sigma-clipped mean, max) or copied (best frame). Nothing is sharpened, upscaled or
 * generated. Output is a JPEG with the reference frame's EXIF.
 */
class StackProcessor {

    /**
     * Runs on Dispatchers.Default and is cancellable. [onProgress] receives increasing values 0..1
     * on a background thread.
     *
     * @throws IllegalArgumentException if [frames] is empty
     * @throws IOException if no frame can be decoded or the output cannot be written
     */
    suspend fun process(
        frames: List<StackFrame>,
        options: StackOptions,
        output: File,
        onProgress: (Float) -> Unit = {},
    ): StackResult = withContext(Dispatchers.Default) {
        require(frames.isNotEmpty()) { "No frames to stack" }
        val quality = options.jpegQuality.coerceIn(1, 100)
        output.absoluteFile.parentFile?.mkdirs()

        JpegFrameSource(frames.map { it.file }).use { source ->
            val engine = StackEngine()
            val plan = engine.plan(frames, options, source) { onProgress(PLAN_SHARE * it) }
            val description = describe(options.mode, plan, frames.size)

            when (val out = plan.output) {
                is PlannedOutput.Copy -> {
                    val plainCopy = options.mode == StackMode.NONE || frames.size == 1
                    writeAtomically(output) { tmp ->
                        frames[out.position].file.copyTo(tmp, overwrite = true)
                        if (!plainCopy) StackExif.annotateCopy(tmp, description)
                    }
                }
                is PlannedOutput.Merge -> {
                    val w = source.width
                    val h = source.height
                    val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                    try {
                        bitmap.setHasAlpha(false)
                        engine.merge(
                            w, h, options.mode, out.layers, source, options.sigma,
                            { y0, rows, px -> bitmap.setPixels(px, 0, w, 0, y0, w, rows) },
                        ) { onProgress(PLAN_SHARE + MERGE_SHARE * it) }
                        ensureActive()
                        writeAtomically(output) { tmp ->
                            FileOutputStream(tmp).use { stream ->
                                if (!bitmap.compress(Bitmap.CompressFormat.JPEG, quality, stream)) {
                                    throw IOException("JPEG encoding failed")
                                }
                            }
                            StackExif.writeMerged(
                                frames[plan.reference].file, tmp, description, w, h,
                                summedExposureSeconds(options.mode, frames, out.layers),
                            )
                        }
                    } finally {
                        bitmap.recycle()
                    }
                }
            }
            onProgress(1f)
            StackResult(output, source.width, source.height, plan.reports, plan.usedFrames)
        }
    }

    private fun describe(mode: StackMode, plan: StackPlan, total: Int): String =
        "TheRealCam stack: ${mode.key} ${plan.usedFrames}/$total frames" + if (plan.aligned) " aligned" else ""

    /** Total light-gathering time of an averaged stack; null if any used frame's exposure is unknown. */
    private fun summedExposureSeconds(mode: StackMode, frames: List<StackFrame>, layers: List<Layer>): Double? {
        if (mode != StackMode.MEAN && mode != StackMode.SIGMA_CLIP && mode != StackMode.LUCKY) return null
        var ns = 0L
        for (layer in layers) ns += frames[layer.position].exposureNs ?: return null
        return ns / 1e9
    }

    /** Writes through a temp file in the same folder so a cancelled or failed run never leaves a partial JPEG. */
    private inline fun writeAtomically(output: File, block: (File) -> Unit) {
        val tmp = File(output.absoluteFile.parentFile, ".${output.name}.part")
        try {
            block(tmp)
            if (output.exists() && !output.delete()) throw IOException("Cannot replace $output")
            if (!tmp.renameTo(output)) tmp.copyTo(output, overwrite = true)
        } finally {
            tmp.delete()
        }
    }

    private companion object {
        const val PLAN_SHARE = 0.4f
        const val MERGE_SHARE = 0.55f
    }
}
