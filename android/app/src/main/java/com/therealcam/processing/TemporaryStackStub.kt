package com.therealcam.processing

import com.therealcam.data.StackMode
import java.io.File

// TEMPORARY: placeholder with the agreed API until the real engine branch is merged; removed then.

data class StackFrame(val file: File, val index: Int, val exposureNs: Long?, val iso: Int?, val motionRms: Double?)

data class StackOptions(
    val mode: StackMode,
    val align: Boolean,
    val rejectOutlierFrames: Boolean,
    val maxMotionRadS: Double = 0.03,
    val luckyKeepFraction: Float = 0.4f,
    val sigma: Float = 2.5f,
    val jpegQuality: Int = 95,
)

data class FrameReport(val index: Int, val accepted: Boolean, val rejectReason: String?, val dx: Int, val dy: Int, val sharpness: Double, val meanLuma: Double)

data class StackResult(val output: File, val width: Int, val height: Int, val frames: List<FrameReport>, val usedFrames: Int)

class StackProcessor {
    suspend fun process(frames: List<StackFrame>, options: StackOptions, output: File, onProgress: (Float) -> Unit = {}): StackResult {
        require(frames.isNotEmpty())
        frames.first().file.copyTo(output, overwrite = true)
        onProgress(1f)
        return StackResult(output, 0, 0, frames.mapIndexed { i, f -> FrameReport(f.index, i == 0, if (i == 0) null else "not_sharpest", 0, 0, 0.0, 0.0) }, 1)
    }
}
