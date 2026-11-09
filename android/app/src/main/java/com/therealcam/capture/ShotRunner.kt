package com.therealcam.capture

import android.graphics.BitmapFactory
import android.location.Location
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import com.therealcam.AppGraph
import com.therealcam.ai.SceneFeatures
import com.therealcam.ai.ScenePreview
import com.therealcam.camera.CameraCaps
import com.therealcam.camera.CameraController
import com.therealcam.camera.CameraMode
import com.therealcam.camera.CapturedFrame
import com.therealcam.camera.PreviewSnapshot
import com.therealcam.camera.ShotSettings
import com.therealcam.camera.StillRequest
import com.therealcam.core.Ids
import com.therealcam.core.ImageIo
import com.therealcam.data.AppSettings
import com.therealcam.data.Profile
import com.therealcam.data.StackMode
import com.therealcam.data.db.CaptureEntity
import com.therealcam.data.db.Roles
import com.therealcam.dataset.AiInfo
import com.therealcam.dataset.MetaBuilder
import com.therealcam.dataset.PhotoInfo
import com.therealcam.dataset.SampleMeta
import com.therealcam.dataset.SeriesFrameInfo
import com.therealcam.dataset.SeriesInfo
import com.therealcam.processing.StackFrame
import com.therealcam.processing.StackOptions
import com.therealcam.processing.StackProcessor
import com.therealcam.sensors.MotionState
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

/** Executes one press of the shutter: a single frame or a series that is then merged honestly. */
class ShotRunner(private val g: AppGraph, private val camera: CameraController) {

    data class Plan(
        val mode: CameraMode,
        val settings: ShotSettings,
        val profile: Profile,
        /** Category assigned up front (chosen by the user or the AI); can be changed when rating. */
        val categoryId: String?,
        val ai: AiInfo?,
        val timerS: Int,
    )

    data class Env(
        val caps: CameraCaps,
        val preview: PreviewSnapshot?,
        val scene: ScenePreview?,
        val features: Map<String, Double?>,
        val motion: MotionState,
        val location: Location?,
        val app: AppSettings,
    )

    sealed interface Progress {
        data class Capturing(val frame: Int, val total: Int, val exposureNs: Long?) : Progress
        data class Merging(val fraction: Float) : Progress
        data object Saving : Progress
    }

    private class FrameRecord(
        val index: Int,
        val id: String,
        val timeMs: Long,
        val file: File,
        val frame: CapturedFrame,
        val settings: ShotSettings,
        val motion: MetaBuilder.ExposureMotion,
    )

    suspend fun run(plan: Plan, env: Env, onProgress: (Progress) -> Unit): CaptureEntity {
        val timeMs = System.currentTimeMillis()
        val sampleId = Ids.newSampleId(timeMs)
        val orientation = CameraController.jpegOrientation(env.caps, env.motion.deviceOrientation)
        val location = if (env.app.geotag) env.location else null
        val request = StillRequest(plan.settings, orientation, env.app.jpegQuality, location)
        val baseName = "TRC_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date(timeMs))
        val frames = plan.settings.frames.coerceAtLeast(1)

        if (frames == 1) {
            onProgress(Progress.Capturing(1, 1, plan.settings.exposureNs))
            val f = camera.capture(request)
            onProgress(Progress.Saving)
            val uri = g.photos.save(f.jpeg, "$baseName.jpg", env.app.saveToGallery)
            val (w, h) = ImageIo.jpegSize(f.jpeg)
            val e = entity(sampleId, timeMs, Roles.SINGLE, null, -1, 1, plan, uri, w, h, orientation, f, location, env.app.datasetMode)
            g.captures.insert(e)
            if (env.app.datasetMode) {
                val jpeg = f.jpeg
                g.scope.launch {
                    writeSample(
                        id = sampleId, timeMs = timeMs, plan = plan, env = env, frame = f, photoRole = Roles.SINGLE,
                        fileName = "$baseName.jpg", series = null, width = w, height = h, orientation = orientation,
                        location = location, image = { jpeg.inputStream() },
                    )
                }
            }
            return e
        }

        // ---- series
        val seriesId = Ids.newSeriesId(timeMs)
        val dir = g.paths.seriesDir(seriesId).apply { mkdirs() }
        val records = ArrayList<FrameRecord>(frames)
        var settings = plan.settings
        for (i in 0 until frames) {
            onProgress(Progress.Capturing(i + 1, frames, settings.exposureNs))
            val frameTime = System.currentTimeMillis()
            val f = camera.capture(request.copy(settings = settings))
            val file = File(dir, "frame_%02d.jpg".format(i))
            file.writeBytes(f.jpeg)
            val m = MetaBuilder.exposureMotion(g.sensors, f.exposureStartRt, f.exposureEndRt, env.caps.focalLengthPx)
            records += FrameRecord(i, Ids.newSampleId(frameTime), frameTime, file, f, settings, m)
            if (i == 0 && frames >= 3) settings = adaptAfterFirstFrame(settings, plan.profile, file, env.caps)
        }

        val usable = records.filter { it.settings == settings }
        onProgress(Progress.Merging(0f))
        val resultFile = File(dir, "result.jpg")
        val stack = StackProcessor().process(
            frames = usable.map { StackFrame(it.file, it.index, it.frame.meta.exposureNs, it.frame.meta.iso, it.motion.rms) },
            options = StackOptions(
                mode = if (plan.settings.stackMode == StackMode.NONE) StackMode.NONE else plan.settings.stackMode,
                align = plan.profile.align,
                rejectOutlierFrames = plan.profile.rejectOutlierFrames,
                jpegQuality = env.app.jpegQuality,
            ),
            output = resultFile,
            onProgress = { onProgress(Progress.Merging(it)) },
        )
        onProgress(Progress.Saving)
        val uri = g.photos.save(resultFile, "${baseName}_S$frames.jpg", env.app.saveToGallery)
        val (rw, rh) = BitmapFactory.Options().apply { inJustDecodeBounds = true }.also { BitmapFactory.decodeFile(resultFile.path, it) }
            .let { it.outWidth to it.outHeight }

        val reports = stack.frames.associateBy { it.index }
        val frameInfos = records.map { r ->
            val rep = reports[r.index]
            SeriesFrameInfo(
                index = r.index, exposureTimeNs = r.frame.meta.exposureNs, iso = r.frame.meta.iso,
                accepted = rep?.accepted ?: false,
                rejectReason = rep?.rejectReason ?: if (r.settings != settings) "metering" else null,
                offsetPx = rep?.let { listOf(it.dx, it.dy) }, motionRms = r.motion.rms,
                sharpness = rep?.sharpness, meanLuma = rep?.meanLuma,
            )
        }

        val ds = env.app.datasetMode
        for (r in records) {
            val (w, h) = ImageIo.jpegSize(r.frame.jpeg)
            g.captures.insert(
                entity(r.id, r.timeMs, Roles.FRAME, seriesId, r.index, frames, plan, Uri.fromFile(r.file), w, h, orientation, r.frame, location, ds),
            )
        }
        val last = records.last()
        val result = entity(sampleId, timeMs, Roles.RESULT, seriesId, -1, frames, plan, uri, rw, rh, orientation, last.frame, location, ds)
        g.captures.insert(result)
        resultFile.delete()

        if (ds) g.scope.launch {
            for (r in records) {
                val (w, h) = ImageIo.jpegSize(r.frame.jpeg)
                writeSample(
                    id = r.id, timeMs = r.timeMs, plan = plan.copy(settings = r.settings), env = env, frame = r.frame,
                    photoRole = Roles.FRAME, fileName = r.file.name,
                    series = SeriesInfo(seriesId, "frame", r.index, frames, plan.profile.key, plan.settings.stackMode.key),
                    width = w, height = h, orientation = orientation, location = location, image = { r.file.inputStream() },
                )
            }
            writeSample(
                id = sampleId, timeMs = timeMs, plan = plan.copy(settings = settings), env = env, frame = last.frame,
                photoRole = Roles.RESULT, fileName = "${baseName}_S$frames.jpg",
                series = SeriesInfo(seriesId, "result", -1, frames, plan.profile.key, plan.settings.stackMode.key, frameInfos),
                width = rw, height = rh, orientation = orientation, location = location,
                image = { g.photos.open(uri) },
                exposureStartRt = records.first().frame.exposureStartRt,
            )
        }
        return result
    }

    /** Frame 0 of a sky/moon series meters the highlights; the remaining frames use the corrected exposure. */
    private fun adaptAfterFirstFrame(s: ShotSettings, profile: Profile, file: File, caps: CameraCaps): ShotSettings {
        if (!s.manualExposure || (profile != Profile.LUNAR && profile != Profile.SKY)) return s
        val bmp = BitmapFactory.decodeFile(file.path, BitmapFactory.Options().apply { inSampleSize = 8 }) ?: return s
        val px = ImageIo.toArgb(bmp)
        bmp.recycle()
        var clipped = 0
        var bright = 0
        for (p in px) {
            val l = SceneFeatures.luma(p)
            if (l >= 0.98f) clipped++
            if (l >= 0.45f) bright++
        }
        val exp = s.exposureNs!!
        val range = caps.exposureRange
        return when {
            clipped > px.size * 0.002 -> s.copy(exposureNs = max(exp / 2, range?.lower ?: 1L))
            profile == Profile.LUNAR && bright == 0 -> s.copy(exposureNs = min(exp * 2, range?.upper ?: exp * 2))
            else -> s
        }
    }

    private fun entity(
        id: String, timeMs: Long, role: String, seriesId: String?, index: Int, count: Int, plan: Plan, uri: Uri,
        w: Int, h: Int, orientation: Int, f: CapturedFrame, location: Location?, hasMeta: Boolean,
    ) = CaptureEntity(
        id = id, createdAt = timeMs, role = role, seriesId = seriesId, frameIndex = index, seriesCount = count,
        mode = plan.mode.key, profile = plan.profile.key, imageUri = uri.toString(), width = w, height = h,
        orientation = orientation, exposureNs = f.meta.exposureNs, iso = f.meta.iso, focusDiopters = f.meta.focusDiopters,
        lat = location?.latitude, lon = location?.longitude, hasMeta = hasMeta,
        categoryId = plan.categoryId, aiCategoryId = plan.ai?.chosenClass,
    )

    private suspend fun writeSample(
        id: String, timeMs: Long, plan: Plan, env: Env, frame: CapturedFrame, photoRole: String, fileName: String,
        series: SeriesInfo?, width: Int, height: Int, orientation: Int, location: Location?,
        image: () -> java.io.InputStream?, exposureStartRt: Long = frame.exposureStartRt,
    ) {
        try {
            // Wait for the sensor samples after the exposure.
            val waitNs = frame.exposureEndRt + MetaBuilder.WINDOW_AFTER_MS * 1_000_000L - SystemClock.elapsedRealtimeNanos()
            if (waitNs > 0) delay(waitNs / 1_000_000L + 1)
            val meta = SampleMeta(
                sampleId = id, createdUtcMs = timeMs, tzOffsetMin = MetaBuilder.tzOffsetMin(timeMs),
                localTime = MetaBuilder.localTime(timeMs), app = MetaBuilder.app(), device = MetaBuilder.device(env.caps),
                mode = plan.mode.key,
                photo = PhotoInfo(photoRole, width, height, orientation, env.app.jpegQuality, fileName),
                capture = MetaBuilder.capture(frame.meta),
                userSettings = MetaBuilder.userSettings(plan.settings, plan.timerS, plan.categoryId),
                series = series,
                location = MetaBuilder.location(location),
                orientation = MetaBuilder.orientation(env.motion),
                motion = MetaBuilder.motion(g.sensors, exposureStartRt, frame.exposureEndRt, env.caps.focalLengthPx),
                environment = MetaBuilder.environment(g.context, env.motion, location, timeMs),
                preview = MetaBuilder.preview(env.preview, env.scene?.w ?: 0, env.scene?.h ?: 0),
                features = env.features,
                ai = plan.ai,
            )
            g.dataset.writePending(meta, image, env.scene, env.app.datasetImageMaxSide)
            // The user may have rated already while the sample was being written.
            g.captures.rate(id) { it }
        } catch (t: Throwable) {
            Log.e("ShotRunner", "dataset sample $id failed", t)
        }
    }
}
