package com.therealcam.dataset

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.location.Location
import android.os.BatteryManager
import android.os.Build
import android.os.SystemClock
import com.therealcam.BuildConfig
import com.therealcam.camera.CameraCaps
import com.therealcam.camera.FrameMetadata
import com.therealcam.camera.PreviewSnapshot
import com.therealcam.camera.ShotSettings
import com.therealcam.camera.FocusMode
import com.therealcam.core.finiteOrNull
import com.therealcam.sensors.Astro
import com.therealcam.sensors.MotionState
import com.therealcam.sensors.SampleWindow
import com.therealcam.sensors.SensorHub
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.TimeZone
import kotlin.math.max
import kotlin.math.sqrt

/** Builds the pieces of meta.json from camera, sensor and location state. */
object MetaBuilder {
    const val WINDOW_BEFORE_MS = 1500L
    const val WINDOW_AFTER_MS = 750L

    fun app() = AppInfo(BuildConfig.VERSION_NAME, BuildConfig.VERSION_CODE)

    fun tzOffsetMin(timeMs: Long) = TimeZone.getDefault().getOffset(timeMs) / 60_000

    fun localTime(timeMs: Long): String =
        OffsetDateTime.ofInstant(Instant.ofEpochMilli(timeMs), ZoneId.systemDefault()).format(DateTimeFormatter.ISO_OFFSET_DATE_TIME)

    fun device(c: CameraCaps) = DeviceInfo(
        manufacturer = Build.MANUFACTURER,
        model = Build.MODEL,
        sdkInt = Build.VERSION.SDK_INT,
        cameraId = c.id,
        lensFacing = if (c.isFront) "front" else "back",
        hardwareLevel = c.hardwareLevelName,
        manualSensor = c.manualSensor,
        sensorOrientation = c.sensorOrientation,
        physicalSizeMm = c.physicalSize?.let { listOf(it.width, it.height) },
        pixelArray = c.pixelArray?.let { listOf(it.width, it.height) },
        focalLengthsMm = c.focalLengths.toList(),
        apertures = c.apertures.toList(),
        isoRange = c.isoRange?.let { listOf(it.lower, it.upper) },
        maxAnalogIso = c.maxAnalogIso,
        exposureRangeNs = c.exposureRange?.let { listOf(it.lower, it.upper) },
        minFocusDistanceDiopters = c.minFocusDistance,
    )

    fun capture(m: FrameMetadata) = CaptureInfo(
        exposureTimeNs = m.exposureNs, iso = m.iso, aperture = m.aperture, focalLengthMm = m.focalLengthMm,
        focusDistanceDiopters = m.focusDiopters, aeMode = m.aeMode, afMode = m.afMode, awbMode = m.awbMode,
        wbGains = m.wbGains, evCompensation = m.evCompensation, flash = m.flash, ois = m.ois,
        zoomRatio = m.zoomRatio, frameDurationNs = m.frameDurationNs, noiseReduction = m.noiseReduction,
        sensorTimestampNs = m.sensorTimestampNs,
    )

    fun userSettings(s: ShotSettings, timerS: Int, categoryHint: String?) = UserSettingsInfo(
        iso = s.iso, exposureTimeNs = s.exposureNs, focus = s.focus.key,
        focusDistanceDiopters = when (s.focus) {
            FocusMode.INFINITY -> 0f
            FocusMode.MANUAL -> s.focusDiopters
            FocusMode.AUTO -> null
        },
        wb = s.wb.key, ev = s.ev, frames = s.frames, stackMode = s.stackMode.key, timerS = timerS, categoryHint = categoryHint,
    )

    fun location(l: Location?): LocationInfo? = l?.let {
        LocationInfo(
            lat = it.latitude, lon = it.longitude,
            altM = if (it.hasAltitude()) it.altitude else null,
            accuracyM = if (it.hasAccuracy()) it.accuracy else null,
            provider = it.provider,
            ageMs = (SystemClock.elapsedRealtimeNanos() - it.elapsedRealtimeNanos) / 1_000_000L,
            speedMps = if (it.hasSpeed()) it.speed else null,
            bearingDeg = if (it.hasBearing()) it.bearing else null,
        )
    }

    fun orientation(m: MotionState) = OrientationInfo(
        pitchDeg = m.pitchDeg.finiteOrNull(), rollDeg = m.rollDeg.finiteOrNull(),
        azimuthDeg = m.azimuthDeg.finiteOrNull(), deviceRotationDeg = m.deviceOrientation,
    )

    fun preview(p: PreviewSnapshot?, w: Int, h: Int) = PreviewInfo(
        exposureTimeNs = p?.exposureNs, iso = p?.iso, aperture = p?.aperture,
        aeState = p?.aeStateName, afState = p?.afStateName, focusDistanceDiopters = p?.focusDiopters,
        width = w, height = h,
    )

    fun batteryTempC(context: Context): Double? = runCatching {
        val i = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val t = i?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE) ?: Int.MIN_VALUE
        if (t == Int.MIN_VALUE) null else t / 10.0
    }.getOrNull()

    fun environment(context: Context, m: MotionState, l: Location?, timeMs: Long): EnvironmentInfo {
        val sun = l?.let { Astro.sun(timeMs, it.latitude, it.longitude) }
        val moon = l?.let { Astro.moon(timeMs, it.latitude, it.longitude) }
        return EnvironmentInfo(
            lux = m.lux, pressureHpa = m.pressureHpa, batteryTempC = batteryTempC(context),
            sunAltDeg = sun?.altDeg, sunAzDeg = sun?.azDeg, moonAltDeg = moon?.altDeg, moonAzDeg = moon?.azDeg,
            moonIllum = Astro.moonIllumination(timeMs),
        )
    }

    /** Gyro statistics for one exposure, also used to reject shaken frames. */
    data class ExposureMotion(val rms: Double?, val max: Double?, val rotationRad: Double?, val blurPx: Double?)

    fun exposureMotion(sensors: SensorHub, startRt: Long, endRt: Long, focalPx: Double): ExposureMotion {
        // Pad short exposures so at least a couple of samples fall inside.
        val pad = max(0L, 20_000_000L - (endRt - startRt)) / 2
        val w = sensors.gyroWindow(startRt - pad, endRt + pad)
        if (w.size == 0) return ExposureMotion(null, null, null, null)
        var s2 = 0.0
        var mx = 0.0
        var rot = 0.0
        var rotPerp = 0.0
        for (i in 0 until w.size) {
            val m2 = (w.x[i] * w.x[i] + w.y[i] * w.y[i] + w.z[i] * w.z[i]).toDouble()
            s2 += m2
            mx = max(mx, sqrt(m2))
            if (i > 0) {
                val dt = (w.t[i] - w.t[i - 1]) / 1e9
                rot += sqrt(m2) * dt
                rotPerp += sqrt((w.x[i] * w.x[i] + w.y[i] * w.y[i]).toDouble()) * dt
            }
        }
        if (w.size == 1) {
            val dt = (endRt - startRt) / 1e9
            rot = sqrt(s2) * dt
            rotPerp = sqrt((w.x[0] * w.x[0] + w.y[0] * w.y[0]).toDouble()) * dt
        }
        return ExposureMotion(sqrt(s2 / w.size), mx, rot, if (focalPx.isFinite()) rotPerp * focalPx else null)
    }

    fun motion(sensors: SensorHub, startRt: Long, endRt: Long, focalPx: Double): MotionInfo {
        val from = startRt - WINDOW_BEFORE_MS * 1_000_000L
        val to = endRt + WINDOW_AFTER_MS * 1_000_000L
        val g = sensors.gyroWindow(from, to)
        val a = sensors.accelWindow(from, to)
        val pre = SensorHub.rms(sensors.gyroWindow(startRt - 1_000_000_000L, startRt))
        val em = exposureMotion(sensors, startRt, endRt, focalPx)
        fun rows(w: SampleWindow) = List(w.size) { i ->
            listOf(((w.t[i] - startRt) / 1e6).toFloat(), w.x[i], w.y[i], w.z[i])
        }
        return MotionInfo(
            windowMs = listOf(-WINDOW_BEFORE_MS.toDouble(), (to - startRt) / 1e6),
            exposureMs = listOf(0.0, (endRt - startRt) / 1e6),
            gyro = rows(g),
            accel = rows(a),
            stats = MotionStats(
                preGyroRms = pre.finiteOrNull(), exposureGyroRms = em.rms, exposureGyroMax = em.max,
                exposureRotationRad = em.rotationRad, estBlurPx = em.blurPx,
            ),
        )
    }
}
