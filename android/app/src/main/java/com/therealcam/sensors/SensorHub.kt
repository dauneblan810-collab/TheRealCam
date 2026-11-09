package com.therealcam.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.sqrt

/** Orientation and motion summary, refreshed ~10 times per second. */
data class MotionState(
    /** Elevation of the back camera axis above the horizon, -90..90. */
    val pitchDeg: Double = Double.NaN,
    /** Rotation around the camera axis, -180..180; 0 = upright portrait, +90 = rotated counter-clockwise. */
    val rollDeg: Double = Double.NaN,
    val azimuthDeg: Double = Double.NaN,
    /** OrientationEventListener convention: 0, 90 (rotated clockwise), 180, 270. */
    val deviceOrientation: Int = 0,
    /** RMS angular speed over the last second, rad/s. */
    val gyroRms: Double = Double.NaN,
    val steady: Boolean = false,
    val lux: Double? = null,
    val pressureHpa: Double? = null,
    val hasGyro: Boolean = false,
)

/** Sensor samples copied out of the ring buffer. Times are elapsedRealtimeNanos. */
class SampleWindow(val t: LongArray, val x: FloatArray, val y: FloatArray, val z: FloatArray) {
    val size get() = t.size
}

private class Ring(private val capacity: Int) {
    private val t = LongArray(capacity)
    private val x = FloatArray(capacity)
    private val y = FloatArray(capacity)
    private val z = FloatArray(capacity)
    private var head = 0
    private var count = 0

    @Synchronized
    fun add(ts: Long, vx: Float, vy: Float, vz: Float) {
        t[head] = ts; x[head] = vx; y[head] = vy; z[head] = vz
        head = (head + 1) % capacity
        if (count < capacity) count++
    }

    @Synchronized
    fun window(fromNs: Long, toNs: Long): SampleWindow {
        val idx = ArrayList<Int>()
        val start = (head - count + capacity) % capacity
        for (i in 0 until count) {
            val k = (start + i) % capacity
            if (t[k] in fromNs..toNs) idx.add(k)
        }
        return SampleWindow(
            LongArray(idx.size) { t[idx[it]] },
            FloatArray(idx.size) { x[idx[it]] },
            FloatArray(idx.size) { y[idx[it]] },
            FloatArray(idx.size) { z[idx[it]] },
        )
    }
}

/**
 * Collects gyroscope / accelerometer history and the current orientation while the camera is open.
 * Reference counted: [acquire] / [release].
 */
class SensorHub(context: Context) : SensorEventListener {
    private val sm = context.getSystemService(SensorManager::class.java)
    private val gyro: Sensor? = sm?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val accel: Sensor? = sm?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gravity: Sensor? = sm?.getDefaultSensor(Sensor.TYPE_GRAVITY)
    private val rotation: Sensor? = sm?.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        ?: sm?.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
    private val light: Sensor? = sm?.getDefaultSensor(Sensor.TYPE_LIGHT)
    private val pressure: Sensor? = sm?.getDefaultSensor(Sensor.TYPE_PRESSURE)

    private val gyroRing = Ring(8192)
    private val accelRing = Ring(8192)

    private var thread: HandlerThread? = null
    private var users = 0

    /** Up vector in device coordinates (normalized accelerometer at rest). */
    private val up = FloatArray(3)
    private var hasUp = false
    private var azimuth = Double.NaN
    private var lux: Double? = null
    private var pressureHpa: Double? = null
    private var deviceOrientation = 0
    private var lastPublishNs = 0L

    /** Offset to convert sensor timestamps to elapsedRealtimeNanos (0 on most devices). */
    private var tsOffset: Long? = null

    private val _state = MutableStateFlow(MotionState(hasGyro = gyro != null))
    val state: StateFlow<MotionState> = _state.asStateFlow()

    val hasGyro: Boolean get() = gyro != null

    @Synchronized
    fun acquire() {
        if (users++ > 0 || sm == null) return
        val t = HandlerThread("sensors").also { it.start() }
        thread = t
        val h = Handler(t.looper)
        val rate = SensorManager.SENSOR_DELAY_GAME
        listOfNotNull(gyro, accel, gravity, rotation).forEach { sm.registerListener(this, it, rate, h) }
        listOfNotNull(light, pressure).forEach { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL, h) }
    }

    @Synchronized
    fun release() {
        if (users == 0 || --users > 0) return
        sm?.unregisterListener(this)
        thread?.quitSafely()
        thread = null
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    private fun toRealtime(ts: Long): Long {
        val off = tsOffset ?: run {
            val now = SystemClock.elapsedRealtimeNanos()
            // Sensor timestamps are elapsedRealtime-based on almost all devices; detect the rare monotonic ones.
            val o = if (abs(now - ts) < 2_000_000_000L) 0L else now - System.nanoTime()
            tsOffset = o
            o
        }
        return ts + off
    }

    override fun onSensorChanged(e: SensorEvent) {
        val ts = toRealtime(e.timestamp)
        when (e.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> gyroRing.add(ts, e.values[0], e.values[1], e.values[2])
            Sensor.TYPE_ACCELEROMETER -> {
                accelRing.add(ts, e.values[0], e.values[1], e.values[2])
                if (gravity == null) updateUp(e.values, lowPass = true)
            }
            Sensor.TYPE_GRAVITY -> updateUp(e.values, lowPass = false)
            Sensor.TYPE_ROTATION_VECTOR, Sensor.TYPE_GAME_ROTATION_VECTOR -> {
                val r = FloatArray(9)
                SensorManager.getRotationMatrixFromVector(r, e.values)
                // Heading of the back camera axis (-Z): remap so the camera axis becomes the reference.
                val remapped = FloatArray(9)
                SensorManager.remapCoordinateSystem(r, SensorManager.AXIS_X, SensorManager.AXIS_Z, remapped)
                val o = FloatArray(3)
                SensorManager.getOrientation(remapped, o)
                azimuth = ((Math.toDegrees(o[0].toDouble()) + 360.0) % 360.0)
            }
            Sensor.TYPE_LIGHT -> lux = e.values[0].toDouble()
            Sensor.TYPE_PRESSURE -> pressureHpa = e.values[0].toDouble()
        }
        if (ts - lastPublishNs > 100_000_000L) {
            lastPublishNs = ts
            publish()
        }
    }

    private fun updateUp(v: FloatArray, lowPass: Boolean) {
        val n = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        if (n < 1e-3f) return
        val a = if (lowPass && hasUp) 0.15f else 1f
        for (i in 0..2) up[i] = up[i] + a * (v[i] / n - up[i])
        hasUp = true
    }

    private fun publish() {
        val pitch: Double
        val roll: Double
        if (hasUp) {
            val n = sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).toDouble()
            pitch = Math.toDegrees(asin((-up[2] / n).coerceIn(-1.0, 1.0)))
            roll = Math.toDegrees(atan2(up[0].toDouble(), up[1].toDouble()))
            // Keep the previous orientation when the phone is nearly flat or between sectors.
            if (abs(up[2] / n) < 0.8) {
                val cw = ((-roll) % 360 + 360) % 360
                val candidate = (((cw + 45) / 90).toInt() % 4) * 90
                val dist = abs(((cw - candidate + 540) % 360) - 180)
                if (candidate != deviceOrientation && dist < 30) deviceOrientation = candidate
            }
        } else {
            pitch = Double.NaN
            roll = Double.NaN
        }
        val rms = gyroRms(1_000)
        _state.value = MotionState(
            pitchDeg = pitch,
            rollDeg = roll,
            azimuthDeg = azimuth,
            deviceOrientation = deviceOrientation,
            gyroRms = rms,
            steady = rms.isFinite() && rms < STEADY_RAD_S,
            lux = lux,
            pressureHpa = pressureHpa,
            hasGyro = gyro != null,
        )
    }

    fun gyroWindow(fromNs: Long, toNs: Long): SampleWindow = gyroRing.window(fromNs, toNs)
    fun accelWindow(fromNs: Long, toNs: Long): SampleWindow = accelRing.window(fromNs, toNs)

    /** RMS of |ω| over the last [ms] milliseconds, NaN without gyroscope data. */
    fun gyroRms(ms: Long): Double {
        val now = SystemClock.elapsedRealtimeNanos()
        return rms(gyroWindow(now - ms * 1_000_000L, now))
    }

    companion object {
        /** Below this RMS angular speed the phone is considered fixed (tripod / leaning on something). */
        const val STEADY_RAD_S = 0.012

        fun rms(w: SampleWindow): Double {
            if (w.size == 0) return Double.NaN
            var s = 0.0
            for (i in 0 until w.size) s += (w.x[i] * w.x[i] + w.y[i] * w.y[i] + w.z[i] * w.z[i]).toDouble()
            return sqrt(s / w.size)
        }
    }
}
