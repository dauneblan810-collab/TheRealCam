package com.therealcam.dataset

import kotlinx.serialization.Serializable

/**
 * `meta.json` of a dataset sample, schema `therealcam.sample/1` (docs/DATASET_FORMAT.md).
 * Property names are converted to snake_case by [com.therealcam.core.SnakeJson].
 */
@Serializable
data class SampleMeta(
    val schema: String = SCHEMA,
    val sampleId: String,
    val createdUtcMs: Long,
    val tzOffsetMin: Int,
    val localTime: String,
    val app: AppInfo,
    val device: DeviceInfo?,
    val mode: String,
    val photo: PhotoInfo,
    val capture: CaptureInfo?,
    val userSettings: UserSettingsInfo?,
    val series: SeriesInfo?,
    val location: LocationInfo?,
    val orientation: OrientationInfo?,
    val motion: MotionInfo?,
    val environment: EnvironmentInfo?,
    val preview: PreviewInfo?,
    val features: Map<String, Double?>,
    val ai: AiInfo?,
    val labels: Labels = Labels(),
) {
    companion object {
        const val SCHEMA = "therealcam.sample/1"
    }
}

@Serializable
data class AppInfo(val version: String, val versionCode: Int)

@Serializable
data class DeviceInfo(
    val manufacturer: String,
    val model: String,
    val sdkInt: Int,
    val cameraId: String,
    val lensFacing: String,
    val hardwareLevel: String,
    val manualSensor: Boolean,
    val sensorOrientation: Int,
    val physicalSizeMm: List<Float>?,
    val pixelArray: List<Int>?,
    val focalLengthsMm: List<Float>?,
    val apertures: List<Float>?,
    val isoRange: List<Int>?,
    val maxAnalogIso: Int?,
    val exposureRangeNs: List<Long>?,
    val minFocusDistanceDiopters: Float?,
)

@Serializable
data class PhotoInfo(
    val role: String,
    val width: Int,
    val height: Int,
    val orientationDeg: Int,
    val jpegQuality: Int,
    val fileName: String?,
)

@Serializable
data class CaptureInfo(
    val exposureTimeNs: Long?,
    val iso: Int?,
    val aperture: Float?,
    val focalLengthMm: Float?,
    val focusDistanceDiopters: Float?,
    val aeMode: String?,
    val afMode: String?,
    val awbMode: String?,
    val wbGains: List<Float>?,
    val evCompensation: Float?,
    val flash: String?,
    val ois: String?,
    val zoomRatio: Float?,
    val frameDurationNs: Long?,
    val noiseReduction: String?,
    val sensorTimestampNs: Long?,
)

@Serializable
data class UserSettingsInfo(
    val iso: Int?,
    val exposureTimeNs: Long?,
    val focus: String,
    val focusDistanceDiopters: Float?,
    val wb: String,
    val ev: Float,
    val frames: Int,
    val stackMode: String,
    val timerS: Int,
    val categoryHint: String?,
)

@Serializable
data class SeriesFrameInfo(
    val index: Int,
    val exposureTimeNs: Long?,
    val iso: Int?,
    val accepted: Boolean,
    val rejectReason: String?,
    val offsetPx: List<Int>?,
    val motionRms: Double?,
    val sharpness: Double?,
    val meanLuma: Double?,
)

@Serializable
data class SeriesInfo(
    val seriesId: String,
    val role: String,
    val index: Int,
    val count: Int,
    val profile: String,
    val stackMode: String,
    val frames: List<SeriesFrameInfo>? = null,
)

@Serializable
data class LocationInfo(
    val lat: Double,
    val lon: Double,
    val altM: Double?,
    val accuracyM: Float?,
    val provider: String?,
    val ageMs: Long?,
    val speedMps: Float?,
    val bearingDeg: Float?,
)

@Serializable
data class OrientationInfo(
    val pitchDeg: Double?,
    val rollDeg: Double?,
    val azimuthDeg: Double?,
    val deviceRotationDeg: Int,
)

@Serializable
data class MotionStats(
    val preGyroRms: Double?,
    val exposureGyroRms: Double?,
    val exposureGyroMax: Double?,
    val exposureRotationRad: Double?,
    val estBlurPx: Double?,
)

@Serializable
data class MotionInfo(
    val windowMs: List<Double>,
    val exposureMs: List<Double>,
    val gyro: List<List<Float>>,
    val accel: List<List<Float>>,
    val stats: MotionStats,
)

@Serializable
data class EnvironmentInfo(
    val lux: Double?,
    val pressureHpa: Double?,
    val batteryTempC: Double?,
    val sunAltDeg: Double?,
    val sunAzDeg: Double?,
    val moonAltDeg: Double?,
    val moonAzDeg: Double?,
    val moonIllum: Double?,
)

@Serializable
data class PreviewInfo(
    val exposureTimeNs: Long?,
    val iso: Int?,
    val aperture: Float?,
    val aeState: String?,
    val afState: String?,
    val focusDistanceDiopters: Float?,
    val width: Int,
    val height: Int,
)

@Serializable
data class AiInfo(
    val modelName: String,
    val modelSha256: String,
    val probs: Map<String, Double>,
    val chosenClass: String?,
    val userOverride: Boolean,
    val params: Map<String, Double>,
)

@Serializable
data class Labels(
    val rating: Int? = null,
    val exposure: Int? = null,
    val sharpness: Int? = null,
    val noise: Int? = null,
    val color: Int? = null,
    val motionBlur: Int? = null,
    val categoryId: String? = null,
    val categoryName: String? = null,
    val categoryIcon: String? = null,
    val note: String? = null,
    val ratedUtcMs: Long? = null,
    val complete: Boolean = false,
)
