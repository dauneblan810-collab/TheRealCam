package com.therealcam.camera

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.util.Range
import android.util.Rational
import android.util.Size
import android.util.SizeF
import kotlin.math.abs

/** Everything the app needs to know about one camera, read once from CameraCharacteristics. */
data class CameraCaps(
    val id: String,
    val facing: Int,
    val hardwareLevel: Int,
    val manualSensor: Boolean,
    val sensorOrientation: Int,
    val isoRange: Range<Int>?,
    val exposureRange: Range<Long>?,
    val maxAnalogIso: Int?,
    /** Diopters; 0 means fixed focus. */
    val minFocusDistance: Float,
    val focalLengths: FloatArray,
    val apertures: FloatArray,
    val physicalSize: SizeF?,
    val pixelArray: Size?,
    val activeArray: Rect?,
    val aeCompRange: Range<Int>,
    val aeCompStep: Rational,
    val flash: Boolean,
    val awbModes: IntArray,
    val afModes: IntArray,
    val maxAfRegions: Int,
    val maxAeRegions: Int,
    val jpegSize: Size,
    val previewSize: Size,
    val jpegMinFrameNs: Long,
    val timestampRealtime: Boolean,
    val noiseReductionModes: IntArray,
) {
    val isFront get() = facing == CameraMetadata.LENS_FACING_FRONT
    val canManualFocus get() = manualSensor && minFocusDistance > 0f
    val hasAutoFocus get() = afModes.any { it != CameraMetadata.CONTROL_AF_MODE_OFF }

    val hardwareLevelName: String
        get() = when (hardwareLevel) {
            CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> "legacy"
            CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "limited"
            CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "full"
            CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "level_3"
            CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL -> "external"
            else -> "unknown"
        }

    /** Horizontal field of view of the long sensor side, degrees (NaN if unknown). */
    val focalLengthPx: Double
        get() {
            val f = focalLengths.firstOrNull() ?: return Double.NaN
            val w = physicalSize?.width ?: return Double.NaN
            return (f / w * jpegSize.width).toDouble()
        }

    companion object {
        fun read(manager: CameraManager, id: String): CameraCaps {
            val c = manager.getCameraCharacteristics(id)
            val capabilities = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: IntArray(0)
            val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                ?: error("Camera $id has no stream configuration")
            val jpegSizes = map.getOutputSizes(ImageFormat.JPEG)?.toList().orEmpty()
            val jpeg = chooseJpegSize(jpegSizes)
            val previewSizes = map.getOutputSizes(SurfaceTexture::class.java)?.toList().orEmpty()
            val preview = choosePreviewSize(previewSizes, jpeg)
            return CameraCaps(
                id = id,
                facing = c.get(CameraCharacteristics.LENS_FACING) ?: CameraMetadata.LENS_FACING_BACK,
                hardwareLevel = c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
                    ?: CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY,
                manualSensor = CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR in capabilities,
                sensorOrientation = c.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90,
                isoRange = c.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE),
                exposureRange = c.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE),
                maxAnalogIso = c.get(CameraCharacteristics.SENSOR_MAX_ANALOG_SENSITIVITY),
                minFocusDistance = c.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f,
                focalLengths = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS) ?: FloatArray(0),
                apertures = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES) ?: FloatArray(0),
                physicalSize = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE),
                pixelArray = c.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE),
                activeArray = c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE),
                aeCompRange = c.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE) ?: Range(0, 0),
                aeCompStep = c.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_STEP) ?: Rational(1, 3),
                flash = c.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true,
                awbModes = c.get(CameraCharacteristics.CONTROL_AWB_AVAILABLE_MODES) ?: IntArray(0),
                afModes = c.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES) ?: IntArray(0),
                maxAfRegions = c.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF) ?: 0,
                maxAeRegions = c.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AE) ?: 0,
                jpegSize = jpeg,
                previewSize = preview,
                jpegMinFrameNs = runCatching { map.getOutputMinFrameDuration(ImageFormat.JPEG, jpeg) }.getOrDefault(0L),
                timestampRealtime = c.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) ==
                    CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME,
                noiseReductionModes = c.get(CameraCharacteristics.NOISE_REDUCTION_AVAILABLE_NOISE_REDUCTION_MODES) ?: IntArray(0),
            )
        }

        private fun aspect(s: Size) = s.width.toDouble() / s.height

        /** Largest 4:3 JPEG size (most sensors are 4:3), else the largest overall. */
        fun chooseJpegSize(sizes: List<Size>): Size {
            require(sizes.isNotEmpty()) { "No JPEG sizes" }
            val fourThree = sizes.filter { abs(aspect(it) - 4.0 / 3.0) < 0.02 }
            return (fourThree.ifEmpty { sizes }).maxBy { it.width.toLong() * it.height }
        }

        /** Largest preview size with the JPEG aspect ratio that is not bigger than 1920×1440. */
        fun choosePreviewSize(sizes: List<Size>, jpeg: Size): Size {
            require(sizes.isNotEmpty()) { "No preview sizes" }
            val target = aspect(jpeg)
            val fitting = sizes.filter { it.width <= 1920 && it.height <= 1440 }
            val sameAspect = fitting.filter { abs(aspect(it) - target) < 0.02 }
            return (sameAspect.ifEmpty { fitting.ifEmpty { sizes } }).maxBy { it.width.toLong() * it.height }
        }

        /** Primary back and front cameras (first of each facing as listed by the HAL). */
        fun primaryIds(manager: CameraManager): Pair<String?, String?> {
            var back: String? = null
            var front: String? = null
            for (id in manager.cameraIdList) {
                val facing = runCatching { manager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING) }.getOrNull()
                if (facing == CameraMetadata.LENS_FACING_BACK && back == null) back = id
                if (facing == CameraMetadata.LENS_FACING_FRONT && front == null) front = id
            }
            return back to front
        }
    }
}
