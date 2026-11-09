package com.therealcam.processing

import android.util.Log
import androidx.exifinterface.media.ExifInterface
import java.io.File

/** EXIF for stacked outputs. Failures are logged, never fatal: the pixels matter more than metadata. */
internal object StackExif {
    private const val TAG = "StackExif"
    private const val SOFTWARE = "TheRealCam"

    /** Copied from the reference frame. Thumbnails, maker notes, serials and unique ids are not. */
    private val COPY_TAGS = arrayOf(
        ExifInterface.TAG_ORIENTATION,
        ExifInterface.TAG_DATETIME,
        ExifInterface.TAG_DATETIME_ORIGINAL,
        ExifInterface.TAG_DATETIME_DIGITIZED,
        ExifInterface.TAG_OFFSET_TIME,
        ExifInterface.TAG_OFFSET_TIME_ORIGINAL,
        ExifInterface.TAG_OFFSET_TIME_DIGITIZED,
        ExifInterface.TAG_SUBSEC_TIME,
        ExifInterface.TAG_SUBSEC_TIME_ORIGINAL,
        ExifInterface.TAG_SUBSEC_TIME_DIGITIZED,
        ExifInterface.TAG_MAKE,
        ExifInterface.TAG_MODEL,
        ExifInterface.TAG_ARTIST,
        ExifInterface.TAG_COPYRIGHT,
        ExifInterface.TAG_LENS_MAKE,
        ExifInterface.TAG_LENS_MODEL,
        ExifInterface.TAG_LENS_SPECIFICATION,
        ExifInterface.TAG_F_NUMBER,
        ExifInterface.TAG_APERTURE_VALUE,
        ExifInterface.TAG_MAX_APERTURE_VALUE,
        ExifInterface.TAG_EXPOSURE_TIME,
        ExifInterface.TAG_SHUTTER_SPEED_VALUE,
        ExifInterface.TAG_EXPOSURE_PROGRAM,
        ExifInterface.TAG_EXPOSURE_MODE,
        ExifInterface.TAG_EXPOSURE_BIAS_VALUE,
        ExifInterface.TAG_EXPOSURE_INDEX,
        ExifInterface.TAG_PHOTOGRAPHIC_SENSITIVITY,
        ExifInterface.TAG_SENSITIVITY_TYPE,
        ExifInterface.TAG_ISO_SPEED,
        ExifInterface.TAG_RECOMMENDED_EXPOSURE_INDEX,
        ExifInterface.TAG_STANDARD_OUTPUT_SENSITIVITY,
        ExifInterface.TAG_FOCAL_LENGTH,
        ExifInterface.TAG_FOCAL_LENGTH_IN_35MM_FILM,
        ExifInterface.TAG_SUBJECT_DISTANCE,
        ExifInterface.TAG_SUBJECT_DISTANCE_RANGE,
        ExifInterface.TAG_METERING_MODE,
        ExifInterface.TAG_LIGHT_SOURCE,
        ExifInterface.TAG_FLASH,
        ExifInterface.TAG_WHITE_BALANCE,
        ExifInterface.TAG_BRIGHTNESS_VALUE,
        ExifInterface.TAG_DIGITAL_ZOOM_RATIO,
        ExifInterface.TAG_SCENE_CAPTURE_TYPE,
        ExifInterface.TAG_SENSING_METHOD,
        ExifInterface.TAG_CONTRAST,
        ExifInterface.TAG_SATURATION,
        ExifInterface.TAG_SHARPNESS,
        ExifInterface.TAG_GAIN_CONTROL,
        ExifInterface.TAG_CUSTOM_RENDERED,
        ExifInterface.TAG_GPS_VERSION_ID,
        ExifInterface.TAG_GPS_LATITUDE,
        ExifInterface.TAG_GPS_LATITUDE_REF,
        ExifInterface.TAG_GPS_LONGITUDE,
        ExifInterface.TAG_GPS_LONGITUDE_REF,
        ExifInterface.TAG_GPS_ALTITUDE,
        ExifInterface.TAG_GPS_ALTITUDE_REF,
        ExifInterface.TAG_GPS_TIMESTAMP,
        ExifInterface.TAG_GPS_DATESTAMP,
        ExifInterface.TAG_GPS_PROCESSING_METHOD,
        ExifInterface.TAG_GPS_SATELLITES,
        ExifInterface.TAG_GPS_STATUS,
        ExifInterface.TAG_GPS_MEASURE_MODE,
        ExifInterface.TAG_GPS_DOP,
        ExifInterface.TAG_GPS_H_POSITIONING_ERROR,
        ExifInterface.TAG_GPS_SPEED,
        ExifInterface.TAG_GPS_SPEED_REF,
        ExifInterface.TAG_GPS_TRACK,
        ExifInterface.TAG_GPS_TRACK_REF,
        ExifInterface.TAG_GPS_IMG_DIRECTION,
        ExifInterface.TAG_GPS_IMG_DIRECTION_REF,
        ExifInterface.TAG_GPS_MAP_DATUM,
        ExifInterface.TAG_GPS_DIFFERENTIAL,
    )

    /**
     * EXIF of a merged output: reference tags, stack description, and (if known) the summed exposure.
     * The original single-frame ShutterSpeedValue is dropped when the exposure is replaced.
     */
    fun writeMerged(
        reference: File, target: File, description: String,
        width: Int, height: Int, exposureSeconds: Double?,
    ): Boolean = runExif(target) {
        val src = ExifInterface(reference.absolutePath)
        val dst = ExifInterface(target.absolutePath)
        for (tag in COPY_TAGS) {
            if (exposureSeconds != null &&
                (tag == ExifInterface.TAG_EXPOSURE_TIME || tag == ExifInterface.TAG_SHUTTER_SPEED_VALUE)
            ) {
                continue
            }
            src.getAttribute(tag)?.let { dst.setAttribute(tag, it) }
        }
        dst.setAttribute(ExifInterface.TAG_PIXEL_X_DIMENSION, width.toString())
        dst.setAttribute(ExifInterface.TAG_PIXEL_Y_DIMENSION, height.toString())
        dst.setAttribute(ExifInterface.TAG_COLOR_SPACE, ExifInterface.COLOR_SPACE_S_RGB.toString())
        dst.setAttribute(ExifInterface.TAG_SOFTWARE, SOFTWARE)
        // Decimal string: ExifInterface converts it to a rational for this tag.
        exposureSeconds?.let { dst.setAttribute(ExifInterface.TAG_EXPOSURE_TIME, it.toString()) }
        describe(dst, description)
        dst.saveAttributes()
    }

    /** Adds the stack description to a byte copy of a frame (its own EXIF stays intact). */
    fun annotateCopy(target: File, description: String): Boolean = runExif(target) {
        val exif = ExifInterface(target.absolutePath)
        describe(exif, description)
        exif.saveAttributes()
    }

    private fun describe(exif: ExifInterface, description: String) {
        exif.setAttribute(ExifInterface.TAG_IMAGE_DESCRIPTION, description)
        // UserComment starts with an 8-byte character code.
        exif.setAttribute(ExifInterface.TAG_USER_COMMENT, "ASCII\u0000\u0000\u0000$description")
    }

    private inline fun runExif(target: File, block: () -> Unit): Boolean = try {
        block()
        true
    } catch (e: Exception) {
        Log.w(TAG, "EXIF not written to ${target.name}", e)
        false
    }
}
