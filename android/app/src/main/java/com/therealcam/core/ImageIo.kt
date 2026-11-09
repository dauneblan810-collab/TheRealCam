package com.therealcam.core

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import kotlin.math.max
import kotlin.math.roundToInt

object ImageIo {
    fun toArgb(b: Bitmap): IntArray = IntArray(b.width * b.height).also { b.getPixels(it, 0, b.width, 0, 0, b.width, b.height) }

    fun fromArgb(px: IntArray, w: Int, h: Int): Bitmap = Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)

    fun writeJpeg(b: Bitmap, file: File, quality: Int = 92) {
        file.parentFile?.mkdirs()
        FileOutputStream(file).use { b.compress(Bitmap.CompressFormat.JPEG, quality, it) }
    }

    fun writePng(b: Bitmap, file: File) {
        file.parentFile?.mkdirs()
        FileOutputStream(file).use { b.compress(Bitmap.CompressFormat.PNG, 100, it) }
    }

    fun exifRotation(open: () -> InputStream): Int = runCatching {
        open().use { ExifInterface(it).rotationDegrees }
    }.getOrDefault(0)

    /** Decode with EXIF rotation applied and the long side scaled down to at most [maxSide]. */
    fun decodeUpright(open: () -> InputStream, maxSide: Int): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        open().use { BitmapFactory.decodeStream(it, null, bounds) }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxSide) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        val raw = open().use { BitmapFactory.decodeStream(it, null, opts) } ?: return null
        val rot = exifRotation(open)
        val scale = minOf(1f, maxSide.toFloat() / max(raw.width, raw.height))
        if (rot == 0 && scale >= 0.999f) return raw
        val m = Matrix().apply {
            postScale(scale, scale)
            postRotate(rot.toFloat())
        }
        val out = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
        if (out !== raw) raw.recycle()
        return out
    }

    fun decodeUpright(bytes: ByteArray, maxSide: Int): Bitmap? = decodeUpright({ ByteArrayInputStream(bytes) }, maxSide)
    fun decodeUpright(file: File, maxSide: Int): Bitmap? = decodeUpright({ file.inputStream() }, maxSide)

    fun jpegSize(bytes: ByteArray): Pair<Int, Int> {
        val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, o)
        return o.outWidth to o.outHeight
    }

    fun scaleToFit(b: Bitmap, maxSide: Int): Bitmap {
        val s = maxSide.toFloat() / max(b.width, b.height)
        if (s >= 1f) return b
        return Bitmap.createScaledBitmap(b, (b.width * s).roundToInt().coerceAtLeast(1), (b.height * s).roundToInt().coerceAtLeast(1), true)
    }
}
