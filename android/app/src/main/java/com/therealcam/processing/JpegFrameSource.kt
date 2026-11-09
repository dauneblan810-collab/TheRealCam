package com.therealcam.processing

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapRegionDecoder
import android.graphics.ColorSpace
import android.graphics.Rect
import android.os.Build
import com.therealcam.processing.core.FrameSource
import com.therealcam.processing.core.Luma8
import com.therealcam.processing.core.analysisSampleSize
import java.io.Closeable
import java.io.File
import java.io.IOException
import java.util.concurrent.atomic.AtomicReferenceArray

/**
 * JPEG frames of a series: downscaled decodes (inSampleSize) for analysis and BitmapRegionDecoder
 * strips at full resolution for merging, so no full-size frame is ever held in memory.
 * Pixels are raw sensor orientation (EXIF orientation is not applied), decoded to sRGB.
 */
internal class JpegFrameSource(private val files: List<File>) : FrameSource, Closeable {
    private val sizes: List<Pair<Int, Int>> = files.map(::boundsOf)
    private val canonical: Pair<Int, Int> = sizes.firstOrNull { it.first > 0 && it.second > 0 } ?: (0 to 0)
    override val width: Int = canonical.first
    override val height: Int = canonical.second
    private val sampleSize = analysisSampleSize(width, height)
    private val decoders = AtomicReferenceArray<BitmapRegionDecoder?>(files.size)
    private val srgb: ColorSpace = ColorSpace.get(ColorSpace.Named.SRGB)
    private var closed = false

    override fun analysisLuma(position: Int): Luma8? {
        if (width == 0 || sizes[position] != canonical) return null
        val opts = BitmapFactory.Options().apply {
            inSampleSize = sampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inPreferredColorSpace = srgb
        }
        val bmp = BitmapFactory.decodeFile(files[position].absolutePath, opts) ?: return null
        try {
            val w = bmp.width
            val h = bmp.height
            val px = IntArray(w * h)
            bmp.getPixels(px, 0, w, 0, 0, w, h)
            return Luma8.fromArgb(w, h, px)
        } finally {
            bmp.recycle()
        }
    }

    override fun readRegion(
        position: Int, left: Int, top: Int, width: Int, height: Int,
        dst: IntArray, dstOffset: Int, dstStride: Int,
    ) {
        val opts = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inPreferredColorSpace = srgb
        }
        val name = files[position].name
        val bmp = decoder(position).decodeRegion(Rect(left, top, left + width, top + height), opts)
            ?: throw IOException("Cannot decode region of $name")
        try {
            if (bmp.width < width || bmp.height < height) {
                throw IOException("Region of $name decoded as ${bmp.width}x${bmp.height}, expected ${width}x$height")
            }
            // getPixels returns sRGB values whatever the decoded color space is.
            bmp.getPixels(dst, dstOffset, dstStride, 0, 0, width, height)
        } finally {
            bmp.recycle()
        }
    }

    private fun decoder(position: Int): BitmapRegionDecoder {
        decoders.get(position)?.let { return it }
        synchronized(this) {
            decoders.get(position)?.let { return it }
            if (closed) throw IOException("Frame source is closed")
            val path = files[position].absolutePath
            val decoder = if (Build.VERSION.SDK_INT >= 31) {
                BitmapRegionDecoder.newInstance(path)
            } else {
                @Suppress("DEPRECATION")
                BitmapRegionDecoder.newInstance(path, false)
            } ?: throw IOException("Cannot open ${files[position].name}")
            decoders.set(position, decoder)
            return decoder
        }
    }

    override fun close() {
        synchronized(this) {
            closed = true
            for (i in 0 until decoders.length()) decoders.getAndSet(i, null)?.recycle()
        }
    }

    private fun boundsOf(file: File): Pair<Int, Int> {
        val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, opts)
        return opts.outWidth to opts.outHeight
    }
}
