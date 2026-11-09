package com.therealcam.ai

import android.graphics.Bitmap
import android.view.TextureView
import com.therealcam.core.ImageIo

/** Upright preview of the scene, 192×256 or 256×192 (see SceneFeatures). */
class ScenePreview(val px: IntArray, val w: Int, val h: Int) {
    val thumb: IntArray by lazy { SceneFeatures.thumb64(px, w, h) }

    fun bitmap(): Bitmap = ImageIo.fromArgb(px, w, h)
    fun thumbBitmap(): Bitmap = ImageIo.fromArgb(thumb, SceneFeatures.THUMB, SceneFeatures.THUMB)

    companion object {
        private const val GRAB_W = SceneFeatures.PREVIEW_SHORT * 3
        private const val GRAB_H = SceneFeatures.PREVIEW_LONG * 3

        /**
         * Grabs the live preview (portrait, 3:4) at 3× the target size and box-filters it, so that
         * point sources such as stars survive the downscale. Must be called on the main thread.
         */
        fun grab(view: TextureView, deviceOrientation: Int): ScenePreview? {
            if (!view.isAvailable) return null
            val bmp = runCatching { view.getBitmap(GRAB_W, GRAB_H) }.getOrNull() ?: return null
            val px = ImageIo.toArgb(bmp)
            bmp.recycle()
            val small = SceneFeatures.boxDownscale(px, GRAB_W, GRAB_H, 3)
            val (r, w, h) = SceneFeatures.rotateCw(small, SceneFeatures.PREVIEW_SHORT, SceneFeatures.PREVIEW_LONG, deviceOrientation)
            return ScenePreview(r, w, h)
        }
    }
}
