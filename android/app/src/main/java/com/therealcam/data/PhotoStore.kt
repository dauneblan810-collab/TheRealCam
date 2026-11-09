package com.therealcam.data

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.core.content.ContextCompat
import com.therealcam.core.AppPaths
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.io.InputStream

/**
 * Saves final photos. With "save to gallery" on they go to DCIM/TheRealCam (visible to every
 * gallery and kept after uninstall); otherwise to the app folder. Series frames never go here.
 */
class PhotoStore(private val context: Context, private val paths: AppPaths) {

    suspend fun save(jpeg: File, displayName: String, toGallery: Boolean): Uri = withContext(Dispatchers.IO) {
        if (toGallery) {
            runCatching { saveToGallery(displayName) { out -> jpeg.inputStream().use { it.copyTo(out) } } }
                .getOrNull()?.let { return@withContext it }
        }
        val dst = File(paths.photos, displayName)
        jpeg.copyTo(dst, overwrite = true)
        Uri.fromFile(dst)
    }

    suspend fun save(jpeg: ByteArray, displayName: String, toGallery: Boolean): Uri = withContext(Dispatchers.IO) {
        if (toGallery) {
            runCatching { saveToGallery(displayName) { it.write(jpeg) } }.getOrNull()?.let { return@withContext it }
        }
        val dst = File(paths.photos, displayName)
        dst.writeBytes(jpeg)
        Uri.fromFile(dst)
    }

    private fun saveToGallery(displayName: String, write: (java.io.OutputStream) -> Unit): Uri {
        if (Build.VERSION.SDK_INT >= 29) {
            val resolver = context.contentResolver
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_DCIM}/TheRealCam")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
            val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                ?: throw IOException("MediaStore insert failed")
            try {
                resolver.openOutputStream(uri)?.use(write) ?: throw IOException("open failed")
                resolver.update(uri, ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }, null, null)
                return uri
            } catch (t: Throwable) {
                resolver.delete(uri, null, null)
                throw t
            }
        }
        val granted = ContextCompat.checkSelfPermission(context, Manifest.permission.WRITE_EXTERNAL_STORAGE) ==
            PackageManager.PERMISSION_GRANTED
        if (!granted) throw IOException("no storage permission")
        @Suppress("DEPRECATION")
        val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM), "TheRealCam").apply { mkdirs() }
        val f = File(dir, displayName)
        f.outputStream().use(write)
        MediaScannerConnection.scanFile(context, arrayOf(f.absolutePath), arrayOf("image/jpeg"), null)
        return Uri.fromFile(f)
    }

    fun open(uri: Uri): InputStream? = runCatching {
        if (uri.scheme == "file") File(uri.path!!).inputStream() else context.contentResolver.openInputStream(uri)
    }.getOrNull()

    fun exists(uri: Uri): Boolean = open(uri)?.use { true } ?: false

    fun delete(uri: Uri) {
        runCatching {
            if (uri.scheme == "file") File(uri.path!!).delete() else context.contentResolver.delete(uri, null, null)
        }
    }
}
