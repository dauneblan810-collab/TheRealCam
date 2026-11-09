package com.therealcam.dataset

import android.util.Log
import com.therealcam.BuildConfig
import com.therealcam.ai.ScenePreview
import com.therealcam.core.AppPaths
import com.therealcam.core.ImageIo
import com.therealcam.core.PrettySnakeJson
import com.therealcam.core.SnakeJson
import com.therealcam.data.db.CaptureEntity
import com.therealcam.data.db.CategoryEntity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

const val MIN_DATASET_SAMPLES = 150

@Serializable
data class CategoriesFile(val schema: String = "therealcam.categories/1", val categories: List<CategoryJson>)

@Serializable
data class CategoryJson(val id: String, val name: String, val icon: String, val createdUtcMs: Long)

@Serializable
data class DatasetManifest(
    val schema: String = "therealcam.dataset/1",
    val appVersion: String,
    val updatedUtcMs: Long,
    val completeSamples: Int,
    val minSamplesForTraining: Int = MIN_DATASET_SAMPLES,
    val samples: List<String>,
)

/**
 * Owns the dataset folders (docs/DATASET_FORMAT.md). A sample is written to pending/ at capture
 * time and moved to dataset/samples/ once every rating criterion is filled in.
 */
class DatasetManager(private val paths: AppPaths) {
    private val lock = Mutex()

    suspend fun writePending(
        meta: SampleMeta,
        image: () -> InputStream?,
        preview: ScenePreview?,
        maxSide: Int,
    ) = withContext(Dispatchers.IO) {
        lock.withLock {
            val dir = paths.pendingDir(meta.sampleId).apply { mkdirs() }
            writeMeta(dir, meta)
            runCatching {
                ImageIo.decodeUpright({ image() ?: throw java.io.IOException("no image") }, maxSide)?.let {
                    ImageIo.writeJpeg(it, File(dir, "image.jpg"), 90)
                    it.recycle()
                }
            }.onFailure { Log.w(TAG, "dataset image failed", it) }
            preview?.let {
                ImageIo.writeJpeg(it.bitmap(), File(dir, "preview.jpg"), 92)
                ImageIo.writePng(it.thumbBitmap(), File(dir, "thumb64.png"))
            }
        }
    }

    fun readMeta(sampleId: String): SampleMeta? = paths.sampleDir(sampleId)?.let { readMeta(it) }

    private fun readMeta(dir: File): SampleMeta? = runCatching {
        SnakeJson.decodeFromString(SampleMeta.serializer(), File(dir, "meta.json").readText())
    }.onFailure { Log.w(TAG, "bad meta in $dir", it) }.getOrNull()

    private fun writeMeta(dir: File, meta: SampleMeta) {
        val tmp = File(dir, "meta.json.tmp")
        tmp.writeText(PrettySnakeJson.encodeToString(SampleMeta.serializer(), meta))
        val dst = File(dir, "meta.json")
        if (!tmp.renameTo(dst)) {
            dst.delete()
            tmp.renameTo(dst)
        }
    }

    /**
     * Copies the ratings of [e] into meta.json and moves the sample between pending/ and dataset/.
     * Returns true when the sample is (now) part of the dataset.
     */
    suspend fun syncLabels(e: CaptureEntity, category: CategoryEntity?): Boolean = withContext(Dispatchers.IO) {
        lock.withLock {
            val dir = paths.sampleDir(e.id) ?: return@withLock false
            val meta = readMeta(dir) ?: return@withLock false
            val complete = e.isRatingComplete && category != null
            val labels = Labels(
                rating = e.rating, exposure = e.fbExposure, sharpness = e.fbSharpness, noise = e.fbNoise,
                color = e.fbColor, motionBlur = e.fbMotionBlur,
                categoryId = category?.id, categoryName = category?.name, categoryIcon = category?.icon,
                note = e.note, ratedUtcMs = System.currentTimeMillis(), complete = complete,
            )
            writeMeta(dir, meta.copy(labels = labels))
            val target = if (complete) paths.datasetDir(e.id) else paths.pendingDir(e.id)
            if (dir.absolutePath != target.absolutePath) {
                target.parentFile?.mkdirs()
                if (!dir.renameTo(target)) {
                    dir.copyRecursively(target, overwrite = true)
                    dir.deleteRecursively()
                }
            }
            complete
        }
    }

    suspend fun deleteSample(sampleId: String) = withContext(Dispatchers.IO) {
        lock.withLock {
            paths.pendingDir(sampleId).deleteRecursively()
            paths.datasetDir(sampleId).deleteRecursively()
        }
    }

    suspend fun writeCategories(categories: List<CategoryEntity>) = withContext(Dispatchers.IO) {
        val file = CategoriesFile(categories = categories.map { CategoryJson(it.id, it.name, it.icon, it.createdAt) })
        File(paths.dataset, "categories.json").writeText(PrettySnakeJson.encodeToString(CategoriesFile.serializer(), file))
    }

    fun completeSampleIds(): List<String> =
        paths.datasetSamples.listFiles()?.filter { File(it, "meta.json").isFile }?.map { it.name }?.sorted().orEmpty()

    suspend fun rebuildManifest() = withContext(Dispatchers.IO) {
        val ids = completeSampleIds()
        val m = DatasetManifest(
            appVersion = BuildConfig.VERSION_NAME,
            updatedUtcMs = System.currentTimeMillis(),
            completeSamples = ids.size,
            samples = ids,
        )
        File(paths.dataset, "dataset.json").writeText(PrettySnakeJson.encodeToString(DatasetManifest.serializer(), m))
    }

    /** Writes `dataset/` (complete samples only) as a ZIP. Returns the number of samples. */
    suspend fun exportZip(out: OutputStream, onProgress: (done: Int, total: Int) -> Unit = { _, _ -> }): Int =
        withContext(Dispatchers.IO) {
            rebuildManifest()
            val ids = completeSampleIds()
            ZipOutputStream(out.buffered()).use { zip ->
                zip.setLevel(Deflater.BEST_SPEED)
                fun put(file: File, name: String) {
                    if (!file.isFile) return
                    zip.putNextEntry(ZipEntry(name).apply { time = file.lastModified() })
                    file.inputStream().use { it.copyTo(zip) }
                    zip.closeEntry()
                }
                put(File(paths.dataset, "dataset.json"), "dataset/dataset.json")
                put(File(paths.dataset, "categories.json"), "dataset/categories.json")
                ids.forEachIndexed { i, id ->
                    val dir = paths.datasetDir(id)
                    dir.listFiles()?.sortedBy { it.name }?.forEach { put(it, "dataset/samples/$id/${it.name}") }
                    onProgress(i + 1, ids.size)
                }
            }
            ids.size
        }

    companion object {
        private const val TAG = "DatasetManager"
    }
}
