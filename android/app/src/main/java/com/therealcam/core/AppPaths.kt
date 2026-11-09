package com.therealcam.core

import android.content.Context
import java.io.File

/** Storage layout, see docs/DATASET_FORMAT.md. Everything lives in app-specific external storage. */
class AppPaths(context: Context) {
    val root: File = File(context.getExternalFilesDir(null) ?: context.filesDir, "TheRealCam")
    val dataset = File(root, "dataset")
    val datasetSamples = File(dataset, "samples")
    val pending = File(root, "pending")
    val series = File(root, "series")
    val photos = File(root, "photos")
    val models = File(root, "models")
    val exports = File(root, "exports")

    init {
        ensure()
    }

    fun ensure() {
        listOf(root, dataset, datasetSamples, pending, series, photos, models, exports).forEach { it.mkdirs() }
    }

    fun pendingDir(sampleId: String) = File(pending, sampleId)
    fun datasetDir(sampleId: String) = File(datasetSamples, sampleId)
    fun seriesDir(seriesId: String) = File(series, seriesId)

    /** Folder that currently holds the sample (dataset if complete, else pending), or null. */
    fun sampleDir(sampleId: String): File? =
        datasetDir(sampleId).takeIf { it.isDirectory } ?: pendingDir(sampleId).takeIf { it.isDirectory }
}
