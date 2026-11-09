package com.therealcam.ai

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import com.therealcam.ai.engine.ModelClass
import com.therealcam.ai.engine.TrcmFormatException
import com.therealcam.ai.engine.TrcmModel
import com.therealcam.core.AppPaths
import com.therealcam.data.SettingsRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class ModelFileInfo(
    val file: File,
    val name: String,
    val classes: List<ModelClass>,
    val samples: Int?,
    val created: String?,
    val sizeBytes: Long,
    val error: String?,
)

/** Imported `.trcm` files in TheRealCam/models and the currently selected one. */
class ModelRepository(
    private val context: Context,
    private val paths: AppPaths,
    private val settings: SettingsRepository,
    scope: CoroutineScope,
) {
    private val _active = MutableStateFlow<TrcmModel?>(null)
    val active: StateFlow<TrcmModel?> = _active.asStateFlow()

    private val _activeError = MutableStateFlow<String?>(null)
    val activeError: StateFlow<String?> = _activeError.asStateFlow()

    private val _files = MutableStateFlow<List<ModelFileInfo>>(emptyList())
    val files: StateFlow<List<ModelFileInfo>> = _files.asStateFlow()

    init {
        scope.launch(Dispatchers.IO) {
            refresh()
            settings.settings.map { it.activeModel }.distinctUntilChanged().collect { name -> load(name) }
        }
    }

    private fun load(fileName: String?) {
        if (fileName == null) {
            _active.value = null
            _activeError.value = null
            return
        }
        val f = File(paths.models, fileName)
        try {
            _active.value = TrcmModel.load(f)
            _activeError.value = null
        } catch (e: Exception) {
            _active.value = null
            _activeError.value = "Модель «$fileName» не загружена: ${e.message}"
        }
    }

    suspend fun refresh() = withContext(Dispatchers.IO) {
        _files.value = paths.models.listFiles { f -> f.isFile && f.name.endsWith(".trcm", ignoreCase = true) }
            ?.sortedByDescending { it.lastModified() }
            ?.map { f ->
                try {
                    val m = TrcmModel.load(f)
                    ModelFileInfo(f, m.name, m.classes, m.trainedSamples, m.createdUtc, f.length(), null)
                } catch (e: Exception) {
                    ModelFileInfo(f, f.nameWithoutExtension, emptyList(), null, null, f.length(), e.message)
                }
            }.orEmpty()
    }

    /** Validates and copies a model chosen by the user. Returns the stored info. */
    suspend fun import(uri: Uri): Result<ModelFileInfo> = withContext(Dispatchers.IO) {
        runCatching {
            val bytes = context.contentResolver.openInputStream(uri)?.use { input ->
                val limit = 70L shl 20
                val buf = java.io.ByteArrayOutputStream()
                val chunk = ByteArray(64 shl 10)
                var total = 0L
                while (true) {
                    val n = input.read(chunk)
                    if (n < 0) break
                    total += n
                    if (total > limit) throw TrcmFormatException("Файл слишком большой")
                    buf.write(chunk, 0, n)
                }
                buf.toByteArray()
            } ?: throw TrcmFormatException("Не удалось открыть файл")
            val model = TrcmModel.parse(bytes)
            val base = (displayName(uri)?.substringBeforeLast('.') ?: model.name)
                .replace(Regex("[^\\p{L}\\p{N}._-]+"), "_").take(60).ifEmpty { "model" }
            var target = File(paths.models, "$base.trcm")
            var i = 2
            while (target.exists()) target = File(paths.models, "${base}_${i++}.trcm")
            target.writeBytes(bytes)
            refresh()
            ModelFileInfo(target, model.name, model.classes, model.trainedSamples, model.createdUtc, target.length(), null)
        }
    }

    suspend fun select(file: File?) = settings.setActiveModel(file?.name)

    suspend fun delete(file: File) = withContext(Dispatchers.IO) {
        if (settings.current().activeModel == file.name) settings.setActiveModel(null)
        file.delete()
        refresh()
    }

    private fun displayName(uri: Uri): String? = runCatching {
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        }
    }.getOrNull()
}
