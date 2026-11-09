package com.therealcam.ui.settings

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.therealcam.ai.ModelFileInfo
import com.therealcam.appGraph
import com.therealcam.data.AppSettings
import com.therealcam.data.db.CategoryEntity
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.io.File

class SettingsViewModel(app: Application) : AndroidViewModel(app) {
    private val g = app.appGraph
    val settings: StateFlow<AppSettings> = g.settings.settings.stateIn(viewModelScope, SharingStarted.Eagerly, AppSettings())
    val models: StateFlow<List<ModelFileInfo>> = g.models.files
    val modelError: StateFlow<String?> = g.models.activeError
    val categories: StateFlow<List<CategoryEntity>> = g.categories.all.stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())
    val datasetCount: StateFlow<Int> = g.captures.datasetCount.stateIn(viewModelScope, SharingStarted.Eagerly, 0)
    val pendingCount: StateFlow<Int> = g.captures.pendingCount.stateIn(viewModelScope, SharingStarted.Eagerly, 0)
    val datasetPath: String = g.paths.dataset.absolutePath

    private val _status = MutableStateFlow<String?>(null)
    val status = _status.asStateFlow()
    private val _busy = MutableStateFlow(false)
    val busy = _busy.asStateFlow()

    init {
        viewModelScope.launch { g.models.refresh() }
    }

    fun clearStatus() {
        _status.value = null
    }

    fun setDatasetMode(on: Boolean) = viewModelScope.launch { g.settings.setDatasetMode(on) }
    fun setGeotag(on: Boolean) = viewModelScope.launch { g.settings.setGeotag(on) }
    fun setAskRating(on: Boolean) = viewModelScope.launch { g.settings.setAskRating(on) }
    fun setSaveToGallery(on: Boolean) = viewModelScope.launch { g.settings.setSaveToGallery(on) }
    fun setGrid(on: Boolean) = viewModelScope.launch { g.settings.setGrid(on) }
    fun setLevel(on: Boolean) = viewModelScope.launch { g.settings.setLevel(on) }
    fun setVolumeShutter(on: Boolean) = viewModelScope.launch { g.settings.setVolumeShutter(on) }
    fun setJpegQuality(q: Int) = viewModelScope.launch { g.settings.setJpegQuality(q) }
    fun setDatasetImageMaxSide(px: Int) = viewModelScope.launch { g.settings.setDatasetImageMaxSide(px) }

    fun importModel(uri: Uri) = viewModelScope.launch {
        _busy.value = true
        g.models.import(uri)
            .onSuccess {
                _status.value = "Модель «${it.name}» загружена: ${it.classes.size} категорий"
                g.models.select(it.file)
            }
            .onFailure { _status.value = "Не удалось загрузить модель: ${it.message}" }
        _busy.value = false
    }

    fun selectModel(file: File?) = viewModelScope.launch { g.models.select(file) }
    fun deleteModel(file: File) = viewModelScope.launch { g.models.delete(file) }

    fun exportDataset(uri: Uri) = viewModelScope.launch {
        _busy.value = true
        runCatching {
            g.categories.list().let { g.dataset.writeCategories(it) }
            val out = getApplication<Application>().contentResolver.openOutputStream(uri) ?: error("не удалось открыть файл")
            out.use { g.dataset.exportZip(it) }
        }.onSuccess { n -> _status.value = "Экспортировано снимков: $n" }
            .onFailure { _status.value = "Ошибка экспорта: ${it.message}" }
        _busy.value = false
    }

    fun createCategory(name: String, icon: String) = viewModelScope.launch { g.categories.create(name, icon) }
    fun updateCategory(c: CategoryEntity) = viewModelScope.launch { g.categories.update(c) }
    fun deleteCategory(id: String) = viewModelScope.launch { g.categories.delete(id) }
}
