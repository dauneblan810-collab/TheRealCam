package com.therealcam.ui.gallery

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.therealcam.appGraph
import com.therealcam.data.db.CaptureEntity
import com.therealcam.data.db.CategoryEntity
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

enum class GalleryFilter(val label: String) {
    ALL("Все"),
    TO_RATE("Нужна оценка"),
    DATASET("В датасете"),
    SERIES("Серии"),
}

class GalleryViewModel(app: Application) : AndroidViewModel(app) {
    private val g = app.appGraph

    val categories: StateFlow<List<CategoryEntity>> = g.categories.all.stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())
    val datasetCount: StateFlow<Int> = g.captures.datasetCount.stateIn(viewModelScope, SharingStarted.Eagerly, 0)
    val pendingCount: StateFlow<Int> = g.captures.pendingCount.stateIn(viewModelScope, SharingStarted.Eagerly, 0)

    private val _filter = MutableStateFlow(GalleryFilter.ALL)
    val filter = _filter.asStateFlow()
    private val _categoryFilter = MutableStateFlow<String?>(null)
    val categoryFilter = _categoryFilter.asStateFlow()

    private val all: StateFlow<List<CaptureEntity>?> = g.captures.main.stateIn(viewModelScope, SharingStarted.Eagerly, null)

    val items: StateFlow<List<CaptureEntity>?> = combine(all, _filter, _categoryFilter) { list, f, cat ->
        list?.filter { e ->
            (cat == null || e.categoryId == cat) && when (f) {
                GalleryFilter.ALL -> true
                GalleryFilter.TO_RATE -> e.hasMeta && !e.inDataset
                GalleryFilter.DATASET -> e.inDataset
                GalleryFilter.SERIES -> e.seriesId != null
            }
        }
    }.stateIn(viewModelScope, SharingStarted.Eagerly, null)

    fun setFilter(f: GalleryFilter) {
        _filter.value = f
    }

    fun setCategoryFilter(id: String?) {
        _categoryFilter.value = if (_categoryFilter.value == id) null else id
    }

    fun observe(id: String): Flow<CaptureEntity?> = g.captures.observe(id)
    fun series(seriesId: String): Flow<List<CaptureEntity>> = g.captures.series(seriesId)

    fun rate(id: String, change: (CaptureEntity) -> CaptureEntity) {
        viewModelScope.launch { g.captures.rate(id, change) }
    }

    fun delete(e: CaptureEntity) {
        viewModelScope.launch { g.captures.delete(e) }
    }

    fun createCategory(name: String, icon: String, assignTo: String? = null) {
        viewModelScope.launch {
            val c = g.categories.create(name, icon)
            if (assignTo != null) g.captures.rate(assignTo) { it.copy(categoryId = c.id) }
        }
    }

    fun updateCategory(c: CategoryEntity) {
        viewModelScope.launch { g.categories.update(c) }
    }

    fun deleteCategory(id: String) {
        viewModelScope.launch { g.categories.delete(id) }
    }

    fun meta(id: String) = g.dataset.readMeta(id)
}
