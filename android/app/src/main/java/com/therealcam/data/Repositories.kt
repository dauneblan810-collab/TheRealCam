package com.therealcam.data

import com.therealcam.core.Ids
import com.therealcam.data.db.AppDatabase
import com.therealcam.data.db.CaptureEntity
import com.therealcam.data.db.CategoryEntity
import com.therealcam.data.db.Roles
import com.therealcam.dataset.DatasetManager
import com.therealcam.core.AppPaths
import android.net.Uri
import kotlinx.coroutines.flow.Flow
import java.io.File

class CaptureRepository(
    private val db: AppDatabase,
    private val dataset: DatasetManager,
    private val photos: PhotoStore,
    private val paths: AppPaths,
) {
    private val dao = db.captures()

    val main: Flow<List<CaptureEntity>> = dao.observeMain()
    val latest: Flow<CaptureEntity?> = dao.observeLatest()
    val datasetCount: Flow<Int> = dao.observeDatasetCount()
    val pendingCount: Flow<Int> = dao.observePendingCount()

    fun observe(id: String) = dao.observe(id)
    fun series(seriesId: String) = dao.observeSeries(seriesId)
    suspend fun get(id: String) = dao.get(id)
    suspend fun insert(e: CaptureEntity) = dao.upsert(e)

    /** Applies a rating change and keeps meta.json / the dataset folders in sync. */
    suspend fun rate(id: String, change: (CaptureEntity) -> CaptureEntity) {
        val cur = dao.get(id) ?: return
        val next = change(cur)
        val category = next.categoryId?.let { db.categories().get(it) }
        val inDataset = if (next.hasMeta) dataset.syncLabels(next, category) else false
        dao.update(next.copy(inDataset = inDataset))
        if (inDataset != cur.inDataset) dataset.rebuildManifest()
    }

    /** Deletes a photo; for a series result also every frame of that series. */
    suspend fun delete(e: CaptureEntity) {
        val victims = if (e.role == Roles.RESULT && e.seriesId != null) dao.seriesMembers(e.seriesId) else listOf(e)
        for (v in victims) {
            photos.delete(Uri.parse(v.imageUri))
            dataset.deleteSample(v.id)
            dao.delete(v.id)
        }
        if (e.role == Roles.RESULT && e.seriesId != null) File(paths.series, e.seriesId).deleteRecursively()
        dataset.rebuildManifest()
    }
}

class CategoryRepository(private val db: AppDatabase, private val dataset: DatasetManager, private val captures: CaptureRepository) {
    val all: Flow<List<CategoryEntity>> = db.categories().observeAll()

    suspend fun create(name: String, icon: String): CategoryEntity {
        val existing = db.categories().all()
        val c = CategoryEntity(Ids.newCategoryId(), name.trim().ifEmpty { SceneKeys.entry(icon).label }, icon,
            System.currentTimeMillis(), (existing.maxOfOrNull { it.sortOrder } ?: 0) + 1)
        db.categories().upsert(c)
        dataset.writeCategories(db.categories().all())
        return c
    }

    suspend fun update(c: CategoryEntity) {
        db.categories().upsert(c)
        dataset.writeCategories(db.categories().all())
        // Labels in meta.json carry the category name/icon: refresh them.
        for (e in db.captures().byCategory(c.id)) captures.rate(e.id) { it }
    }

    suspend fun delete(id: String) {
        val affected = db.captures().byCategory(id)
        db.categories().delete(id)
        for (e in affected) captures.rate(e.id) { it.copy(categoryId = null) }
        dataset.writeCategories(db.categories().all())
    }

    suspend fun get(id: String) = db.categories().get(id)
    suspend fun list() = db.categories().all()
}
