package com.therealcam.data.db

import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Index
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.RoomDatabase
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

object Roles {
    const val SINGLE = "single"
    const val FRAME = "series_frame"
    const val RESULT = "series_result"
}

/** One saved image: a single photo, a frame of a series or the merged result of a series. */
@Entity(
    tableName = "captures",
    indices = [Index("seriesId"), Index("createdAt"), Index("categoryId")],
)
data class CaptureEntity(
    @PrimaryKey val id: String,
    val createdAt: Long,
    val role: String,
    val seriesId: String? = null,
    val frameIndex: Int = -1,
    val seriesCount: Int = 0,
    val mode: String,
    val profile: String? = null,
    /** content:// (system gallery) or file:// URI of the full-size JPEG. */
    val imageUri: String,
    val width: Int,
    val height: Int,
    val orientation: Int = 0,
    val exposureNs: Long? = null,
    val iso: Int? = null,
    val focusDiopters: Float? = null,
    val lat: Double? = null,
    val lon: Double? = null,
    /** Dataset mode was on at capture time: meta.json exists in pending/ or dataset/. */
    val hasMeta: Boolean = false,
    val rating: Int? = null,
    val fbExposure: Int? = null,
    val fbSharpness: Int? = null,
    val fbNoise: Int? = null,
    val fbColor: Int? = null,
    val fbMotionBlur: Int? = null,
    val categoryId: String? = null,
    val note: String? = null,
    val inDataset: Boolean = false,
    val aiCategoryId: String? = null,
) {
    val isRatingComplete: Boolean
        get() = rating != null && fbExposure != null && fbSharpness != null && fbNoise != null &&
            fbColor != null && fbMotionBlur != null && categoryId != null
}

@Entity(tableName = "categories")
data class CategoryEntity(
    @PrimaryKey val id: String,
    val name: String,
    val icon: String,
    val createdAt: Long,
    val sortOrder: Int = 0,
)

@Dao
interface CaptureDao {
    /** Gallery stream: singles and series results; frames are reached through their series. */
    @Query("SELECT * FROM captures WHERE role != 'series_frame' ORDER BY createdAt DESC")
    fun observeMain(): Flow<List<CaptureEntity>>

    @Query("SELECT * FROM captures WHERE seriesId = :seriesId ORDER BY role = 'series_result' DESC, frameIndex ASC")
    fun observeSeries(seriesId: String): Flow<List<CaptureEntity>>

    @Query("SELECT * FROM captures WHERE id = :id")
    fun observe(id: String): Flow<CaptureEntity?>

    @Query("SELECT * FROM captures WHERE id = :id")
    suspend fun get(id: String): CaptureEntity?

    @Query("SELECT * FROM captures WHERE role != 'series_frame' ORDER BY createdAt DESC LIMIT 1")
    fun observeLatest(): Flow<CaptureEntity?>

    @Query("SELECT COUNT(*) FROM captures WHERE inDataset = 1")
    fun observeDatasetCount(): Flow<Int>

    @Query("SELECT COUNT(*) FROM captures WHERE hasMeta = 1 AND inDataset = 0")
    fun observePendingCount(): Flow<Int>

    @Query("SELECT * FROM captures WHERE seriesId = :seriesId")
    suspend fun seriesMembers(seriesId: String): List<CaptureEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(entity: CaptureEntity)

    @Update
    suspend fun update(entity: CaptureEntity)

    @Query("DELETE FROM captures WHERE id = :id")
    suspend fun delete(id: String)

    @Query("UPDATE captures SET categoryId = NULL, inDataset = 0 WHERE categoryId = :categoryId")
    suspend fun clearCategory(categoryId: String)
}

@Dao
interface CategoryDao {
    @Query("SELECT * FROM categories ORDER BY sortOrder ASC, createdAt ASC")
    fun observeAll(): Flow<List<CategoryEntity>>

    @Query("SELECT * FROM categories ORDER BY sortOrder ASC, createdAt ASC")
    suspend fun all(): List<CategoryEntity>

    @Query("SELECT * FROM categories WHERE id = :id")
    suspend fun get(id: String): CategoryEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(entity: CategoryEntity)

    @Query("DELETE FROM categories WHERE id = :id")
    suspend fun delete(id: String)
}

@Database(entities = [CaptureEntity::class, CategoryEntity::class], version = 1, exportSchema = true)
abstract class AppDatabase : RoomDatabase() {
    abstract fun captures(): CaptureDao
    abstract fun categories(): CategoryDao
}
