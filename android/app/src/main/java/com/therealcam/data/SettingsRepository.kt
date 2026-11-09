package com.therealcam.data

import android.content.Context
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "settings")

/** Bump when the list of collected data changes so the consent screen is shown again. */
const val DATASET_CONSENT_VERSION = 1

data class AppSettings(
    val datasetMode: Boolean = false,
    val consentVersion: Int = 0,
    val geotag: Boolean = true,
    val askRating: Boolean = true,
    val saveToGallery: Boolean = true,
    val datasetImageMaxSide: Int = 1600,
    val jpegQuality: Int = 95,
    val grid: Boolean = true,
    val level: Boolean = true,
    val volumeShutter: Boolean = true,
    val timerSeconds: Int = 0,
    val activeModel: String? = null,
    val lastMode: String = "auto",
    val lastCategoryHint: String? = null,
)

class SettingsRepository(private val context: Context) {
    private object K {
        val datasetMode = booleanPreferencesKey("dataset_mode")
        val consentVersion = intPreferencesKey("consent_version")
        val geotag = booleanPreferencesKey("geotag")
        val askRating = booleanPreferencesKey("ask_rating")
        val saveToGallery = booleanPreferencesKey("save_to_gallery")
        val datasetImageMaxSide = intPreferencesKey("dataset_image_max_side")
        val jpegQuality = intPreferencesKey("jpeg_quality")
        val grid = booleanPreferencesKey("grid")
        val level = booleanPreferencesKey("level")
        val volumeShutter = booleanPreferencesKey("volume_shutter")
        val timerSeconds = intPreferencesKey("timer_seconds")
        val activeModel = stringPreferencesKey("active_model")
        val lastMode = stringPreferencesKey("last_mode")
        val lastCategoryHint = stringPreferencesKey("last_category_hint")
    }

    val settings: Flow<AppSettings> = context.dataStore.data.map { it.toSettings() }

    suspend fun current(): AppSettings = settings.first()

    private fun Preferences.toSettings(): AppSettings {
        val d = AppSettings()
        return AppSettings(
            datasetMode = this[K.datasetMode] ?: d.datasetMode,
            consentVersion = this[K.consentVersion] ?: d.consentVersion,
            geotag = this[K.geotag] ?: d.geotag,
            askRating = this[K.askRating] ?: d.askRating,
            saveToGallery = this[K.saveToGallery] ?: d.saveToGallery,
            datasetImageMaxSide = this[K.datasetImageMaxSide] ?: d.datasetImageMaxSide,
            jpegQuality = this[K.jpegQuality] ?: d.jpegQuality,
            grid = this[K.grid] ?: d.grid,
            level = this[K.level] ?: d.level,
            volumeShutter = this[K.volumeShutter] ?: d.volumeShutter,
            timerSeconds = this[K.timerSeconds] ?: d.timerSeconds,
            activeModel = this[K.activeModel],
            lastMode = this[K.lastMode] ?: d.lastMode,
            lastCategoryHint = this[K.lastCategoryHint],
        )
    }

    suspend fun setDatasetMode(enabled: Boolean) = context.dataStore.edit {
        it[K.datasetMode] = enabled
        if (enabled) it[K.consentVersion] = DATASET_CONSENT_VERSION
    }

    suspend fun setGeotag(v: Boolean) = context.dataStore.edit { it[K.geotag] = v }
    suspend fun setAskRating(v: Boolean) = context.dataStore.edit { it[K.askRating] = v }
    suspend fun setSaveToGallery(v: Boolean) = context.dataStore.edit { it[K.saveToGallery] = v }
    suspend fun setDatasetImageMaxSide(v: Int) = context.dataStore.edit { it[K.datasetImageMaxSide] = v }
    suspend fun setJpegQuality(v: Int) = context.dataStore.edit { it[K.jpegQuality] = v.coerceIn(70, 100) }
    suspend fun setGrid(v: Boolean) = context.dataStore.edit { it[K.grid] = v }
    suspend fun setLevel(v: Boolean) = context.dataStore.edit { it[K.level] = v }
    suspend fun setVolumeShutter(v: Boolean) = context.dataStore.edit { it[K.volumeShutter] = v }
    suspend fun setTimerSeconds(v: Int) = context.dataStore.edit { it[K.timerSeconds] = v }
    suspend fun setLastMode(v: String) = context.dataStore.edit { it[K.lastMode] = v }

    suspend fun setActiveModel(fileName: String?) = context.dataStore.edit {
        if (fileName == null) it.remove(K.activeModel) else it[K.activeModel] = fileName
    }

    suspend fun setLastCategoryHint(id: String?) = context.dataStore.edit {
        if (id == null) it.remove(K.lastCategoryHint) else it[K.lastCategoryHint] = id
    }
}
