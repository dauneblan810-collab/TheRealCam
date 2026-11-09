package com.therealcam

import android.content.Context
import androidx.room.Room
import com.therealcam.ai.ModelRepository
import com.therealcam.core.AppPaths
import com.therealcam.data.CaptureRepository
import com.therealcam.data.CategoryRepository
import com.therealcam.data.PhotoStore
import com.therealcam.data.SettingsRepository
import com.therealcam.data.db.AppDatabase
import com.therealcam.dataset.DatasetManager
import com.therealcam.sensors.LocationTracker
import com.therealcam.sensors.SensorHub
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

/** Process-wide singletons. Created once in [TheRealCamApp.onCreate]. */
class AppGraph(val context: Context) {
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    val paths = AppPaths(context)
    val db: AppDatabase = Room.databaseBuilder(context, AppDatabase::class.java, "therealcam.db").build()
    val settings = SettingsRepository(context)
    val photos = PhotoStore(context, paths)
    val dataset = DatasetManager(paths)
    val captures = CaptureRepository(db, dataset, photos, paths)
    val categories = CategoryRepository(db, dataset, captures)
    val sensors = SensorHub(context)
    val location = LocationTracker(context)
    val models = ModelRepository(context, paths, settings, scope)
}
