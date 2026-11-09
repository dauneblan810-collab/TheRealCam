package com.therealcam.ui.common

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AcUnit
import androidx.compose.material.icons.filled.AccountBalance
import androidx.compose.material.icons.filled.Apartment
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.BlurOn
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.BugReport
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Celebration
import androidx.compose.material.icons.filled.Chair
import androidx.compose.material.icons.filled.ChildCare
import androidx.compose.material.icons.filled.Cloud
import androidx.compose.material.icons.filled.DarkMode
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.DirectionsCar
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.FilterDrama
import androidx.compose.material.icons.filled.FlutterDash
import androidx.compose.material.icons.filled.Forest
import androidx.compose.material.icons.filled.Gradient
import androidx.compose.material.icons.filled.Groups
import androidx.compose.material.icons.filled.Landscape
import androidx.compose.material.icons.filled.LocalFireDepartment
import androidx.compose.material.icons.filled.LocalFlorist
import androidx.compose.material.icons.filled.LocationCity
import androidx.compose.material.icons.filled.MusicNote
import androidx.compose.material.icons.filled.NightsStay
import androidx.compose.material.icons.filled.Pets
import androidx.compose.material.icons.filled.Portrait
import androidx.compose.material.icons.filled.Restaurant
import androidx.compose.material.icons.filled.SportsSoccer
import androidx.compose.material.icons.filled.Terrain
import androidx.compose.material.icons.filled.Timeline
import androidx.compose.material.icons.filled.TrackChanges
import androidx.compose.material.icons.filled.Tv
import androidx.compose.material.icons.filled.WaterDrop
import androidx.compose.material.icons.filled.Waves
import androidx.compose.material.icons.filled.WbSunny
import androidx.compose.material.icons.filled.WbTwilight
import androidx.compose.material.icons.filled.ZoomIn
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.path
import androidx.compose.ui.unit.dp
import com.therealcam.data.SceneKeys

/** Visuals for the built-in scene icon keys (see data/Profiles.kt and docs/MODEL_FORMAT.md §5). */
object SceneIcons {
    private val map: Map<String, ImageVector> = mapOf(
        "stars" to Icons.Filled.AutoAwesome,
        "milky_way" to Icons.Filled.BlurOn,
        "night_sky" to Icons.Filled.NightsStay,
        "aurora" to Icons.Filled.Gradient,
        "moon" to Icons.Filled.DarkMode,
        "clouds" to Icons.Filled.Cloud,
        "sky" to Icons.Filled.FilterDrama,
        "sunset" to Icons.Filled.WbTwilight,
        "sun" to Icons.Filled.WbSunny,
        "city_night" to Icons.Filled.Apartment,
        "fireworks" to Icons.Filled.Celebration,
        "concert" to Icons.Filled.MusicNote,
        "fire" to Icons.Filled.LocalFireDepartment,
        "lightning" to Icons.Filled.Bolt,
        "light_trails" to Icons.Filled.Timeline,
        "star_trails" to Icons.Filled.TrackChanges,
        "waterfall" to Icons.Filled.WaterDrop,
        "water" to Icons.Filled.Waves,
        "sport" to Icons.Filled.SportsSoccer,
        "pet" to Icons.Filled.Pets,
        "bird" to Icons.Filled.FlutterDash,
        "car" to Icons.Filled.DirectionsCar,
        "kids" to Icons.Filled.ChildCare,
        "landscape" to Icons.Filled.Landscape,
        "mountains" to Icons.Filled.Terrain,
        "forest" to Icons.Filled.Forest,
        "snow" to Icons.Filled.AcUnit,
        "city" to Icons.Filled.LocationCity,
        "architecture" to Icons.Filled.AccountBalance,
        "macro" to Icons.Filled.ZoomIn,
        "flower" to Icons.Filled.LocalFlorist,
        "insect" to Icons.Filled.BugReport,
        "portrait" to Icons.Filled.Portrait,
        "people" to Icons.Filled.Groups,
        "selfie" to Icons.Filled.Face,
        "interior" to Icons.Filled.Chair,
        "food" to Icons.Filled.Restaurant,
        "document" to Icons.Filled.Description,
        "screen" to Icons.Filled.Tv,
        "other" to Icons.Filled.CameraAlt,
    )

    init {
        check(SceneKeys.all.all { it.key in map })
    }

    fun of(key: String?): ImageVector = map[key] ?: Icons.Filled.CameraAlt

    /** Tripod: shown when the phone has to be fixed. Not part of the Material set. */
    val Tripod: ImageVector by lazy {
        ImageVector.Builder("Tripod", 24.dp, 24.dp, 24f, 24f).apply {
            path(fill = SolidColor(Color.White)) {
                moveTo(7f, 3f); lineTo(17f, 3f); lineTo(17f, 8f); lineTo(7f, 8f); close()
                moveTo(11f, 8f); lineTo(13f, 8f); lineTo(13f, 11f); lineTo(11f, 11f); close()
            }
            path(stroke = SolidColor(Color.White), strokeLineWidth = 2f, strokeLineCap = StrokeCap.Round, strokeLineJoin = StrokeJoin.Round) {
                moveTo(12f, 11f); lineTo(5f, 21f)
                moveTo(12f, 11f); lineTo(19f, 21f)
                moveTo(12f, 11f); lineTo(12f, 21f)
            }
        }.build()
    }
}
