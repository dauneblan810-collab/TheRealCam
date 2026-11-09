package com.therealcam.ui.gallery

import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import com.therealcam.ui.common.CategoriesScreen
import com.therealcam.ui.settings.SettingsActivity
import com.therealcam.ui.theme.RealTheme

private sealed interface Route {
    data object Grid : Route
    data class Detail(val ids: List<String>, val index: Int) : Route
    data object Categories : Route
}

/** "Real Галерея": second launcher entry of the same APK. */
class GalleryActivity : ComponentActivity() {
    private val vm: GalleryViewModel by viewModels()
    private val stack = mutableStateListOf<Route>(Route.Grid)

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge(SystemBarStyle.dark(Color.TRANSPARENT), SystemBarStyle.dark(Color.BLACK))
        super.onCreate(savedInstanceState)
        handle(intent)
        setContent {
            RealTheme {
                val route = stack.last()
                BackHandler(enabled = stack.size > 1) { stack.removeAt(stack.lastIndex) }
                when (route) {
                    Route.Grid -> GalleryGrid(
                        vm,
                        onOpen = { ids, i -> stack.add(Route.Detail(ids, i)) },
                        onCategories = { stack.add(Route.Categories) },
                        onSettings = { startActivity(Intent(this, SettingsActivity::class.java)) },
                    )
                    is Route.Detail -> DetailScreen(
                        vm, route.ids, route.index,
                        onBack = { if (stack.size > 1) stack.removeAt(stack.lastIndex) else finish() },
                        onOpenFrames = { ids, i -> stack.add(Route.Detail(ids, i)) },
                    )
                    Route.Categories -> {
                        val categories by vm.categories.collectAsState()
                        CategoriesScreen(
                            categories,
                            onBack = { stack.removeAt(stack.lastIndex) },
                            onCreate = { n, i -> vm.createCategory(n, i) },
                            onUpdate = vm::updateCategory,
                            onDelete = vm::deleteCategory,
                        )
                    }
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handle(intent)
    }

    private fun handle(intent: Intent?) {
        val id = intent?.getStringExtra(EXTRA_OPEN_ID) ?: return
        intent.removeExtra(EXTRA_OPEN_ID)
        stack.clear()
        stack.add(Route.Grid)
        stack.add(Route.Detail(listOf(id), 0))
    }

    companion object {
        const val EXTRA_OPEN_ID = "open_id"

        fun intent(context: Context, openId: String?) = Intent(context, GalleryActivity::class.java).apply {
            openId?.let { putExtra(EXTRA_OPEN_ID, it) }
        }
    }
}
