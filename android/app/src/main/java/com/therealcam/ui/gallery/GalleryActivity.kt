package com.therealcam.ui.gallery

import android.content.Context
import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.Text
import com.therealcam.ui.theme.RealTheme

class GalleryActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { RealTheme { Text("Real Галерея") } }
    }

    companion object {
        const val EXTRA_OPEN_ID = "open_id"

        fun intent(context: Context, openId: String?) = Intent(context, GalleryActivity::class.java).apply {
            openId?.let { putExtra(EXTRA_OPEN_ID, it) }
        }
    }
}
