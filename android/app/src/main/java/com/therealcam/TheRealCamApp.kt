package com.therealcam

import android.app.Application
import android.content.Context

class TheRealCamApp : Application() {
    lateinit var graph: AppGraph
        private set

    override fun onCreate() {
        super.onCreate()
        graph = AppGraph(this)
    }
}

val Context.appGraph: AppGraph
    get() = (applicationContext as TheRealCamApp).graph
