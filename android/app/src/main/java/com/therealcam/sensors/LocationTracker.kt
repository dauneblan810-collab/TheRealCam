package com.therealcam.sensors

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Looper
import androidx.core.content.ContextCompat
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Plain LocationManager (no Google services): GPS + network while the camera is open. */
class LocationTracker(private val context: Context) : LocationListener {
    private val lm = context.getSystemService(LocationManager::class.java)
    private val _location = MutableStateFlow<Location?>(null)
    val location: StateFlow<Location?> = _location.asStateFlow()
    private var running = false

    fun hasPermission(): Boolean =
        ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED

    @SuppressLint("MissingPermission")
    fun start() {
        if (running || lm == null || !hasPermission()) return
        running = true
        for (p in listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER, LocationManager.PASSIVE_PROVIDER)) {
            runCatching { lm.getLastKnownLocation(p) }.getOrNull()?.let { onLocationChanged(it) }
        }
        for (p in listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER)) {
            if (runCatching { lm.isProviderEnabled(p) }.getOrDefault(false)) {
                runCatching { lm.requestLocationUpdates(p, 5_000L, 5f, this, Looper.getMainLooper()) }
            }
        }
    }

    fun stop() {
        if (!running) return
        running = false
        runCatching { lm?.removeUpdates(this) }
    }

    override fun onLocationChanged(location: Location) {
        val cur = _location.value
        // Prefer fresher fixes; within 2 minutes prefer the more accurate one.
        val better = cur == null ||
            location.time - cur.time > 120_000 ||
            (location.time >= cur.time - 120_000 && location.accuracy <= cur.accuracy * 1.5f)
        if (better) _location.value = location
    }

    @Deprecated("Deprecated in Java")
    override fun onStatusChanged(provider: String?, status: Int, extras: android.os.Bundle?) = Unit
    override fun onProviderEnabled(provider: String) = Unit
    override fun onProviderDisabled(provider: String) = Unit
}
