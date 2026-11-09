package com.therealcam.sensors

import kotlin.math.PI
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.tan

/** Low-precision Sun and Moon positions (~1°), good enough for scene context features. */
object Astro {
    data class AltAz(val altDeg: Double, val azDeg: Double)

    private const val DEG = PI / 180.0

    private fun daysSinceJ2000(timeMs: Long): Double = timeMs / 86_400_000.0 + 2440587.5 - 2451545.0

    private fun norm360(x: Double): Double = ((x % 360.0) + 360.0) % 360.0

    private fun toAltAz(raRad: Double, decRad: Double, d: Double, latDeg: Double, lonDeg: Double): AltAz {
        val gmstDeg = norm360((18.697374558 + 24.06570982441908 * d) * 15.0)
        val haRad = (gmstDeg + lonDeg) * DEG - raRad
        val lat = latDeg * DEG
        val alt = asin(sin(lat) * sin(decRad) + cos(lat) * cos(decRad) * cos(haRad))
        val az = atan2(-sin(haRad), tan(decRad) * cos(lat) - sin(lat) * cos(haRad))
        return AltAz(alt / DEG, norm360(az / DEG))
    }

    private fun sunEcliptic(d: Double): Double {
        val g = norm360(357.529 + 0.98560028 * d) * DEG
        val q = norm360(280.459 + 0.98564736 * d)
        return norm360(q + 1.915 * sin(g) + 0.020 * sin(2 * g)) * DEG
    }

    private fun obliquity(d: Double) = (23.439 - 0.00000036 * d) * DEG

    fun sun(timeMs: Long, latDeg: Double, lonDeg: Double): AltAz {
        val d = daysSinceJ2000(timeMs)
        val l = sunEcliptic(d)
        val e = obliquity(d)
        val ra = atan2(cos(e) * sin(l), cos(l))
        val dec = asin(sin(e) * sin(l))
        return toAltAz(ra, dec, d, latDeg, lonDeg)
    }

    private data class MoonEcl(val lambda: Double, val beta: Double)

    private fun moonEcliptic(d: Double): MoonEcl {
        val l = norm360(218.316 + 13.176396 * d)
        val m = norm360(134.963 + 13.064993 * d) * DEG
        val f = norm360(93.272 + 13.229350 * d) * DEG
        return MoonEcl(norm360(l + 6.289 * sin(m)) * DEG, 5.128 * sin(f) * DEG)
    }

    fun moon(timeMs: Long, latDeg: Double, lonDeg: Double): AltAz {
        val d = daysSinceJ2000(timeMs)
        val (lambda, beta) = moonEcliptic(d)
        val e = obliquity(d)
        val ra = atan2(sin(lambda) * cos(e) - tan(beta) * sin(e), cos(lambda))
        val dec = asin(sin(beta) * cos(e) + cos(beta) * sin(e) * sin(lambda))
        val geo = toAltAz(ra, dec, d, latDeg, lonDeg)
        // Horizontal parallax ~0.95°.
        return AltAz(geo.altDeg - 0.95 * cos(geo.altDeg * DEG), geo.azDeg)
    }

    /** Illuminated fraction of the Moon's disk, 0..1. */
    fun moonIllumination(timeMs: Long): Double {
        val d = daysSinceJ2000(timeMs)
        val (lambda, beta) = moonEcliptic(d)
        val cosElong = cos(beta) * cos(lambda - sunEcliptic(d))
        return ((1.0 - cosElong) / 2.0).coerceIn(0.0, 1.0)
    }
}
