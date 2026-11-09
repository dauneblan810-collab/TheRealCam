package com.therealcam.processing.core

import kotlin.math.abs
import kotlin.math.max

/** Scale factor that turns a MAD into a standard deviation for Gaussian data. */
internal const val MAD_TO_SIGMA = 1.4826

internal object RobustStats {

    fun median(values: DoubleArray): Double {
        require(values.isNotEmpty()) { "median of empty array" }
        val s = values.sortedArray()
        val n = s.size
        return if (n % 2 == 1) s[n / 2] else 0.5 * (s[n / 2 - 1] + s[n / 2])
    }

    /** Median absolute deviation from the median. */
    fun mad(values: DoubleArray, median: Double = median(values)): Double =
        median(DoubleArray(values.size) { abs(values[it] - median) })

    /**
     * Robust z-scores (value - median) / max(1.4826 * MAD, floor). The floor keeps very stable
     * series (MAD ~ 0) from flagging tiny differences.
     */
    fun robustZ(values: DoubleArray, floor: Double): DoubleArray {
        if (values.isEmpty()) return DoubleArray(0)
        val med = median(values)
        val scale = max(MAD_TO_SIGMA * mad(values, med), floor).coerceAtLeast(1e-12)
        return DoubleArray(values.size) { (values[it] - med) / scale }
    }

    /**
     * k-th smallest (0-based) of a[0 until n], partially reordering a. Wirth's selection:
     * allocation-free, average O(n); used per pixel.
     */
    fun kthSmallest(a: IntArray, n: Int, k: Int): Int {
        var l = 0
        var m = n - 1
        while (l < m) {
            val x = a[k]
            var i = l
            var j = m
            do {
                while (a[i] < x) i++
                while (x < a[j]) j--
                if (i <= j) {
                    val t = a[i]
                    a[i] = a[j]
                    a[j] = t
                    i++
                    j--
                }
            } while (i <= j)
            if (j < k) l = i
            if (k < i) m = j
        }
        return a[k]
    }
}
