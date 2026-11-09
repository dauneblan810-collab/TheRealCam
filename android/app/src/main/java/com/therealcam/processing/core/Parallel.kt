package com.therealcam.processing.core

import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.min

internal fun defaultWorkers(): Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 8)

/** Runs body(index, worker) for 0 until count on up to [workers] coroutines of the current dispatcher. */
internal suspend fun parallelFor(count: Int, workers: Int, body: (index: Int, worker: Int) -> Unit) {
    if (count <= 0) return
    val t = min(workers, count).coerceAtLeast(1)
    coroutineScope {
        val next = AtomicInteger()
        repeat(t) { w ->
            launch {
                while (true) {
                    val i = next.getAndIncrement()
                    if (i >= count) break
                    ensureActive()
                    body(i, w)
                }
            }
        }
    }
}

/** Splits 0 until total into contiguous ranges, one per worker; worker indices are unique per call. */
internal suspend fun parallelRanges(total: Int, workers: Int, body: (from: Int, to: Int, worker: Int) -> Unit) {
    if (total <= 0) return
    val t = min(workers, total).coerceAtLeast(1)
    coroutineScope {
        repeat(t) { w ->
            launch {
                ensureActive()
                body(total * w / t, total * (w + 1) / t, w)
            }
        }
    }
}

/** Thread-safe, monotonic progress forwarding. */
internal class Progress(private val sink: (Float) -> Unit) {
    private var last = -1f

    @Synchronized
    fun report(fraction: Float) {
        val v = fraction.coerceIn(0f, 1f)
        if (v > last) {
            last = v
            sink(v)
        }
    }
}
