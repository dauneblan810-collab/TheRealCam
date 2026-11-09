package com.therealcam.core

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNamingStrategy
import java.security.SecureRandom
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object Ids {
    private val rnd = SecureRandom()

    private fun hex(n: Int): String = buildString { repeat(n) { append("0123456789abcdef"[rnd.nextInt(16)]) } }

    /** `yyyyMMdd_HHmmss_SSS_xxxx` in local time: sortable and unique. */
    fun newSampleId(timeMs: Long = System.currentTimeMillis()): String =
        SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date(timeMs)) + "_" + hex(4)

    fun newSeriesId(timeMs: Long = System.currentTimeMillis()): String = "S" + newSampleId(timeMs)

    fun newCategoryId(): String = "cat_" + hex(8)
}

@OptIn(ExperimentalSerializationApi::class)
val SnakeJson = Json {
    namingStrategy = JsonNamingStrategy.SnakeCase
    ignoreUnknownKeys = true
    explicitNulls = true
    encodeDefaults = true
    coerceInputValues = true
}

@OptIn(ExperimentalSerializationApi::class)
val PrettySnakeJson = Json(SnakeJson) {
    prettyPrint = true
    prettyPrintIndent = "  "
}

/** JSON has no NaN/Infinity: map them to null. */
fun Double.finiteOrNull(): Double? = if (isFinite()) this else null
fun Float.finiteOrNull(): Double? = if (isFinite()) toDouble() else null
