package com.therealcam.ai.engine

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import kotlin.math.abs

/**
 * Cross-checks the Kotlin interpreter against vectors produced by the desktop trainer
 * (`python -m therealcam_trainer make-golden -o testdata/golden`), see docs/MODEL_FORMAT.md §6.
 */
class TrcmGoldenTest {
    private fun goldenDir(): File? {
        var d: File? = File(System.getProperty("user.dir") ?: ".").absoluteFile
        repeat(5) {
            val g = File(d, "testdata/golden")
            if (File(g, "golden.json").isFile) return g
            d = d?.parentFile
        }
        return null
    }

    @Test
    fun matchesTrainerReference() {
        val dir = goldenDir()
        assumeTrue("testdata/golden not found", dir != null)
        val golden = Json.parseToJsonElement(File(dir, "golden.json").readText()).jsonObject
        val model = TrcmModel.load(File(dir, golden["model"]!!.jsonPrimitive.content))
        val atol = golden["atol"]?.jsonPrimitive?.doubleOrNull ?: 1e-4
        val rtol = golden["rtol"]?.jsonPrimitive?.doubleOrNull ?: 1e-4
        val cases = golden["cases"]!!.jsonArray
        assert(cases.isNotEmpty())
        for ((ci, case) in cases.withIndex()) {
            val o = case.jsonObject
            val hwc = o["thumb_rgb_hwc"]!!.jsonArray.map { it.jsonPrimitive.int }
            assertEquals(64 * 64 * 3, hwc.size)
            val argb = IntArray(64 * 64) { i ->
                (0xFF shl 24) or (hwc[i * 3] shl 16) or (hwc[i * 3 + 1] shl 8) or hwc[i * 3 + 2]
            }
            val feats = (o["features"] as JsonObject).mapValues { (_, v) -> if (v is JsonNull) null else v.jsonPrimitive.doubleOrNull }
            val out = model.run(argb, feats)
            fun check(name: String, expected: List<Double>, actual: FloatArray) {
                assertEquals("$name size (case $ci)", expected.size, actual.size)
                for (i in expected.indices) {
                    val tol = atol + rtol * abs(expected[i])
                    assert(abs(expected[i] - actual[i]) <= tol) {
                        "case $ci $name[$i]: expected ${expected[i]}, got ${actual[i]}"
                    }
                }
            }
            check("class_logits", o["expected_class_logits"]!!.jsonArray.map { it.jsonPrimitive.doubleOrNull!! }, out.logits)
            check("params", o["expected_params"]!!.jsonArray.map { it.jsonPrimitive.doubleOrNull!! }, out.params)
        }
    }
}
