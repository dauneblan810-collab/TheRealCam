package com.therealcam.ai.engine

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.CRC32
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random

/** Builds .trcm files in tests exactly as described in docs/MODEL_FORMAT.md §1. */
object TrcmTestWriter {
    class T(val name: String, val shape: IntArray, val data: FloatArray)

    fun build(headerWithoutData: String, tensors: List<T>, corruptCrc: Boolean = false): ByteArray {
        val all = tensors.flatMap { it.data.toList() }.toFloatArray()
        var off = 0
        val tensorJson = tensors.joinToString(",") { t ->
            val s = """{"name":"${t.name}","shape":${t.shape.joinToString(",", "[", "]")},"offset":$off}"""
            off += t.data.size
            s
        }
        val data = ByteBuffer.allocate(all.size * 4).order(ByteOrder.LITTLE_ENDIAN).apply { asFloatBuffer().put(all) }.array()
        val crc = CRC32().apply { update(data) }.value + if (corruptCrc) 1 else 0
        val header = headerWithoutData.trimEnd().removeSuffix("}") +
            ""","tensors":[$tensorJson],"dataFloats":${all.size},"dataCrc32":$crc}"""
        val hb = header.toByteArray(Charsets.UTF_8)
        val out = ByteArrayOutputStream()
        out.write("TRCM".toByteArray())
        out.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putInt(1).putInt(hb.size).array())
        out.write(hb)
        while (out.size() % 4 != 0) out.write(0)
        out.write(data)
        return out.toByteArray()
    }
}

class TrcmEngineTest {
    private val rnd = Random(42)

    private fun rand(n: Int) = FloatArray(n) { rnd.nextFloat() * 2f - 1f }

    /** Straightforward reference convolution (PyTorch semantics). */
    private fun naiveConv(x: FloatArray, c: Int, h: Int, w: Int, wt: FloatArray, b: FloatArray, oc: Int, k: Int, s: Int, p: Int, g: Int): FloatArray {
        val ho = (h + 2 * p - k) / s + 1
        val wo = (w + 2 * p - k) / s + 1
        val icg = c / g
        val ocg = oc / g
        val out = FloatArray(oc * ho * wo)
        for (o in 0 until oc) for (oy in 0 until ho) for (ox in 0 until wo) {
            var acc = b[o].toDouble()
            val grp = o / ocg
            for (ci in 0 until icg) for (ky in 0 until k) for (kx in 0 until k) {
                val iy = oy * s + ky - p
                val ix = ox * s + kx - p
                if (iy in 0 until h && ix in 0 until w) {
                    acc += wt[((o * icg + ci) * k + ky) * k + kx] * x[((grp * icg + ci) * h + iy) * w + ix]
                }
            }
            out[(o * ho + oy) * wo + ox] = acc.toFloat()
        }
        return out
    }

    private fun header(c: Int, h: Int, w: Int, nodes: String, outputs: String, classes: Int = 1, params: Int = 0, features: Int = 0) = """
        {"format":"trcm","formatVersion":1,"name":"t",
         "inputs":{"image":{"width":$w,"height":$h,"channels":$c,"layout":"CHW","norm":{"type":"per_image"}},
                   "features":[${(0 until features).joinToString(",") { """{"name":"f$it","mean":0,"std":1}""" }}]},
         "classes":[${(0 until classes).joinToString(",") { """{"id":"c$it","name":"C$it","icon":"stars"}""" }}],
         "params":[${(0 until params).joinToString(",") { """{"name":"p$it","mean":0,"std":1}""" }}],
         "outputs":$outputs,
         "nodes":$nodes}
    """.trimIndent()

    @Test
    fun conv2dMatchesNaiveReference() {
        val cases = listOf(
            intArrayOf(3, 9, 7, 4, 3, 1, 1, 1),
            intArrayOf(4, 8, 8, 6, 3, 2, 1, 2),
            intArrayOf(2, 5, 6, 2, 1, 1, 0, 1),
            intArrayOf(6, 11, 9, 6, 5, 3, 2, 3),
            intArrayOf(1, 4, 4, 1, 3, 1, 0, 1),
        )
        for ((c, h, w, oc, k, s, p, g) in cases.map { it.toList() }.map { Octet(it) }) {
            val wt = rand(oc * (c / g) * k * k)
            val b = rand(oc)
            val x = rand(c * h * w)
            val ho = (h + 2 * p - k) / s + 1
            val wo = (w + 2 * p - k) / s + 1
            val nodes = """[{"op":"conv2d","name":"cv","inputs":["image"],"attrs":{"inChannels":$c,"outChannels":$oc,"kernel":$k,"stride":$s,"padding":$p,"groups":$g},"weights":["w","b"]},
                {"op":"flatten","name":"fl","inputs":["cv"]}]"""
            // Expose the conv output itself through both outputs (K = output size, P = 1).
            val hdr = header(c, h, w, nodes, """{"class_logits":"fl","params":"fl"}""", classes = oc * ho * wo, params = 1)
            val bytes = TrcmTestWriter.build(hdr, listOf(
                TrcmTestWriter.T("w", intArrayOf(oc, c / g, k, k), wt),
                TrcmTestWriter.T("b", intArrayOf(oc), b),
            ))
            val m = runCatching { TrcmModel.parse(bytes) }.getOrElse { throw AssertionError("parse failed for case c=$c k=$k s=$s p=$p g=$g", it) }
            val out = m.runPrepared(x, FloatArray(0)).logits
            val ref = naiveConv(x, c, h, w, wt, b, oc, k, s, p, g)
            assertArrayEquals("conv c=$c k=$k s=$s p=$p g=$g", ref, out, 1e-4f)
        }
    }

    private data class Octet(val c: Int, val h: Int, val w: Int, val oc: Int, val k: Int, val s: Int, val p: Int, val g: Int) {
        constructor(l: List<Int>) : this(l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7])
    }

    private fun smallNet(classes: Int, params: Int, features: Int, corrupt: Boolean = false, op: String = "relu"): ByteArray {
        val nodes = """[
            {"op":"conv2d","name":"c1","inputs":["image"],"attrs":{"inChannels":3,"outChannels":4,"kernel":3,"padding":1},"weights":["c1.w","c1.b"]},
            {"op":"batchnorm","name":"bn","inputs":["c1"],"attrs":{"eps":0.00001},"weights":["g","be","mu","var"]},
            {"op":"$op","name":"r1","inputs":["bn"]},
            {"op":"maxpool2d","name":"p1","inputs":["r1"],"attrs":{"kernel":2}},
            {"op":"avgpool2d","name":"p2","inputs":["p1"],"attrs":{"kernel":2,"stride":1}},
            {"op":"global_avgpool","name":"gap","inputs":["p2"]},
            {"op":"concat","name":"cat","inputs":["gap","features"]},
            {"op":"dense","name":"h","inputs":["cat"],"attrs":{"in":${4 + features},"out":5},"weights":["h.w","h.b"]},
            {"op":"tanh","name":"ht","inputs":["h"]},
            {"op":"dense","name":"logits","inputs":["ht"],"attrs":{"in":5,"out":$classes},"weights":["l.w","l.b"]},
            {"op":"dense","name":"params","inputs":["ht"],"attrs":{"in":5,"out":${classes * params}},"weights":["pp.w"]}
        ]"""
        val hdr = header(3, 6, 6, nodes, """{"class_logits":"logits","params":"params"}""", classes, params, features)
        val r = Random(7)
        fun t(name: String, vararg shape: Int) = TrcmTestWriter.T(name, shape, FloatArray(shape.fold(1) { a, b -> a * b }) { r.nextFloat() - 0.5f })
        return TrcmTestWriter.build(hdr, listOf(
            t("c1.w", 4, 3, 3, 3), t("c1.b", 4),
            t("g", 4), t("be", 4), t("mu", 4), TrcmTestWriter.T("var", intArrayOf(4), floatArrayOf(0.5f, 1f, 2f, 0.1f)),
            t("h.w", 5, 4 + features), t("h.b", 5), t("l.w", classes, 5), t("l.b", classes), t("pp.w", classes * params, 5),
        ), corruptCrc = corrupt)
    }

    @Test
    fun smallNetworkRunsAndDenormalizes() {
        val m = TrcmModel.parse(smallNet(classes = 3, params = 2, features = 2))
        assertEquals(3, m.classes.size)
        val out = m.runPrepared(rand(3 * 36), floatArrayOf(0.3f, -1f))
        assertEquals(3, out.logits.size)
        assertEquals(6, out.params.size)
        assertEquals(1f, out.probs.sum(), 1e-5f)
        assertEquals(out.params[1 * 2 + 1].toDouble(), out.param(1, "p1")!!, 1e-6)
        assertEquals(null, out.param(0, "nope"))
    }

    @Test
    fun corruptedCrcIsRejected() {
        try {
            TrcmModel.parse(smallNet(2, 1, 0, corrupt = true))
            fail("expected TrcmFormatException")
        } catch (e: TrcmFormatException) {
            assertTrue(e.message!!.contains("Контрольная"))
        }
    }

    @Test
    fun unknownOpIsRejected() {
        try {
            TrcmModel.parse(smallNet(2, 1, 0, op = "gelu"))
            fail("expected TrcmFormatException")
        } catch (e: TrcmFormatException) {
            assertTrue(e.message!!.contains("gelu"))
        }
    }

    @Test
    fun garbageIsRejected() {
        for (bytes in listOf(ByteArray(0), "TRCM".toByteArray(), "NOPE00000000".toByteArray(), smallNet(2, 1, 0).copyOf(60))) {
            try {
                TrcmModel.parse(bytes)
                fail("expected TrcmFormatException")
            } catch (_: TrcmFormatException) {
            }
        }
    }

    @Test
    fun perImageNormalizationAndFeatureImputation() {
        val m = TrcmModel.parse(smallNet(classes = 2, params = 1, features = 2))
        val argb = IntArray(36) { if (it % 2 == 0) 0xFFFFFFFF.toInt() else 0xFF000000.toInt() }
        val x = m.prepareImage(argb)
        val mean = x.average()
        val std = sqrt(x.map { (it - mean) * (it - mean) }.average())
        assertEquals(0.0, mean, 1e-5)
        assertEquals(1.0, std, 1e-4)
        val f = m.prepareFeatures(mapOf("f0" to Double.NaN, "f1" to 100.0))
        assertEquals(0f, f[0])
        assertEquals(6f, f[1])
        assertTrue(abs(m.prepareFeatures(mapOf("f0" to null))[0]) == 0f)
    }
}
