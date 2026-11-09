package com.therealcam.ai.engine

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.zip.CRC32
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

class TrcmFormatException(message: String) : Exception(message)

data class ModelClass(val id: String, val name: String, val icon: String)
data class NormSpec(val name: String, val mean: Double, val std: Double)

class ModelOutput(val logits: FloatArray, val params: FloatArray, private val model: TrcmModel) {
    val probs: FloatArray = softmax(logits)

    /** De-normalized value of parameter [name] for class [k], or null if the model has no such parameter. */
    fun param(k: Int, name: String): Double? {
        val p = model.params.indexOfFirst { it.name == name }
        if (p < 0 || k !in model.classes.indices) return null
        val spec = model.params[p]
        return params[k * model.params.size + p] * spec.std + spec.mean
    }

    fun paramsOf(k: Int): Map<String, Double> = model.params.associate { it.name to param(k, it.name)!! }

    /** Class indices sorted by probability, best first. */
    fun ranking(): List<Int> = probs.indices.sortedByDescending { probs[it] }

    companion object {
        fun softmax(x: FloatArray): FloatArray {
            if (x.isEmpty()) return x
            val m = x.max()
            val e = FloatArray(x.size) { exp((x[it] - m).toDouble()).toFloat() }
            val s = e.sum()
            for (i in e.indices) e[i] /= s
            return e
        }
    }
}

/**
 * A `.trcm` model (docs/MODEL_FORMAT.md): JSON header + float32 weights, executed by a small
 * interpreter. Pure Kotlin; validated on load so a broken or foreign file fails early.
 */
class TrcmModel private constructor(
    val header: JsonObject,
    val name: String,
    val description: String,
    val classes: List<ModelClass>,
    val params: List<NormSpec>,
    val features: List<NormSpec>,
    private val imageW: Int,
    private val imageH: Int,
    private val imageC: Int,
    private val normType: String,
    private val normMean: FloatArray?,
    private val normStd: FloatArray?,
    private val nodes: List<Node>,
    private val outLogits: String,
    private val outParams: String,
    val sha256: String,
    val sizeBytes: Int,
) {
    val trainedSamples: Int? get() = header["training"]?.objOrNull()?.get("samples")?.numOrNull()?.toInt()
    val createdUtc: String? get() = header["createdUtc"]?.strOrNull()

    /** Runs the model on a 64×64 ARGB thumbnail and raw (un-normalized) features; null = missing. */
    fun run(thumbArgb: IntArray, rawFeatures: Map<String, Double?>): ModelOutput {
        require(thumbArgb.size == imageW * imageH) { "thumbnail must be ${imageW}x$imageH" }
        return runPrepared(prepareImage(thumbArgb), prepareFeatures(rawFeatures))
    }

    fun prepareImage(argb: IntArray): FloatArray {
        val plane = imageW * imageH
        val x = FloatArray(imageC * plane)
        for (i in 0 until plane) {
            val p = argb[i]
            x[i] = (p shr 16 and 0xFF) / 255f
            if (imageC > 1) x[plane + i] = (p shr 8 and 0xFF) / 255f
            if (imageC > 2) x[2 * plane + i] = (p and 0xFF) / 255f
        }
        if (normType == "per_image") {
            var s = 0.0
            for (v in x) s += v
            val m = s / x.size
            var s2 = 0.0
            for (v in x) s2 += (v - m) * (v - m)
            val sd = max(sqrt(s2 / x.size), 1.0 / sqrt(x.size.toDouble()))
            for (i in x.indices) x[i] = ((x[i] - m) / sd).toFloat()
        } else {
            val mean = normMean!!
            val std = normStd!!
            for (c in 0 until imageC) for (i in 0 until plane) {
                val k = c * plane + i
                x[k] = (x[k] - mean[c]) / std[c]
            }
        }
        return x
    }

    fun prepareFeatures(raw: Map<String, Double?>): FloatArray = FloatArray(features.size) { i ->
        val f = features[i]
        val v = raw[f.name]
        if (v == null || !v.isFinite()) 0f
        else {
            val sd = if (f.std <= 1e-6) 1.0 else f.std
            ((v - f.mean) / sd).coerceIn(-6.0, 6.0).toFloat()
        }
    }

    fun runPrepared(image: FloatArray, feats: FloatArray): ModelOutput {
        val values = HashMap<String, Tensor>(nodes.size * 2)
        values["image"] = Tensor(intArrayOf(imageC, imageH, imageW), image)
        values["features"] = Tensor(intArrayOf(feats.size), feats)
        for (n in nodes) values[n.name] = n.eval(values)
        val logits = values.getValue(outLogits).data
        val p = values.getValue(outParams).data
        return ModelOutput(logits.copyOf(), p.copyOf(), this)
    }

    companion object {
        private const val MAX_HEADER = 1 shl 20
        private const val MAX_DATA_FLOATS = (64 shl 20) / 4
        private const val MAX_NODES = 256
        private val json = Json { ignoreUnknownKeys = true }

        fun load(file: File): TrcmModel {
            if (file.length() > 12L + MAX_HEADER + 3 + MAX_DATA_FLOATS * 4L) throw TrcmFormatException("Файл модели слишком большой")
            return parse(file.readBytes())
        }

        fun parse(bytes: ByteArray): TrcmModel {
            fail(bytes.size >= 12, "Файл слишком короткий")
            fail(String(bytes, 0, 4, Charsets.US_ASCII) == "TRCM", "Это не файл модели TheRealCam (нет сигнатуры TRCM)")
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val version = bb.getInt(4)
            fail(version == 1, "Неподдерживаемая версия формата: $version")
            val hLen = bb.getInt(8)
            fail(hLen in 2..MAX_HEADER && 12L + hLen <= bytes.size, "Повреждён заголовок")
            val header = runCatching { json.parseToJsonElement(String(bytes, 12, hLen, Charsets.UTF_8)).jsonObject }
                .getOrElse { throw TrcmFormatException("Заголовок не JSON: ${it.message}") }
            val dataStart = (12 + hLen + 3) and 3.inv()
            val dataFloats = header.num("dataFloats").toLong()
            fail(dataFloats in 0..MAX_DATA_FLOATS, "Слишком много весов")
            fail(dataStart + dataFloats * 4 <= bytes.size, "Файл обрезан")
            val crc = CRC32().apply { update(bytes, dataStart, (dataFloats * 4).toInt()) }.value
            header["dataCrc32"]?.numOrNull()?.let { fail(it.toLong() == crc, "Контрольная сумма весов не совпадает — файл повреждён") }
            val data = FloatArray(dataFloats.toInt())
            ByteBuffer.wrap(bytes, dataStart, (dataFloats * 4).toInt()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(data)

            val tensors = HashMap<String, Tensor>()
            for (t in header.arr("tensors")) {
                val o = t.jsonObject
                val name = o.str("name")
                val shape = o.arr("shape").map { it.jsonPrimitive.doubleOrNull?.toInt() ?: throw TrcmFormatException("bad shape") }.toIntArray()
                fail(shape.all { it > 0 }, "Неверная форма тензора $name")
                val count = shape.fold(1L) { a, b -> a * b }
                val off = o.num("offset").toLong()
                fail(off >= 0 && off + count <= dataFloats, "Тензор $name выходит за пределы данных")
                tensors[name] = Tensor(shape, data.copyOfRange(off.toInt(), (off + count).toInt()))
            }

            val inputs = header.obj("inputs")
            val img = inputs.obj("image")
            val norm = img["norm"]?.objOrNull()
            val normType = norm?.get("type")?.strOrNull() ?: "per_image"
            val imageC = img["channels"]?.numOrNull()?.toInt() ?: 3
            fail(normType == "per_image" || normType == "channel", "Неизвестная нормализация изображения: $normType")
            val normMean = if (normType == "channel") norm!!.arr("mean").map { it.jsonPrimitive.doubleOrNull!!.toFloat() }.toFloatArray() else null
            val normStd = if (normType == "channel") norm!!.arr("std").map { it.jsonPrimitive.doubleOrNull!!.toFloat() }.toFloatArray() else null
            if (normType == "channel") fail(normMean!!.size == imageC && normStd!!.size == imageC && normStd.all { it != 0f }, "Неверная нормализация каналов")

            val features = inputs["features"]?.jsonArray.orEmpty().map { it.jsonObject.let { o -> NormSpec(o.str("name"), o.num("mean"), o.num("std")) } }
            val classes = header.arr("classes").map {
                val o = it.jsonObject
                ModelClass(o.str("id"), o["name"]?.strOrNull() ?: o.str("id"), o["icon"]?.strOrNull() ?: "other")
            }
            fail(classes.isNotEmpty(), "В модели нет классов")
            val params = header["params"]?.jsonArray.orEmpty().map { it.jsonObject.let { o -> NormSpec(o.str("name"), o.num("mean"), o.num("std")) } }

            val nodesJson = header.arr("nodes")
            fail(nodesJson.size in 1..MAX_NODES, "Неверное число узлов графа")
            val known = hashSetOf("image", "features")
            val nodes = nodesJson.map { el ->
                val o = el.jsonObject
                val n = Node.create(o, tensors)
                n.inputs.forEach { fail(it in known, "Узел ${n.name}: неизвестный вход $it") }
                fail(known.add(n.name), "Повторяющееся имя узла ${n.name}")
                n
            }
            val outputs = header.obj("outputs")
            val outLogits = outputs.str("class_logits")
            val outParams = outputs.str("params")
            fail(outLogits in known && outParams in known, "Выходы модели не найдены в графе")

            val model = TrcmModel(
                header = header,
                name = header["name"]?.strOrNull() ?: "model",
                description = header["description"]?.strOrNull() ?: "",
                classes = classes, params = params, features = features,
                imageW = img["width"]?.numOrNull()?.toInt() ?: 64,
                imageH = img["height"]?.numOrNull()?.toInt() ?: 64,
                imageC = imageC, normType = normType, normMean = normMean, normStd = normStd,
                nodes = nodes, outLogits = outLogits, outParams = outParams,
                sha256 = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) },
                sizeBytes = bytes.size,
            )
            // Dry run: catches shape mismatches before the model is ever used in the camera.
            val out = try {
                model.runPrepared(FloatArray(imageC * model.imageW * model.imageH), FloatArray(features.size))
            } catch (e: TrcmFormatException) {
                throw e
            } catch (e: Exception) {
                throw TrcmFormatException("Граф модели не выполняется: ${e.message}")
            }
            fail(out.logits.size == classes.size, "Выход class_logits (${out.logits.size}) не совпадает с числом классов (${classes.size})")
            fail(out.params.size == classes.size * params.size, "Выход params не совпадает с K×P")
            return model
        }

        internal fun fail(ok: Boolean, msg: String) {
            if (!ok) throw TrcmFormatException(msg)
        }

        internal fun JsonObject.str(k: String): String = this[k]?.strOrNull() ?: throw TrcmFormatException("Нет поля '$k'")
        internal fun JsonObject.num(k: String): Double = this[k]?.numOrNull() ?: throw TrcmFormatException("Нет числового поля '$k'")
        internal fun JsonObject.arr(k: String): JsonArray = (this[k] as? JsonArray) ?: throw TrcmFormatException("Нет массива '$k'")
        internal fun JsonObject.obj(k: String): JsonObject = (this[k] as? JsonObject) ?: throw TrcmFormatException("Нет объекта '$k'")
        internal fun JsonElement.strOrNull(): String? = (this as? JsonPrimitive)?.takeIf { it.isString }?.content
        internal fun JsonElement.numOrNull(): Double? = (this as? JsonPrimitive)?.takeIf { !it.isString && it !is JsonNull }?.doubleOrNull
        internal fun JsonElement.objOrNull(): JsonObject? = this as? JsonObject
        internal fun JsonElement.boolOrNull(): Boolean? = (this as? JsonPrimitive)?.booleanOrNull
    }
}
