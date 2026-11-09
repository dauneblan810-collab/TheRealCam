package com.therealcam.ai.engine

import com.therealcam.ai.engine.TrcmModel.Companion.fail
import com.therealcam.ai.engine.TrcmModel.Companion.numOrNull
import com.therealcam.ai.engine.TrcmModel.Companion.str
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.math.tanh

/** [C,H,W] (batch 1) or a vector [N]. Data is row-major in that order. */
class Tensor(val shape: IntArray, val data: FloatArray) {
    val is3d get() = shape.size == 3
    val channels get() = shape[0]
}

abstract class Node(val name: String, val inputs: List<String>) {
    abstract fun eval(values: Map<String, Tensor>): Tensor

    protected fun input(values: Map<String, Tensor>, i: Int = 0): Tensor = values.getValue(inputs[i])

    companion object {
        fun create(o: JsonObject, tensors: Map<String, Tensor>): Node {
            val op = o.str("op")
            val name = o.str("name")
            val inputs = (o["inputs"] as? JsonArray)?.map { it.jsonPrimitive.content }.orEmpty()
            val attrs = o["attrs"] as? JsonObject ?: JsonObject(emptyMap())
            val weights = (o["weights"] as? JsonArray)?.map {
                val t = it.jsonPrimitive.content
                tensors[t] ?: throw TrcmFormatException("Узел $name: нет тензора $t")
            }.orEmpty()

            fun int(k: String, def: Int? = null): Int =
                attrs[k]?.numOrNull()?.toInt() ?: def ?: throw TrcmFormatException("Узел $name: нет атрибута $k")

            fun float(k: String, def: Float): Float = attrs[k]?.numOrNull()?.toFloat() ?: def

            fun needInputs(n: Int) = fail(inputs.size == n, "Узел $name ($op): ожидается входов: $n")
            return when (op) {
                "conv2d" -> {
                    needInputs(1)
                    val k = int("kernel")
                    Conv2d(name, inputs, int("inChannels"), int("outChannels"), k, int("stride", 1), int("padding", 0), int("groups", 1), weights)
                }
                "dense" -> { needInputs(1); Dense(name, inputs, int("in"), int("out"), weights) }
                "relu" -> { needInputs(1); Elementwise(name, inputs) { max(it, 0f) } }
                "leaky_relu" -> {
                    needInputs(1)
                    val a = float("alpha", 0.01f)
                    Elementwise(name, inputs) { if (it >= 0f) it else it * a }
                }
                "sigmoid" -> { needInputs(1); Elementwise(name, inputs) { (1.0 / (1.0 + exp(-it.toDouble()))).toFloat() } }
                "tanh" -> { needInputs(1); Elementwise(name, inputs) { tanh(it.toDouble()).toFloat() } }
                "maxpool2d", "avgpool2d" -> {
                    needInputs(1)
                    val k = int("kernel")
                    Pool2d(name, inputs, k, int("stride", k), op == "maxpool2d")
                }
                "global_avgpool" -> { needInputs(1); GlobalAvgPool(name, inputs) }
                "channel_affine" -> {
                    needInputs(1)
                    fail(weights.size == 2 && weights[0].data.size == weights[1].data.size, "Узел $name: нужны scale и bias")
                    ChannelAffine(name, inputs, weights[0].data, weights[1].data)
                }
                "batchnorm" -> {
                    needInputs(1)
                    fail(weights.size == 4 && weights.all { it.data.size == weights[0].data.size }, "Узел $name: нужны gamma, beta, mean, var")
                    val eps = float("eps", 1e-5f)
                    val (g, b, m, v) = weights.map { it.data }
                    val scale = FloatArray(g.size) { (g[it] / sqrt(v[it] + eps)) }
                    val bias = FloatArray(g.size) { b[it] - m[it] * scale[it] }
                    ChannelAffine(name, inputs, scale, bias)
                }
                "flatten" -> { needInputs(1); Flatten(name, inputs) }
                "concat" -> { fail(inputs.isNotEmpty(), "Узел $name: нет входов"); Concat(name, inputs) }
                "add" -> { needInputs(2); Add(name, inputs) }
                "dropout", "identity" -> { needInputs(1); Identity(name, inputs) }
                else -> throw TrcmFormatException("Неизвестная операция '$op' в узле $name")
            }
        }
    }
}

private class Conv2d(
    name: String, inputs: List<String>,
    val inC: Int, val outC: Int, val k: Int, val s: Int, val p: Int, val groups: Int,
    weights: List<Tensor>,
) : Node(name, inputs) {
    val w: FloatArray
    val b: FloatArray?

    init {
        fail(k > 0 && s > 0 && p >= 0 && groups > 0 && inC % groups == 0 && outC % groups == 0, "Узел $name: неверные атрибуты conv2d")
        fail(weights.size in 1..2, "Узел $name: нужны weight и (опц.) bias")
        val ws = weights[0].shape
        fail(ws.contentEquals(intArrayOf(outC, inC / groups, k, k)), "Узел $name: форма весов ${ws.contentToString()}")
        w = weights[0].data
        b = weights.getOrNull(1)?.data?.also { fail(it.size == outC, "Узел $name: размер bias") }
    }

    override fun eval(values: Map<String, Tensor>): Tensor {
        val x = input(values)
        fail(x.is3d && x.channels == inC, "Узел $name: ожидается вход [$inC,H,W]")
        val h = x.shape[1]
        val wd = x.shape[2]
        val ho = (h + 2 * p - k) / s + 1
        val wo = (wd + 2 * p - k) / s + 1
        fail(ho > 0 && wo > 0, "Узел $name: слишком маленький вход")
        val out = FloatArray(outC * ho * wo)
        val icg = inC / groups
        val ocg = outC / groups
        val xd = x.data
        for (co in 0 until outC) {
            val outBase = co * ho * wo
            b?.let { bias -> java.util.Arrays.fill(out, outBase, outBase + ho * wo, bias[co]) }
            val g = co / ocg
            for (ci in 0 until icg) {
                val inBase = (g * icg + ci) * h * wd
                for (ky in 0 until k) {
                    val oyMin = max(0, Math.floorDiv(p - ky + s - 1, s))
                    val oyMax = min(ho - 1, Math.floorDiv(h - 1 + p - ky, s))
                    if (oyMin > oyMax) continue
                    for (kx in 0 until k) {
                        val wv = w[((co * icg + ci) * k + ky) * k + kx]
                        if (wv == 0f) continue
                        val oxMin = max(0, Math.floorDiv(p - kx + s - 1, s))
                        val oxMax = min(wo - 1, Math.floorDiv(wd - 1 + p - kx, s))
                        if (oxMin > oxMax) continue
                        for (oy in oyMin..oyMax) {
                            val orow = outBase + oy * wo
                            val irow = inBase + (oy * s + ky - p) * wd + kx - p
                            var ox = oxMin
                            while (ox <= oxMax) {
                                out[orow + ox] += wv * xd[irow + ox * s]
                                ox++
                            }
                        }
                    }
                }
            }
        }
        return Tensor(intArrayOf(outC, ho, wo), out)
    }
}

private class Dense(name: String, inputs: List<String>, val inN: Int, val outN: Int, weights: List<Tensor>) : Node(name, inputs) {
    val w: FloatArray
    val b: FloatArray?

    init {
        fail(weights.size in 1..2, "Узел $name: нужны weight и (опц.) bias")
        fail(weights[0].shape.contentEquals(intArrayOf(outN, inN)), "Узел $name: форма весов ${weights[0].shape.contentToString()}")
        w = weights[0].data
        b = weights.getOrNull(1)?.data?.also { fail(it.size == outN, "Узел $name: размер bias") }
    }

    override fun eval(values: Map<String, Tensor>): Tensor {
        val x = input(values).data
        fail(x.size == inN, "Узел $name: вход ${x.size}, ожидается $inN")
        val out = FloatArray(outN)
        for (o in 0 until outN) {
            var acc = b?.get(o)?.toDouble() ?: 0.0
            val base = o * inN
            for (i in 0 until inN) acc += w[base + i] * x[i]
            out[o] = acc.toFloat()
        }
        return Tensor(intArrayOf(outN), out)
    }
}

private class Elementwise(name: String, inputs: List<String>, val f: (Float) -> Float) : Node(name, inputs) {
    override fun eval(values: Map<String, Tensor>): Tensor {
        val x = input(values)
        return Tensor(x.shape, FloatArray(x.data.size) { f(x.data[it]) })
    }
}

private class Pool2d(name: String, inputs: List<String>, val k: Int, val s: Int, val isMax: Boolean) : Node(name, inputs) {
    init {
        fail(k > 0 && s > 0, "Узел $name: неверные атрибуты пула")
    }

    override fun eval(values: Map<String, Tensor>): Tensor {
        val x = input(values)
        fail(x.is3d, "Узел $name: ожидается [C,H,W]")
        val (c, h, w) = Triple(x.shape[0], x.shape[1], x.shape[2])
        val ho = (h - k) / s + 1
        val wo = (w - k) / s + 1
        fail(ho > 0 && wo > 0, "Узел $name: слишком маленький вход")
        val out = FloatArray(c * ho * wo)
        val inv = 1f / (k * k)
        for (ch in 0 until c) for (oy in 0 until ho) for (ox in 0 until wo) {
            var acc = if (isMax) Float.NEGATIVE_INFINITY else 0f
            for (ky in 0 until k) {
                val row = ch * h * w + (oy * s + ky) * w + ox * s
                for (kx in 0 until k) {
                    val v = x.data[row + kx]
                    if (isMax) { if (v > acc) acc = v } else acc += v
                }
            }
            out[(ch * ho + oy) * wo + ox] = if (isMax) acc else acc * inv
        }
        return Tensor(intArrayOf(c, ho, wo), out)
    }
}

private class GlobalAvgPool(name: String, inputs: List<String>) : Node(name, inputs) {
    override fun eval(values: Map<String, Tensor>): Tensor {
        val x = input(values)
        fail(x.is3d, "Узел $name: ожидается [C,H,W]")
        val c = x.shape[0]
        val plane = x.shape[1] * x.shape[2]
        return Tensor(intArrayOf(c), FloatArray(c) { ch ->
            var s = 0.0
            for (i in 0 until plane) s += x.data[ch * plane + i]
            (s / plane).toFloat()
        })
    }
}

private class ChannelAffine(name: String, inputs: List<String>, val scale: FloatArray, val bias: FloatArray) : Node(name, inputs) {
    override fun eval(values: Map<String, Tensor>): Tensor {
        val x = input(values)
        val c = x.shape[0]
        fail(c == scale.size, "Узел $name: каналов $c, параметров ${scale.size}")
        val plane = x.data.size / c
        val out = FloatArray(x.data.size)
        for (ch in 0 until c) for (i in 0 until plane) {
            val k = ch * plane + i
            out[k] = x.data[k] * scale[ch] + bias[ch]
        }
        return Tensor(x.shape, out)
    }
}

private class Flatten(name: String, inputs: List<String>) : Node(name, inputs) {
    override fun eval(values: Map<String, Tensor>): Tensor {
        val x = input(values)
        return Tensor(intArrayOf(x.data.size), x.data)
    }
}

private class Concat(name: String, inputs: List<String>) : Node(name, inputs) {
    override fun eval(values: Map<String, Tensor>): Tensor {
        val parts = inputs.map { values.getValue(it).data }
        val out = FloatArray(parts.sumOf { it.size })
        var off = 0
        for (p in parts) {
            System.arraycopy(p, 0, out, off, p.size)
            off += p.size
        }
        return Tensor(intArrayOf(out.size), out)
    }
}

private class Add(name: String, inputs: List<String>) : Node(name, inputs) {
    override fun eval(values: Map<String, Tensor>): Tensor {
        val a = input(values, 0)
        val b = input(values, 1)
        fail(a.shape.contentEquals(b.shape), "Узел $name: формы не совпадают")
        return Tensor(a.shape, FloatArray(a.data.size) { a.data[it] + b.data[it] })
    }
}

private class Identity(name: String, inputs: List<String>) : Node(name, inputs) {
    override fun eval(values: Map<String, Tensor>): Tensor = input(values)
}
