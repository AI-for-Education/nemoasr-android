package com.example.nemoasrdemo

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

class OnnxAsrEngine(private val context: Context) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val idToToken: Map<Int, String>
    private val blankId: Int

    private val sampleRate = 16000
    private val featureDim = 80
    private val windowSize = sampleRate * 25 / 1000
    private val hopSize = sampleRate * 10 / 1000
    private val fftSize = 512
    private val fftBins = fftSize / 2 + 1
    private val hannWindow = FloatArray(windowSize) { i ->
        (0.5 - 0.5 * cos(2.0 * Math.PI * i / (windowSize - 1))).toFloat()
    }
    private val melFilterbank = createMelFilterbank()

    init {
        val modelFile = copyAssetIfNeeded("asr/model.int8.onnx")
        idToToken = loadTokens("asr/tokens.txt")
        blankId = idToToken.entries.firstOrNull { it.value == "<blk>" }?.key ?: (idToToken.size - 1)

        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(1)
            setInterOpNumThreads(1)
        }
        session = environment.createSession(modelFile.absolutePath, options)
    }

    fun transcribeAsset(assetPath: String): String {
        val wave = SimpleWaveReader.read(context.assets.open(assetPath).readBytes())
        return transcribe(wave.samples, wave.sampleRate)
    }

    fun transcribe(samples: FloatArray, inputSampleRate: Int): String {
        val resampled = if (inputSampleRate == sampleRate) {
            samples
        } else {
            resampleLinear(samples, inputSampleRate, sampleRate)
        }

        val features = extractFeatures(resampled)
        val numFrames = features[0].size
        val flattened = FloatArray(featureDim * numFrames)
        for (m in 0 until featureDim) {
            System.arraycopy(features[m], 0, flattened, m * numFrames, numFrames)
        }

        val inputTensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(flattened),
            longArrayOf(1, featureDim.toLong(), numFrames.toLong())
        )
        val lengthTensor = OnnxTensor.createTensor(
            environment,
            LongBuffer.wrap(longArrayOf(numFrames.toLong())),
            longArrayOf(1)
        )

        inputTensor.use { audio ->
            lengthTensor.use { length ->
                session.run(
                    mapOf(
                        "audio_signal" to audio,
                        "length" to length
                    )
                ).use { outputs ->
                    @Suppress("UNCHECKED_CAST")
                    val logits = outputs[0].value as Array<Array<FloatArray>>
                    return decodeGreedy(logits[0])
                }
            }
        }
    }

    override fun close() {
        session.close()
    }

    private fun decodeGreedy(logits: Array<FloatArray>): String {
        val tokenIds = ArrayList<Int>(logits.size)
        var previous = -1
        for (frame in logits) {
            var bestId = 0
            var bestValue = Float.NEGATIVE_INFINITY
            for (i in frame.indices) {
                if (frame[i] > bestValue) {
                    bestValue = frame[i]
                    bestId = i
                }
            }
            if (bestId != previous && bestId != blankId) {
                tokenIds += bestId
            }
            previous = bestId
        }

        val builder = StringBuilder()
        for (id in tokenIds) {
            builder.append(idToToken[id].orEmpty())
        }
        return builder.toString().replace('▁', ' ').trim()
    }

    private fun extractFeatures(samples: FloatArray): Array<FloatArray> {
        val frameCount = if (samples.size < windowSize) 1 else 1 + (samples.size - windowSize) / hopSize
        val features = Array(featureDim) { FloatArray(frameCount) }
        val frame = FloatArray(fftSize)
        val real = FloatArray(fftSize)
        val imag = FloatArray(fftSize)
        val power = FloatArray(fftBins)

        for (frameIndex in 0 until frameCount) {
            val start = frameIndex * hopSize
            java.util.Arrays.fill(frame, 0f)
            for (i in 0 until windowSize) {
                val sampleIndex = start + i
                val sample = if (sampleIndex < samples.size) samples[sampleIndex] else 0f
                frame[i] = sample * hannWindow[i]
            }

            System.arraycopy(frame, 0, real, 0, fftSize)
            java.util.Arrays.fill(imag, 0f)
            fft(real, imag)

            for (i in 0 until fftBins) {
                power[i] = real[i] * real[i] + imag[i] * imag[i]
            }

            for (m in 0 until featureDim) {
                var energy = 0f
                val filter = melFilterbank[m]
                for (k in 0 until fftBins) {
                    energy += filter[k] * power[k]
                }
                features[m][frameIndex] = ln(max(energy, 1e-10f))
            }
        }

        normalizePerFeature(features)
        return features
    }

    private fun normalizePerFeature(features: Array<FloatArray>) {
        for (m in features.indices) {
            val values = features[m]
            var sum = 0f
            for (v in values) {
                sum += v
            }
            val mean = sum / values.size

            var variance = 0f
            for (v in values) {
                val diff = v - mean
                variance += diff * diff
            }
            val std = max(kotlin.math.sqrt(variance / values.size), 1e-5f)
            for (i in values.indices) {
                values[i] = (values[i] - mean) / std
            }
        }
    }

    private fun createMelFilterbank(): Array<FloatArray> {
        val filters = Array(featureDim) { FloatArray(fftBins) }
        val melMin = hzToMel(0.0)
        val melMax = hzToMel(sampleRate / 2.0)
        val melPoints = DoubleArray(featureDim + 2) { i ->
            melMin + (melMax - melMin) * i / (featureDim + 1)
        }
        val hzPoints = DoubleArray(featureDim + 2) { i -> melToHz(melPoints[i]) }
        val binPoints = IntArray(featureDim + 2) { i ->
            (((fftSize + 1) * hzPoints[i] / sampleRate).toInt()).coerceIn(0, fftBins - 1)
        }

        for (m in 1..featureDim) {
            val left = binPoints[m - 1]
            val center = max(binPoints[m], left + 1)
            val right = max(binPoints[m + 1], center + 1)
            for (k in left until min(center, fftBins)) {
                filters[m - 1][k] = (k - left).toFloat() / (center - left).toFloat()
            }
            for (k in center until min(right, fftBins)) {
                filters[m - 1][k] = (right - k).toFloat() / (right - center).toFloat()
            }

            val norm = (2.0 / max(hzPoints[m + 1] - hzPoints[m - 1], 1e-10)).toFloat()
            for (k in 0 until fftBins) {
                filters[m - 1][k] *= norm
            }
        }

        return filters
    }

    private fun hzToMel(hz: Double): Double = 2595.0 * kotlin.math.log10(1.0 + hz / 700.0)

    private fun melToHz(mel: Double): Double = 700.0 * (10.0.pow(mel / 2595.0) - 1.0)

    private fun copyAssetIfNeeded(assetPath: String): File {
        val target = File(context.filesDir, assetPath)
        target.parentFile?.mkdirs()
        val temp = File(target.absolutePath + ".tmp")
        context.assets.open(assetPath).use { input ->
            temp.outputStream().use { output -> input.copyTo(output) }
        }
        if (target.exists()) {
            target.delete()
        }
        check(temp.renameTo(target)) { "Failed to replace asset copy for $assetPath" }
        return target
    }

    private fun loadTokens(assetPath: String): Map<Int, String> {
        val tokens = LinkedHashMap<Int, String>()
        context.assets.open(assetPath).bufferedReader().useLines { lines ->
            lines.forEach { line ->
                val trimmed = line.trim()
                if (trimmed.isEmpty()) return@forEach
                val split = trimmed.lastIndexOf(' ')
                require(split > 0) { "Invalid token line: $line" }
                val token = trimmed.substring(0, split)
                val id = trimmed.substring(split + 1).toInt()
                tokens[id] = token
            }
        }
        return tokens
    }

    private fun resampleLinear(input: FloatArray, fromRate: Int, toRate: Int): FloatArray {
        val outputSize = ((input.size.toLong() * toRate) / fromRate).toInt().coerceAtLeast(1)
        val output = FloatArray(outputSize)
        val scale = fromRate.toDouble() / toRate.toDouble()
        for (i in output.indices) {
            val position = i * scale
            val left = position.toInt().coerceIn(0, input.lastIndex)
            val right = min(left + 1, input.lastIndex)
            val alpha = (position - left).toFloat()
            output[i] = input[left] * (1f - alpha) + input[right] * alpha
        }
        return output
    }

    private fun fft(real: FloatArray, imag: FloatArray) {
        val n = real.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                val tr = real[i]
                real[i] = real[j]
                real[j] = tr
                val ti = imag[i]
                imag[i] = imag[j]
                imag[j] = ti
            }
        }

        var len = 2
        while (len <= n) {
            val angle = (-2.0 * Math.PI / len).toFloat()
            val wLenCos = cos(angle.toDouble()).toFloat()
            val wLenSin = kotlin.math.sin(angle)
            var i = 0
            while (i < n) {
                var wCos = 1f
                var wSin = 0f
                for (k in 0 until len / 2) {
                    val uReal = real[i + k]
                    val uImag = imag[i + k]
                    val vReal = real[i + k + len / 2] * wCos - imag[i + k + len / 2] * wSin
                    val vImag = real[i + k + len / 2] * wSin + imag[i + k + len / 2] * wCos

                    real[i + k] = uReal + vReal
                    imag[i + k] = uImag + vImag
                    real[i + k + len / 2] = uReal - vReal
                    imag[i + k + len / 2] = uImag - vImag

                    val nextCos = wCos * wLenCos - wSin * wLenSin
                    val nextSin = wCos * wLenSin + wSin * wLenCos
                    wCos = nextCos
                    wSin = nextSin
                }
                i += len
            }
            len = len shl 1
        }
    }
}
