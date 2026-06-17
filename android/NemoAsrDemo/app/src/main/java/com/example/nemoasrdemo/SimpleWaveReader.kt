package com.example.nemoasrdemo

import java.io.ByteArrayInputStream
import java.io.DataInputStream

data class WaveData(
    val sampleRate: Int,
    val samples: FloatArray
)

object SimpleWaveReader {
    fun read(input: ByteArray): WaveData {
        val dis = DataInputStream(ByteArrayInputStream(input))

        fun readString(n: Int): String {
            val b = ByteArray(n)
            dis.readFully(b)
            return String(b)
        }

        fun readLEInt(): Int {
            val b0 = dis.readUnsignedByte()
            val b1 = dis.readUnsignedByte()
            val b2 = dis.readUnsignedByte()
            val b3 = dis.readUnsignedByte()
            return b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
        }

        fun readLEShort(): Int {
            val b0 = dis.readUnsignedByte()
            val b1 = dis.readUnsignedByte()
            return b0 or (b1 shl 8)
        }

        require(readString(4) == "RIFF")
        readLEInt()
        require(readString(4) == "WAVE")

        var sampleRate = 16000
        var channels = 1
        var bitsPerSample = 16
        var pcmData: ByteArray? = null

        while (dis.available() > 0) {
            val chunkId = readString(4)
            val chunkSize = readLEInt()

            when (chunkId) {
                "fmt " -> {
                    val audioFormat = readLEShort()
                    channels = readLEShort()
                    sampleRate = readLEInt()
                    readLEInt()
                    readLEShort()
                    bitsPerSample = readLEShort()

                    if (chunkSize > 16) {
                        dis.skipBytes(chunkSize - 16)
                    }

                    require(audioFormat == 1) { "Only PCM WAV supported" }
                }

                "data" -> {
                    pcmData = ByteArray(chunkSize)
                    dis.readFully(pcmData)
                }

                else -> dis.skipBytes(chunkSize)
            }

            if (pcmData != null) break
        }

        require(channels == 1) { "Only mono WAV supported" }
        require(bitsPerSample == 16) { "Only 16-bit WAV supported" }
        require(pcmData != null) { "No data chunk found in WAV" }

        val numSamples = pcmData.size / 2
        val samples = FloatArray(numSamples)

        var i = 0
        var j = 0
        while (i < pcmData.size) {
            val lo = pcmData[i].toInt() and 0xff
            val hi = pcmData[i + 1].toInt()
            val value = (hi shl 8) or lo
            samples[j] = value.toShort() / 32768.0f
            i += 2
            j += 1
        }

        return WaveData(sampleRate, samples)
    }
}

object WaveDebugWriter {
    fun write16kMono16bit(samples: FloatArray, sampleRate: Int): ByteArray {
        val pcm = ByteArray(samples.size * 2)

        var p = 0
        for (s in samples) {
            val clamped = s.coerceIn(-1.0f, 1.0f)
            val v = (clamped * 32767.0f).toInt().toShort()
            pcm[p++] = (v.toInt() and 0xff).toByte()
            pcm[p++] = ((v.toInt() shr 8) and 0xff).toByte()
        }

        val dataSize = pcm.size
        val riffSize = 36 + dataSize
        val out = ByteArray(44 + dataSize)

        fun putAscii(offset: Int, s: String) {
            val b = s.toByteArray(Charsets.US_ASCII)
            System.arraycopy(b, 0, out, offset, b.size)
        }

        fun putLEInt(offset: Int, v: Int) {
            out[offset] = (v and 0xff).toByte()
            out[offset + 1] = ((v shr 8) and 0xff).toByte()
            out[offset + 2] = ((v shr 16) and 0xff).toByte()
            out[offset + 3] = ((v shr 24) and 0xff).toByte()
        }

        fun putLEShort(offset: Int, v: Int) {
            out[offset] = (v and 0xff).toByte()
            out[offset + 1] = ((v shr 8) and 0xff).toByte()
        }

        putAscii(0, "RIFF")
        putLEInt(4, riffSize)
        putAscii(8, "WAVE")

        putAscii(12, "fmt ")
        putLEInt(16, 16)
        putLEShort(20, 1) // PCM
        putLEShort(22, 1) // mono
        putLEInt(24, sampleRate)
        putLEInt(28, sampleRate * 2)
        putLEShort(32, 2)
        putLEShort(34, 16)

        putAscii(36, "data")
        putLEInt(40, dataSize)

        System.arraycopy(pcm, 0, out, 44, dataSize)
        return out
    }
}
