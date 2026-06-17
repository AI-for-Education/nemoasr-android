package com.example.nemoasrdemo

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "ASR_DEMO"
    }

    private lateinit var asrEngine: OnnxAsrEngine
    private lateinit var tvStatus: TextView
    private lateinit var tvResult: TextView
    private lateinit var btnTranscribe: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.i(TAG, "onCreate: start")

        setContentView(R.layout.activity_main)

        tvStatus = findViewById(R.id.tvStatus)
        tvResult = findViewById(R.id.tvResult)
        btnTranscribe = findViewById(R.id.btnTranscribe)

        try {
            tvStatus.text = "Initializing..."
            Log.i(TAG, "Before initRecognizer()")
            initRecognizer()
            Log.i(TAG, "After initRecognizer()")
            tvStatus.text = "Model loaded"
        } catch (t: Throwable) {
            Log.e(TAG, "initRecognizer failed", t)
            tvStatus.text = "Init error"
            tvResult.text = t.stackTraceToString()
            return
        }

        btnTranscribe.setOnClickListener {
            Thread {
                try {
                    Log.i(TAG, "Button pressed")

                    runOnUiThread {
                        tvStatus.text = "Running ONNX..."
                        tvResult.text = ""
                    }
                    val result = asrEngine.transcribeAsset("asr/test.wav")
                    Log.i(TAG, "result text: $result")

                    runOnUiThread {
                        tvStatus.text = "Done"
                        tvResult.text = result
                    }
                } catch (t: Throwable) {
                    Log.e(TAG, "decode failed", t)
                    runOnUiThread {
                        tvStatus.text = "Error"
                        tvResult.text = t.stackTraceToString()
                    }
                }
            }.start()
        }
    }

    private fun initRecognizer() {
        Log.i(TAG, "initRecognizer: start")
        asrEngine = OnnxAsrEngine(this)
        Log.i(TAG, "initRecognizer: engine created")
    }

    override fun onDestroy() {
        Log.i(TAG, "onDestroy")
        if (::asrEngine.isInitialized) {
            asrEngine.close()
            Log.i(TAG, "engine released")
        }
        super.onDestroy()
    }
}
