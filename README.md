# NeMo ASR on Android with ONNX Runtime and onnx-asr

This repository is a starting point for running a custom NVIDIA NeMo ASR model on Android.

Python-side export and validation use [`onnx-asr`](https://pypi.org/project/onnx-asr/).

The workflow is:

1. Restore a `.nemo` ASR model in Python.
2. Export it to ONNX.
3. Quantize it to INT8.
4. Validate the exported model in Python with `onnx-asr`.
5. Package the quantized ONNX model into a minimal Android app that runs inference directly with `onnxruntime-android`.

The Android app in this repository does not embed Python or the `onnx-asr` package. `onnx-asr` is used on the Python side for export-time and desktop-side validation. On Android, the app uses:

- `onnxruntime-android` for model execution
- app-side log-mel feature extraction
- app-side greedy CTC decoding using `vocab.txt`

## Current status

Working:

- NeMo `.nemo` model export to ONNX.
- Dynamic INT8 quantization to `model.int8.onnx`.
- Python-side validation with `onnx-asr`.
- Android demo app that loads `model.int8.onnx` directly with `onnxruntime-android`.
- Asset-based test flow for a short `test.wav`.
- ONNX Runtime updated to `1.22.0`, which avoids the 16 KB page-size warning seen with older Android builds.

Important limitations:

- The Android demo currently targets the exported CTC-style model layout used in this repo: inputs `audio_signal` and `length`, plus token decoding through `tokens.txt`.
- The demo uses a simple greedy decoder. If your production model needs beam search, VAD, punctuation, timestamps or custom normalization logic, that still needs to be added separately.
- Large INT8 models can still be heavy for low-memory emulators. Prefer testing on a physical device for realistic results.

## Repository layout

```text
nemoasr-android/
├── README.md
├── .gitignore
├── scripts/
│   ├── convert-onnx_asr.py
│   ├── infer-manifest.py
│   ├── infer-test.py
│   └── quantize-model.py
├── artifacts/
│   └── .gitkeep
└── android/
    └── NemoAsrDemo/
        ├── app/
        ├── build.gradle.kts
        ├── gradle/
        ├── gradlew
        ├── gradlew.bat
        └── settings.gradle.kts
```

Recommended contents for `artifacts/` in this repository:

```text
artifacts/
├── model.int8.onnx
├── tokens.txt
└── test.wav
```

Optional export-time inputs and intermediates, such as `model_exp41_avg.nemo` and `model.onnx`, are not kept in the repository by default.

The contents for `artifacts/` can be downloaded from: https://drive.google.com/drive/folders/1_fQzWo8yTGE-dmqCmp_-dj0UI3d3TRMZ?usp=sharing

## 1. Create the Python environment

```bash
python3 -m venv .venv-nemo-export
source .venv-nemo-export/bin/activate

pip install --upgrade pip setuptools wheel
pip install "nemo_toolkit[asr]"
pip install onnx onnxruntime
pip install onnx-asr tqdm
```

## 2. Export the NeMo model to ONNX

Use `scripts/convert-onnx_asr.py`:

```bash
python3 scripts/convert-onnx_asr.py \
  --nemo artifacts/model_exp41_avg.nemo \
  --output-dir artifacts
```

This creates the export-time intermediate files:

```text
artifacts/model.onnx
artifacts/tokens.txt
```

`model.onnx` is an intermediate file used before quantization. It is not required by the Android demo once `model.int8.onnx` has been produced.

Notes:

- `tokens.txt` is written in `token id` format, which is what the Android demo expects.
- A `<blk>` token is appended as the final token for greedy CTC decoding.

## 3. Quantize the ONNX model to INT8

Use `scripts/quantize-model.py`:

```bash
python3 scripts/quantize-model.py artifacts
```

This creates:

```text
artifacts/model.int8.onnx
```

## 4. Validate the exported model in Python with onnx-asr

Quick test on one WAV file:

```bash
python3 scripts/infer-test.py \
  --model-dir artifacts \
  --wav artifacts/test.wav \
  --quantization int8
```

Batch test on a JSONL manifest:

```bash
python3 scripts/infer-manifest.py \
  --input-manifest eval.jsonl \
  --model-dir artifacts \
  --quantization int8
```

The manifest is expected to contain at least:

```json
{"audio_filepath": "/abs/path/file.wav", "text": "reference text"}
```

## 5. Prepare the Android project

The Android demo is in `android/NemoAsrDemo`.

It uses:

- Kotlin
- Android API 24+
- `onnxruntime-android`
- app-side feature extraction for 16 kHz mono audio

Copy the model assets into:

```text
android/NemoAsrDemo/app/src/main/assets/asr/
```

Required files:

```text
model.int8.onnx
tokens.txt
test.wav
```

The repository currently keeps these filenames fixed in the demo app:

- `asr/model.int8.onnx`
- `asr/vocab.txt`
- `asr/test.wav`

## 6. Build and run the Android demo

From the Android project directory:

```bash
cd android/NemoAsrDemo
./gradlew assembleDebug
```

Open the project in Android Studio or install the generated APK normally.

When the app starts:

1. It copies `model.int8.onnx` from assets into `filesDir`.
2. It initializes an ONNX Runtime session.
3. Pressing the button loads `test.wav`.
4. The app extracts 80-bin log-mel features.
5. The app runs the ONNX model.
6. The app decodes the frame-level logits with greedy CTC decoding.

## 7. Android implementation notes

The core files are:

- `android/NemoAsrDemo/app/src/main/java/com/example/nemoasrdemo/MainActivity.kt`
- `android/NemoAsrDemo/app/src/main/java/com/example/nemoasrdemo/OnnxAsrEngine.kt`
- `android/NemoAsrDemo/app/src/main/java/com/example/nemoasrdemo/SimpleWaveReader.kt`

Current assumptions in the demo:

- input WAV is mono PCM
- model sample rate is `16 kHz`
- feature dimension is `80`
- model inputs are named `audio_signal` and `length`
- output `0` contains logits compatible with CTC greedy decoding

If your exported ONNX graph differs from these assumptions, update the Android engine accordingly.

## 8. Practical notes

- Keep the Python-side validation step. It is the fastest way to separate model/export issues from Android integration issues.
- If the app fails on an emulator but works on desktop Python, check memory pressure before changing the model.
- If you replace the model, keep `vocab.txt` synchronized with the exported graph.
=======
