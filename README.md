<<<<<<< HEAD
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
- app-side greedy CTC decoding using `tokens.txt`

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
- `asr/tokens.txt`
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
- If you replace the model, keep `tokens.txt` synchronized with the exported graph.
=======
# NeMoASR-android



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

* [Create](https://docs.gitlab.com/user/project/repository/web_editor/#create-a-file) or [upload](https://docs.gitlab.com/user/project/repository/web_editor/#upload-a-file) files
* [Add files using the command line](https://docs.gitlab.com/topics/git/add_files/#add-files-to-a-git-repository) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.com/zevo-tech/voiceai/nemoasr-android.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

* [Set up project integrations](https://gitlab.com/zevo-tech/voiceai/nemoasr-android/-/settings/integrations)

## Collaborate with your team

* [Invite team members and collaborators](https://docs.gitlab.com/user/project/members/)
* [Create a new merge request](https://docs.gitlab.com/user/project/merge_requests/creating_merge_requests/)
* [Automatically close issues from merge requests](https://docs.gitlab.com/user/project/issues/managing_issues/#closing-issues-automatically)
* [Enable merge request approvals](https://docs.gitlab.com/user/project/merge_requests/approvals/)
* [Set auto-merge](https://docs.gitlab.com/user/project/merge_requests/auto_merge/)

## Test and Deploy

Use the built-in continuous integration in GitLab.

* [Get started with GitLab CI/CD](https://docs.gitlab.com/ci/quick_start/)
* [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/user/application_security/sast/)
* [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/topics/autodevops/requirements/)
* [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/user/clusters/agent/)
* [Set up protected environments](https://docs.gitlab.com/ci/environments/protected_environments/)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
>>>>>>> 7d66e5999fc6e5f1f1be748a7e9a82486dd8b72c
