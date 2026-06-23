# Fine-Tuning Moonshine for Italian

This guide covers training a Moonshine ASR model on Italian using the Multilingual LibriSpeech (MLS) dataset and [go-task](https://taskfile.dev) for workflow automation.

## Prerequisites

### Standard setup (NVIDIA / CUDA)

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Arch Linux with ROCm (AMD GPU)

The ROCm PyTorch build must come from pacman/AUR. Installing `torch` from PyPI would pull
hundreds of MB of CUDA/nvidia packages that are useless on AMD hardware.

**1. Install system packages:**

```bash
yay -S python-pytorch-rocm python-numpy python-pandas python-soundfile python-torchvision-rocm python-torchaudio-rocm
```

**2. Create the venv and install pip-only dependencies:**

```bash
uv venv --system-site-packages --python /usr/bin/python .venv
source .venv/bin/activate
task install-arch
```

`task install-arch` runs `uv pip install --excludes excludes-arch.txt -r requirements-arch.txt`.
The `--excludes` flag tells uv to treat `torch`, `torchaudio`, `torchvision`, `triton`, `numpy`,
`pandas`, and `soundfile` as already satisfied — preventing uv from pulling their PyPI (CUDA)
equivalents as transitive dependencies.

## Training

### Quick smoke test

Run 500 samples to verify the pipeline works end-to-end before committing to a full run:

```bash
task train-it-test
```

### Full training

Downloads MLS Italian directly from HuggingFace and trains for 8,000 steps (~2 epochs):

```bash
task train-it
```

Checkpoints and logs are saved to:

```
results-moonshine-it-no-curriculum/   # model checkpoints
logs/moonshine-it-no-curriculum/      # TensorBoard logs
```

Monitor training:

```bash
tensorboard --logdir logs/moonshine-it-no-curriculum
```

### Curriculum learning (optional)

Progressive 3-phase training based on the Moonshine paper — short utterances first, then longer ones:

```bash
task train-it-curriculum
```

This runs the three phases sequentially, each resuming from the previous checkpoint.

## Configuration

The training config is at `configs/mls_italian_no_curriculum.yaml`. Key settings tuned for a 6 GB GPU:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `per_device_train_batch_size` | 4 | Safe for 6 GB VRAM |
| `gradient_accumulation_steps` | 16 | Effective batch size 64 |
| `bf16` | true | BF16 instead of FP16 — more stable on ROCm/AMD |
| `max_steps` | 8000 | ~2 epochs on MLS Italian |
| `learning_rate` | 3e-4 | Scaled from paper for batch size 64 |
| `warmup_steps` | 256 | ~3.2% of total steps |

If you have more VRAM, increase `per_device_train_batch_size` and decrease `gradient_accumulation_steps` proportionally to keep the effective batch size at 64.

## Evaluation

```bash
task eval-it
```

Results are written to `eval-results-it.json`. To evaluate a specific checkpoint:

```bash
python scripts/evaluate.py \
    --model ./results-moonshine-it-no-curriculum/checkpoint-5000 \
    --dataset facebook/multilingual_librispeech \
    --split test \
    --text-column sentence
```

## Inference

```bash
task infer-it FILE=my_audio.wav
```

Or directly:

```bash
python scripts/inference.py \
    --model ./results-moonshine-it-no-curriculum \
    --audio my_audio.wav
```

### From Python

```python
from transformers import pipeline

transcriber = pipeline(
    "automatic-speech-recognition",
    model="./results-moonshine-it-no-curriculum"
)
result = transcriber("audio.wav")
print(result["text"])
```

## Export for deployment

Convert to ONNX for 10–30% faster CPU inference:

```bash
task export-it
# model saved to deploy/moonshine-it/
```

## Using the model with moonshine-ai/moonshine

The [moonshine-ai/moonshine](https://github.com/moonshine-ai/moonshine) library provides a
lightweight runtime for on-device inference with streaming, VAD, word-level timestamps, and
speaker diarization. It uses the OnnxRuntime `.ort` flatbuffer format rather than plain `.onnx`.

### 1. Install

```bash
uv pip install moonshine-voice
```

### 2. Convert ONNX → ORT

After `task export-it` the model is at `deploy/moonshine-it/`. Convert each `.onnx` file to the
`.ort` format that `moonshine-voice` expects:

```bash
python -m onnxruntime.tools.convert_onnx_models_to_ort deploy/moonshine-it/encoder_model.onnx
python -m onnxruntime.tools.convert_onnx_models_to_ort deploy/moonshine-it/decoder_model_merged.onnx
```

This produces `encoder_model.ort` and `decoder_model_merged.ort` in the same directory.
Copy the tokenizer from an official moonshine model download (the tokenizer is unchanged by
fine-tuning):

```bash
python -m moonshine_voice.download --language en   # downloads to cache
# copy tokenizer.bin from the cached path printed above into deploy/moonshine-it/
```

### 3. Batch transcription

```python
from moonshine_voice import Transcriber, ModelArch

transcriber = Transcriber(
    model_path="deploy/moonshine-it",
    model_arch=ModelArch.TINY,
    options={"max_tokens_per_second": "13.0"},  # recommended for non-Latin scripts
)

import soundfile as sf
audio, sr = sf.read("audio.wav", dtype="float32")

transcript = transcriber.transcribe_without_streaming(audio, sr)
for line in transcript.lines:
    print(line.text)
```

### 4. Streaming transcription

```python
from moonshine_voice import Transcriber, ModelArch, TranscriptEventListener

class Listener(TranscriptEventListener):
    def on_line_completed(self, event):
        print(event.line.text)

transcriber = Transcriber(
    model_path="deploy/moonshine-it",
    model_arch=ModelArch.TINY,
    update_interval=0.5,
)
transcriber.add_listener(Listener())
transcriber.start()

# feed audio chunks from a microphone or file
import sounddevice as sd
with sd.InputStream(samplerate=16000, channels=1, dtype="float32") as stream:
    while True:
        chunk, _ = stream.read(1600)   # 0.1 s at 16 kHz
        transcriber.add_audio(chunk.flatten(), 16000)
```

Call `transcriber.stop()` to flush the final segment.

### 5. Word-level timestamps

```python
transcriber = Transcriber(
    model_path="deploy/moonshine-it",
    model_arch=ModelArch.TINY,
    options={"word_timestamps": "true"},
)
transcript = transcriber.transcribe_without_streaming(audio, sr)
for line in transcript.lines:
    for word in line.words:
        print(f"[{word.start:.2f}s – {word.end:.2f}s]  {word.word}")
```

## Cleanup

```bash
task clean-it       # remove checkpoints and logs
task clean-data     # remove downloaded datasets
```

## Taskfile reference

All available tasks for Italian:

| Task | Description |
|------|-------------|
| `task download-it` | Download MLS Italian dataset locally |
| `task train-it` | Full training run |
| `task train-it-test` | Quick 500-sample smoke test |
| `task train-it-curriculum` | 3-phase curriculum training |
| `task eval-it` | Evaluate on MLS test split |
| `task infer-it FILE=…` | Transcribe a single audio file |
| `task export-it` | Export to ONNX |
| `task clean-it` | Remove outputs and logs |
