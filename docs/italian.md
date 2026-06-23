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
sudo pacman -S python-pytorch-opt-rocm python-torchvision python-numpy python-pandas python-soundfile
yay -S python-torchaudio
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
