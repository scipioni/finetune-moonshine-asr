# Moonshine ASR Fine-Tuning Guide

A comprehensive guide and toolkit for fine-tuning the [Moonshine ASR model](https://github.com/usefulsensors/moonshine) for custom languages and domains.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Model](https://img.shields.io/badge/🤗-Model%20Card-yellow)](https://huggingface.co/Cornebidouil/moonshine-tiny-fr)

## 🎯 Overview

This repository provides everything you need to fine-tune Moonshine, a lightweight and efficient automatic speech recognition (ASR) model with only 27M parameters, achieving performance comparable to much larger models.

**What you'll learn:**
- ✅ How to prepare your dataset for fine-tuning
- ✅ Training with curriculum learning and schedule-free optimization
- ✅ Intelligent audio segmentation for better data quality
- ✅ Evaluation and inference with production-ready scripts
- ✅ Live transcription with Voice Activity Detection
- ✅ ONNX export for 10-30% faster inference
- ✅ Complete deployment pipeline

## 🎤 Pre-trained Model Available

**[moonshine-tiny-fr](https://huggingface.co/Cornebidouil/moonshine-tiny-fr)** - Fine-tuned French ASR model ready to use!

Fine-tuned using this guide on the Multilingual LibriSpeech French dataset:
- **WER**: 21.8% on test set
- **Model Size**: Only 27M parameters
- **Inference Speed**: RTF 0.11x (9x faster than real-time on CPU)
- **Training**: 8,000 steps with curriculum learning

Try it now:
```python
from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", model="Cornebidouil/moonshine-tiny-fr")
result = transcriber("french_audio.wav")
print(result['text'])
```

**[➡️ View Model Card on HuggingFace](https://huggingface.co/Cornebidouil/moonshine-tiny-fr)**

## 🚀 Quick Start

### Option 1: Use Pre-trained French Model (Fastest)

```bash
# Install dependencies
pip install transformers torch torchaudio

# Use the model
python
>>> from transformers import pipeline
>>> transcriber = pipeline("automatic-speech-recognition", model="Cornebidouil/moonshine-tiny-fr")
>>> result = transcriber("your_french_audio.wav")
>>> print(result['text'])
```

**[📥 Download Model](https://huggingface.co/Cornebidouil/moonshine-tiny-fr)**

### Option 2: Fine-Tune Your Own Model

```bash
# Clone the repository
git clone https://github.com/pierre-cheneau/finetune-moonshine-asr.git
cd finetune-moonshine-asr

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Live transcription support
pip install -r requirements-live.txt
```

### Fine-Tune Your First Model

```bash
# 1. Prepare your dataset (HuggingFace dataset format)
python scripts/intelligent_segmentation.py \
    --dataset facebook/multilingual_librispeech \
    --language french \
    --output ./data/mls_french_segmented

# 2. Train the model
python train.py --config configs/mls_french_no_curriculum.yaml

# 3. Evaluate on test set
python scripts/evaluate.py \
    --model results-moonshine-fr/checkpoint-6000 \
    --dataset ./data/test \
    --split test

# 4. Run inference
python scripts/inference.py \
    --model results-moonshine-fr/checkpoint-6000 \
    --audio sample.wav
```

## 🌟 Example: Fine-Tuned French Model

This guide was used to create **[moonshine-tiny-fr](https://huggingface.co/Cornebidouil/moonshine-tiny-fr)**, a production-ready French ASR model.

### Model Performance

| Metric | Value |
|--------|-------|
| **Word Error Rate (WER)** | 21.8% |
| **Character Error Rate (CER)** | ~10% |
| **Inference Speed (CPU)** | 9x faster than real-time |
| **Model Size** | 27M parameters |
| **Training Time** | ~24 hours on single GPU |

### Using the Model

**Basic Transcription:**
```python
from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", model="Cornebidouil/moonshine-tiny-fr")
result = transcriber("french_audio.wav")
print(result['text'])
```

**Batch Processing:**
```python
from pathlib import Path

audio_files = Path("./audio").glob("*.wav")
for audio in audio_files:
    result = transcriber(str(audio))
    print(f"{audio.name}: {result['text']}")
```

**Live Transcription:**
```bash
# Clone this repo and use inference.py
python scripts/inference.py --model Cornebidouil/moonshine-tiny-fr --live
```

**[📖 Full Model Documentation](https://huggingface.co/Cornebidouil/moonshine-tiny-fr)**

## 📚 Documentation

### Getting Started
- [Installation Guide](docs/INSTALLATION.md) - Complete setup instructions
- [Training Guide](docs/TRAINING_GUIDE.md) - Step-by-step training tutorial
- [Dataset Preparation](docs/DATASET_PREPARATION.md) - Prepare your audio data

### Language Guides
- [Italian](docs/italian.md) - Fine-tuning on MLS Italian, including ROCm/AMD setup

### Advanced Features
- [Inference Guide](docs/INFERENCE_GUIDE.md) - Single file, batch, and live inference
- [Live Transcription](docs/LIVE_MODE_GUIDE.md) - Real-time transcription with VAD
- [ONNX Runtime](docs/ONNX_MODE_GUIDE.md) - 10-30% faster inference

## 🛠️ Scripts

### Core Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | Main training script with curriculum learning |
| `scripts/inference.py` | Production inference (batch, live, ONNX) |
| `scripts/evaluate.py` | WER/CER evaluation on test sets |
| `scripts/convert_for_deployment.py` | Complete deployment pipeline |

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `scripts/intelligent_segmentation.py` | Segment long audio with forced alignment |
| `scripts/extract_samples.py` | Extract test samples from datasets |
| `scripts/checkpoint_to_dataset.py` | Create datasets from training checkpoints |

## 📦 Repository Structure

```
finetune-moonshine-asr/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── requirements-live.txt       # Optional live mode dependencies
├── train.py                    # Main training script
│
├── docs/                       # Documentation
│   ├── INSTALLATION.md
│   ├── TRAINING_GUIDE.md
│   ├── INFERENCE_GUIDE.md
│   ├── LIVE_MODE_GUIDE.md
│   └── ONNX_MODE_GUIDE.md
│
├── scripts/                    # Utility scripts
│   ├── inference.py
│   ├── evaluate.py
│   ├── convert_for_deployment.py
│   ├── intelligent_segmentation.py
│   └── extract_samples.py
│
├── configs/                    # Training configurations
│   ├── mls_french_no_curriculum.yaml
│   └── example_curriculum.yaml
│
├── examples/                   # Example notebooks
│   └── fine_tune_moonshine_curriculum.ipynb
│
└── moonshine_ft/              # Fine-tuning library
    ├── __init__.py
    ├── data_loader.py
    ├── trainer.py
    └── configs/
```

## 🎓 Tutorial: Fine-Tune for French

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Dataset

```bash
# Option A: Use intelligent segmentation (recommended)
python scripts/intelligent_segmentation.py \
    --dataset facebook/multilingual_librispeech \
    --language french \
    --output ./data/mls_french_segmented \
    --max-duration 10.0 \
    --min-duration 1.0

# Option B: Use pre-segmented dataset
# Just specify the dataset in your config file
```

### Step 3: Configure Training

Create or edit `configs/my_french_model.yaml`:

```yaml
# Dataset configuration
dataset:
  name: "facebook/multilingual_librispeech"
  language: "french"
  train_split: "train"
  test_split: "test"

# Training configuration
training:
  output_dir: "./results-moonshine-fr"
  num_train_epochs: 3
  per_device_train_batch_size: 16
  learning_rate: 5e-5
  warmup_steps: 500

# Model configuration
model:
  name: "UsefulSensors/moonshine-tiny"

# Optimizer
optimizer:
  type: "schedulefree_adamw"
  betas: [0.9, 0.999]
  weight_decay: 0.01
```

### Step 4: Train

```bash
python train.py --config configs/my_french_model.yaml
```

Monitor with TensorBoard:
```bash
tensorboard --logdir results-moonshine-fr/runs
```

### Step 5: Evaluate

```bash
python scripts/evaluate.py \
    --model results-moonshine-fr/checkpoint-best \
    --dataset facebook/multilingual_librispeech \
    --language french \
    --split test
```

### Step 6: Inference

```bash
# Single file
python scripts/inference.py \
    --model results-moonshine-fr/checkpoint-best \
    --audio my_audio.wav

# Live transcription
python scripts/inference.py \
    --model results-moonshine-fr/checkpoint-best \
    --live

# ONNX (faster)
python scripts/convert_for_deployment.py \
    --model results-moonshine-fr/checkpoint-best \
    --output moonshine-fr-onnx

python scripts/inference.py \
    --model moonshine-fr-onnx/onnx \
    --audio my_audio.wav \
    --use-manual-onnx
```

## 🔬 Advanced Features

### Curriculum Learning

Train with progressive difficulty for better convergence:

```yaml
curriculum:
  enabled: true
  stages:
    - duration: 2000  # steps
      max_audio_length: 5.0
      description: "Short audio clips"

    - duration: 3000
      max_audio_length: 10.0
      description: "Medium audio clips"

    - duration: 3000
      max_audio_length: 20.0
      description: "Full-length audio"
```

### Intelligent Audio Segmentation

Use Whisper V3 + forced alignment for optimal segmentation:

```bash
python scripts/intelligent_segmentation.py \
    --dataset your/dataset \
    --language french \
    --output ./data/segmented \
    --use-whisper-v3 \
    --alignment-method "forced" \
    --max-duration 10.0
```

### Schedule-Free Optimization

Modern optimizer without learning rate schedules:

```yaml
optimizer:
  type: "schedulefree_adamw"
  learning_rate: 5e-5
  betas: [0.9, 0.999]
  weight_decay: 0.01
  warmup_steps: 500
```

## 📈 Performance Tips

### Training
- Use curriculum learning for better convergence
- Start with `batch_size=16`, increase if you have more GPU memory
- Use schedule-free AdamW optimizer (no LR scheduling needed)
- Monitor WER on validation set, save best checkpoint

### Inference
- **CPU**: Use ONNX manual mode (20-30% faster)
- **GPU**: Use PyTorch with FP16 (fastest)
- **Live**: Enable VAD for better segmentation
- **Batch**: Process multiple files at once for efficiency

### Deployment
- Convert to ONNX for production
- Use merged decoder for KV cache efficiency
- Binary tokenizer for faster loading
- ORT optimization for additional speedup

## 🐛 Troubleshooting

### Training Issues

**Q: Out of memory during training**
```bash
# Reduce batch size
per_device_train_batch_size: 8  # instead of 16

# Or enable gradient accumulation
gradient_accumulation_steps: 2
```

**Q: Model not converging**
```bash
# Try curriculum learning
# Start with shorter audio clips
# Increase warmup steps
warmup_steps: 1000
```

### Inference Issues

**Q: Transcriptions are truncated**
```bash
# Already fixed in our scripts!
# Uses: max_new_tokens = audio_duration * 5
```

**Q: Slow inference on CPU**
```bash
# Use ONNX mode
python scripts/inference.py --model model-onnx --audio audio.wav --use-manual-onnx
```

### Live Mode Issues

**Q: No microphone detected**
```bash
# Check available devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Install sounddevice
pip install sounddevice
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Multi-language support examples
- [ ] More curriculum learning strategies
- [ ] Quantization support (INT8)
- [ ] Speaker diarization integration
- [ ] Punctuation restoration
- [ ] Docker deployment examples

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Useful Sensors](https://github.com/usefulsensors) for the Moonshine model
- [HuggingFace](https://huggingface.co/) for the Transformers library
- [Schedule-Free Learning](https://github.com/facebookresearch/schedule_free) for the optimizer
- [Multilingual LibriSpeech](https://www.openslr.org/94/) for training data

## 📚 Citation

If you use this guide or the fine-tuned model in your research, please cite:

### This Fine-Tuning Guide
```bibtex
@misc{cheneau2026moonshine-finetune,
  author = {Pierre Chéneau (Cornebidouil)},
  title = {Moonshine ASR Fine-Tuning Guide},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/pierre-cheneau/finetune-moonshine-asr}
}
```

### Fine-Tuned French Model
```bibtex
@misc{cheneau2026moonshine-tiny-fr,
  author = {Pierre Chéneau (Cornebidouil)},
  title = {Moonshine-Tiny-FR: Fine-tuned French Speech Recognition},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/Cornebidouil/moonshine-tiny-fr}
}
```

### Original Moonshine
```bibtex
@misc{jeffries2024moonshinespeechrecognitionlive,
  title={Moonshine: Speech Recognition for Live Transcription and Voice Commands},
  author={Nat Jeffries and Evan King and Manjunath Kudlur and Guy Nicholson and James Wang and Pete Warden},
  year={2024},
  eprint={2410.15608},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2410.15608}
}
```

## 🔗 Related Resources

### Our Models
- **[moonshine-tiny-fr](https://huggingface.co/Cornebidouil/moonshine-tiny-fr)** - Fine-tuned French model (ready to use!)

### Original Moonshine
- [Moonshine Official Repo](https://github.com/usefulsensors/moonshine)
- [Original Moonshine Paper](https://arxiv.org/abs/2410.15608)
- [Base Model Card](https://huggingface.co/UsefulSensors/moonshine-tiny)

### Datasets
- [Multilingual LibriSpeech](https://huggingface.co/datasets/facebook/multilingual_librispeech) - Used for French training
- [Common Voice](https://commonvoice.mozilla.org/) - Alternative dataset

## Contact

For questions or issues:
- Website: [pcheneau.fr](https://pcheneau.fr)
- Github: [@pierre-cheneau](https://github.com/pierre-cheneau)
- HuggingFace: [@Cornebidouil](https://huggingface.co/Cornebidouil)
- Discord: [HogwartsLegacySpellCaster](https://discord.gg/zE4NRsTGdw) (Hogwarts Legacy Spell Recognition project's discord)
---

**Made with ❤️ for the ASR community**
