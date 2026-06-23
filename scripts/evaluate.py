#!/usr/bin/env python3
"""
Moonshine ASR Evaluation Script

Publication-ready evaluation script for computing WER, CER, and other metrics
on test datasets.

Usage:
    # Evaluate on HuggingFace dataset
    python evaluate.py --model ./results-moonshine-fr/final \\
        --dataset ./data/mls_french_split --split test

    # Evaluate on local dataset with specific column names
    python evaluate.py --model ./model --dataset ./data/test \\
        --audio-column audio --text-column transcript

    # Save detailed results
    python evaluate.py --model ./model --dataset ./data/test \\
        --output ./evaluation_results.json --save-predictions
"""

import moonshine_ft.compat  # noqa: F401
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from datasets import load_from_disk, load_dataset
from transformers import (
    MoonshineForConditionalGeneration,
    AutoProcessor
)
from tqdm import tqdm
import jiwer


def compute_wer(predictions: List[str], references: List[str]) -> Dict:
    """
    Compute Word Error Rate and related metrics.

    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions

    Returns:
        Dictionary with WER, CER, insertions, deletions, substitutions
    """
    # Compute WER
    wer = jiwer.wer(references, predictions)

    # Compute CER (Character Error Rate)
    cer = jiwer.cer(references, predictions)

    # Get detailed measures using process_words
    output = jiwer.process_words(references, predictions)

    return {
        'wer': wer * 100,  # Convert to percentage
        'cer': cer * 100,
        'substitutions': output.substitutions,
        'deletions': output.deletions,
        'insertions': output.insertions,
        'hits': output.hits,
        'num_words': output.hits + output.substitutions + output.deletions
    }


def normalize_audio(audio_data: np.ndarray, target_rms: float = 0.075) -> np.ndarray:
    """Normalize audio amplitude."""
    rms = np.sqrt(np.mean(audio_data**2))
    if rms > 0.001:
        scale_factor = target_rms / rms
        normalized = audio_data * scale_factor
        return np.clip(normalized, -1.0, 1.0)
    return audio_data


class MoonshineEvaluator:
    """Moonshine ASR evaluation pipeline."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        fp16: bool = False
    ):
        """
        Initialize evaluation pipeline.

        Args:
            model_path: Path to model directory
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            fp16: Use FP16 precision (CUDA only)
        """
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.fp16 = fp16 and self.device.type == "cuda"

        print(f"Loading model from: {model_path}")
        print(f"Device: {self.device}")
        print(f"FP16: {self.fp16}")

        # Load model and processor
        self.model = MoonshineForConditionalGeneration.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)

        # Move to device
        self.model.to(self.device)

        if self.fp16:
            self.model.half()

        self.model.eval()

        print(f"Model loaded: {self.model.num_parameters():,} parameters")

    def transcribe_sample(
        self,
        audio_array: np.ndarray,
        sampling_rate: int,
        num_beams: int = 5,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 2
    ) -> str:
        """
        Transcribe a single audio sample.

        Args:
            audio_array: Audio array
            sampling_rate: Sampling rate
            num_beams: Number of beams for beam search
            repetition_penalty: Repetition penalty
            no_repeat_ngram_size: Block repeated n-grams

        Returns:
            Transcribed text
        """
        # Normalize audio
        audio_array = normalize_audio(audio_array, target_rms=0.075)

        # Process audio
        inputs = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            return_attention_mask=True
        )

        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        if self.fp16:
            input_values = input_values.half()

        # Calculate max tokens based on audio duration
        # Roughly 5 tokens per second (accounting for French verbosity)
        audio_duration = len(audio_array) / sampling_rate
        max_new_tokens = max(10, min(int(audio_duration * 5), 150))

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_values=input_values,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=False,
                early_stopping=True
            )

        # Decode
        transcription = self.processor.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]

        return transcription.strip()

    def evaluate_dataset(
        self,
        dataset,
        audio_column: str = 'audio',
        text_column: str = 'sentence',
        max_samples: Optional[int] = None,
        batch_size: int = 1,
        num_beams: int = 5,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 2,
        save_predictions: bool = False
    ) -> Dict:
        """
        Evaluate model on a dataset.

        Args:
            dataset: HuggingFace dataset
            audio_column: Name of audio column
            text_column: Name of text column
            max_samples: Maximum samples to evaluate (None = all)
            batch_size: Batch size (currently only 1 supported)
            num_beams: Number of beams
            repetition_penalty: Repetition penalty
            no_repeat_ngram_size: Block repeated n-grams
            save_predictions: Save individual predictions

        Returns:
            Dictionary with metrics and optionally predictions
        """
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        print(f"\nEvaluating on {len(dataset)} samples...")

        predictions = []
        references = []
        durations = []
        inference_times = []

        # Evaluate
        for sample in tqdm(dataset, desc="Evaluating"):
            try:
                # Get audio and reference
                audio = sample[audio_column]
                reference = sample[text_column]

                audio_array = audio['array']
                sampling_rate = audio['sampling_rate']

                # Transcribe
                start_time = time.time()
                prediction = self.transcribe_sample(
                    audio_array,
                    sampling_rate,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size
                )
                inference_time = time.time() - start_time

                predictions.append(prediction)
                references.append(reference)
                durations.append(len(audio_array) / sampling_rate)
                inference_times.append(inference_time)

            except Exception as e:
                print(f"\nError processing sample: {e}")
                predictions.append("")
                references.append(sample.get(text_column, ""))
                durations.append(0)
                inference_times.append(0)

        # Compute metrics
        print("\nComputing metrics...")
        metrics = compute_wer(predictions, references)

        # Add timing metrics
        total_audio_duration = sum(durations)
        total_inference_time = sum(inference_times)
        metrics['total_audio_duration'] = total_audio_duration
        metrics['total_inference_time'] = total_inference_time
        metrics['rtf'] = total_inference_time / total_audio_duration if total_audio_duration > 0 else 0
        metrics['num_samples'] = len(predictions)

        # Add predictions if requested
        if save_predictions:
            metrics['predictions'] = [
                {
                    'prediction': pred,
                    'reference': ref,
                    'duration': dur,
                    'inference_time': inf_time
                }
                for pred, ref, dur, inf_time in zip(predictions, references, durations, inference_times)
            ]

        return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Moonshine ASR Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on local dataset
  python evaluate.py --model ./results-moonshine-fr/final \\
      --dataset ./data/mls_french_split --split test

  # Custom column names
  python evaluate.py --model ./model --dataset ./data/test \\
      --audio-column audio --text-column transcript

  # Limit samples and save predictions
  python evaluate.py --model ./model --dataset ./data/test \\
      --max-samples 1000 --save-predictions --output results.json

  # GPU with FP16
  python evaluate.py --model ./model --dataset ./data/test \\
      --device cuda --fp16
        """
    )

    # Required arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model directory'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset directory or HuggingFace dataset name'
    )

    # Dataset arguments
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Dataset split to evaluate (default: test)'
    )
    parser.add_argument(
        '--audio-column',
        type=str,
        default='audio',
        help='Name of audio column (default: audio)'
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='sentence',
        help='Name of text column (default: sentence)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum samples to evaluate (default: all)'
    )

    # Model arguments
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use FP16 precision (CUDA only)'
    )

    # Generation parameters
    parser.add_argument(
        '--num-beams',
        type=int,
        default=5,
        help='Number of beams (default: 5)'
    )
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.3,
        help='Repetition penalty (default: 1.3)'
    )
    parser.add_argument(
        '--no-repeat-ngram-size',
        type=int,
        default=2,
        help='Block repeated n-grams (default: 2)'
    )

    # Output arguments
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save individual predictions in output file'
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from: {args.dataset}")

    try:
        # Try loading as local dataset first
        if Path(args.dataset).exists():
            dataset_dict = load_from_disk(args.dataset)
            if isinstance(dataset_dict, dict):
                # DatasetDict
                dataset = dataset_dict[args.split]
            else:
                # Single dataset
                dataset = dataset_dict
        else:
            # Try HuggingFace Hub
            dataset = load_dataset(args.dataset, split=args.split)

        print(f"Loaded {len(dataset)} samples from split '{args.split}'")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1

    # Initialize evaluator
    evaluator = MoonshineEvaluator(
        model_path=args.model,
        device=args.device,
        fp16=args.fp16
    )

    # Evaluate
    results = evaluator.evaluate_dataset(
        dataset,
        audio_column=args.audio_column,
        text_column=args.text_column,
        max_samples=args.max_samples,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        save_predictions=args.save_predictions
    )

    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Samples: {results['num_samples']}")
    print(f"\nAccuracy Metrics:")
    print(f"  WER: {results['wer']:.2f}%")
    print(f"  CER: {results['cer']:.2f}%")
    print(f"\nError Breakdown:")
    print(f"  Substitutions: {results['substitutions']}")
    print(f"  Deletions: {results['deletions']}")
    print(f"  Insertions: {results['insertions']}")
    print(f"  Correct words: {results['hits']}")
    print(f"\nPerformance:")
    print(f"  Total audio duration: {results['total_audio_duration']:.1f}s")
    print(f"  Total inference time: {results['total_inference_time']:.1f}s")
    print(f"  Real-time factor: {results['rtf']:.2f}x")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
