#!/usr/bin/env python3
"""Download and save a Facebook VoxPopuli language dataset locally."""

import moonshine_ft.compat  # noqa: F401 - Must be first import for Python 3.14 compatibility
import argparse
from pathlib import Path
from datasets import load_dataset, DatasetDict, Audio

def main():
    parser = argparse.ArgumentParser(description="Download VoxPopuli dataset.")
    parser.add_argument("--language", type=str, default="it", help="Language code (e.g., it, fr, es)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the local dataset")

    args = parser.parse_args()

    dataset_name = "facebook/voxpopuli"
    print(f"Loading {dataset_name} for language '{args.language}'...")

    try:
        train_ds = load_dataset(dataset_name, args.language, split="train")
        test_ds = load_dataset(dataset_name, args.language, split="test")
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset '{dataset_name}' for language '{args.language}'.\n")
        raise e

    print("Formatting dataset splits...")
    ds = DatasetDict({
        "train": train_ds,
        "test": test_ds
    })

    # Rename transcription column 'normalized_text' to 'sentence'
    ds = ds.rename_column("normalized_text", "sentence")

    # Keep only audio and sentence, and cast audio to 16kHz
    ds = ds.select_columns(["audio", "sentence"])
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    print(f"Saving to disk at {args.output}...")
    ds.save_to_disk(args.output)
    print(f"Saved successfully to {args.output}")
    print(f"  Train: {len(ds['train']):,} samples")
    print(f"  Test:  {len(ds['test']):,} samples")

if __name__ == "__main__":
    main()
