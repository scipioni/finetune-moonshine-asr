#!/usr/bin/env python3
"""Download and save a Mozilla Common Voice language dataset locally."""

import moonshine_ft.compat  # noqa: F401 - Must be first import for Python 3.14 compatibility
import argparse
from pathlib import Path
from datasets import load_dataset, DatasetDict, Audio

def main():
    parser = argparse.ArgumentParser(description="Download Common Voice dataset.")
    parser.add_argument("--language", type=str, default="it", help="Language code (e.g., it, fr, es)")
    parser.add_argument("--dataset", type=str, default="fixie-ai/common_voice_17_0", help="Hugging Face dataset name/mirror")
    parser.add_argument("--output", type=str, required=True, help="Path to save the local dataset")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (if using gated datasets)")

    args = parser.parse_args()

    print(f"Loading public dataset mirror '{args.dataset}' for language '{args.language}'...")

    # Load dataset splits (public mirror does not require a gated token)
    token = args.token
    
    try:
        train_ds = load_dataset(args.dataset, args.language, split="train", token=token)
        test_ds = load_dataset(args.dataset, args.language, split="test", token=token)
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset from '{args.dataset}'.")
        print("Please check that the dataset name and language are correct.\n")
        raise e

    print("Formatting dataset splits...")
    ds = DatasetDict({
        "train": train_ds,
        "test": test_ds
    })

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
