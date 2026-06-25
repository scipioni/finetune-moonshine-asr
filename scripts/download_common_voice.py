#!/usr/bin/env python3
"""Download and save a Mozilla Common Voice language dataset locally."""

import argparse
from pathlib import Path
from datasets import load_dataset, DatasetDict, Audio

def main():
    parser = argparse.ArgumentParser(description="Download Common Voice dataset.")
    parser.add_argument("--language", type=str, default="it", help="Language code (e.g., it, fr, es)")
    parser.add_argument("--version", type=str, default="17_0", help="Common Voice version (e.g., 17_0, 13_0)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the local dataset")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (for gated datasets)")

    args = parser.parse_args()

    dataset_name = f"mozilla-foundation/common_voice_{args.version}"
    print(f"Loading {dataset_name} for language '{args.language}'...")

    # Load dataset splits (gated dataset requires a token or pre-login)
    token = args.token if args.token else True
    
    try:
        train_ds = load_dataset(dataset_name, args.language, split="train", token=token)
        test_ds = load_dataset(dataset_name, args.language, split="test", token=token)
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset. Please ensure you have accepted the dataset terms on ")
        print(f"https://huggingface.co/datasets/mozilla-foundation/common_voice_{args.version}")
        print("and passed a valid Hugging Face API token using the --token option.\n")
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
