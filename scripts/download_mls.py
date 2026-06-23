#!/usr/bin/env python3
"""Download and save an MLS language dataset locally."""

import moonshine_ft.compat  # noqa: F401
import argparse
from datasets import load_dataset, DatasetDict, Audio

LANGUAGES = ["french", "italian", "german", "dutch", "spanish", "portuguese", "polish"]


def main():
    parser = argparse.ArgumentParser(description="Download MLS dataset for a given language")
    parser.add_argument("--language", required=True, choices=LANGUAGES, help="Language to download")
    parser.add_argument("--output", required=True, help="Output directory path")
    args = parser.parse_args()

    print(f"Downloading MLS {args.language}...")
    ds = DatasetDict({
        "train": load_dataset("facebook/multilingual_librispeech", args.language, split="train"),
        "test":  load_dataset("facebook/multilingual_librispeech", args.language, split="test"),
    })
    ds = ds.rename_column("transcript", "sentence")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds.save_to_disk(args.output)
    print(f"Saved to {args.output}")
    print(f"  Train: {len(ds['train']):,} samples")
    print(f"  Test:  {len(ds['test']):,} samples")


if __name__ == "__main__":
    main()
