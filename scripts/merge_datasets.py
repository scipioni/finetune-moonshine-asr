#!/usr/bin/env python3
"""Merge multiple preprocessed local ASR datasets into one unified dataset."""

import moonshine_ft.compat  # noqa: F401 - Must be first import for Python 3.14 compatibility
import argparse
from pathlib import Path
from datasets import load_from_disk, concatenate_datasets, DatasetDict

def main():
    parser = argparse.ArgumentParser(description="Merge local datasets.")
    parser.add_argument("--output", type=str, default="./data/merged_italian_split", help="Output path for merged dataset")
    args = parser.parse_args()

    paths = {
        "MLS": "./data/mls_italian_split",
        "Common Voice": "./data/cv_italian_split",
        "VoxPopuli": "./data/vp_italian_split"
    }

    loaded_datasets = {}
    for name, path in paths.items():
        if Path(path).exists():
            print(f"Loading {name} from {path}...")
            loaded_datasets[name] = load_from_disk(path)
        else:
            print(f"Note: {name} not found at {path} (skipping).")

    if not loaded_datasets:
        print("[ERROR] No datasets found to merge! Please download at least one dataset.")
        return

    print("\nMerging dataset splits...")
    train_splits = []
    test_splits = []

    for name, ds in loaded_datasets.items():
        train_splits.append(ds["train"])
        test_splits.append(ds["test"])

    merged_train = concatenate_datasets(train_splits)
    merged_test = concatenate_datasets(test_splits)

    merged_ds = DatasetDict({
        "train": merged_train,
        "test": merged_test
    })

    print(f"\nSaving merged dataset to {args.output}...")
    merged_ds.save_to_disk(args.output)
    print("Saved successfully!")
    print(f"  Total Train samples: {len(merged_ds['train']):,}")
    print(f"  Total Test samples:  {len(merged_ds['test']):,}")

if __name__ == "__main__":
    main()
