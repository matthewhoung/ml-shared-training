"""
Prepare datasets for training by splitting into train/val/test sets.

This script:
1. Loads raw CIFAR-10 and IMDB datasets
2. Creates stratified splits (70% train, 15% val, 15% test)
3. Saves to data/processed/ directory
4. Generates metadata about splits
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torchvision.datasets as vision_datasets
from datasets import DatasetDict, load_from_disk
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def prepare_cifar10(raw_dir: Path, processed_dir: Path) -> dict[str, Any]:
    """
    Prepare CIFAR-10 dataset with stratified splits.
    Args:
        raw_dir: Directory containing raw CIFAR-10
        processed_dir: Directory to save processed splits
    Returns:
        Metadata dictionary
    """
    logger.info("Preparing CIFAR-10 dataset...")

    # Load raw datasets
    train_dataset = vision_datasets.CIFAR10(
        root=str(raw_dir / "images" / "cifar10"),
        train=True,
        download=False
    )
    test_dataset = vision_datasets.CIFAR10(
        root=str(raw_dir / "images" / "cifar10"),
        train=False,
        download=False
    )

    # Get data and labels
    X_train = np.array(train_dataset.data)
    y_train = np.array(train_dataset.targets)
    X_test = np.array(test_dataset.data)
    y_test = np.array(test_dataset.targets)

    # Split training into train and val (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.176,  # 15% of original (15/85)
        stratify=y_train,
        random_state=42
    )

    # Save splits
    output_dir = processed_dir / "images" / "cifar10"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "train_images.npy", X_train)
    np.save(output_dir / "train_labels.npy", y_train)
    np.save(output_dir / "val_images.npy", X_val)
    np.save(output_dir / "val_labels.npy", y_val)
    np.save(output_dir / "test_images.npy", X_test)
    np.save(output_dir / "test_labels.npy", y_test)

    # Metadata
    metadata = {
        "dataset": "CIFAR-10",
        "type": "image_classification",
        "num_classes": 10,
        "image_shape": [32, 32, 3],
        "splits": {
            "train": {"count": len(X_train), "distribution": np.bincount(y_train).tolist()},
            "val": {"count": len(X_val), "distribution": np.bincount(y_val).tolist()},
            "test": {"count": len(X_test), "distribution": np.bincount(y_test).tolist()},
        },
        "class_names": train_dataset.classes,
    }

    logger.info("✅ CIFAR-10 prepared:")
    logger.info(f"   Train: {len(X_train)} samples")
    logger.info(f"   Val:   {len(X_val)} samples")
    logger.info(f"   Test:  {len(X_test)} samples")

    return metadata


def prepare_imdb(raw_dir: Path, processed_dir: Path) -> dict[str, Any]:
    """
    Prepare IMDB dataset with stratified splits.
    Args:
        raw_dir: Directory containing raw IMDB
        processed_dir: Directory to save processed splits
    Returns:
        Metadata dictionary
    """
    logger.info("Preparing IMDB dataset...")

    # Load dataset
    dataset = load_from_disk(str(raw_dir / "text" / "imdb"))

    # Split training data into train and val
    train_val_split = dataset["train"].train_test_split(
        test_size=0.176,  # 15% of original
        stratify_by_column="label",
        seed=42
    )

    # Create new dataset dict
    processed_dataset = DatasetDict({
        "train": train_val_split["train"],
        "val": train_val_split["test"],
        "test": dataset["test"],
    })

    # Save
    output_dir = processed_dir / "text" / "imdb"
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dataset.save_to_disk(str(output_dir))

    # Metadata
    metadata = {
        "dataset": "IMDB",
        "type": "text_classification",
        "num_classes": 2,
        "splits": {
            "train": {
                "count": len(processed_dataset["train"]),
                "distribution": [
                    sum(processed_dataset["train"]["label"]),
                    len(processed_dataset["train"]) - sum(processed_dataset["train"]["label"])
                ]
            },
            "val": {
                "count": len(processed_dataset["val"]),
                "distribution": [
                    sum(processed_dataset["val"]["label"]),
                    len(processed_dataset["val"]) - sum(processed_dataset["val"]["label"])
                ]
            },
            "test": {
                "count": len(processed_dataset["test"]),
                "distribution": [
                    sum(processed_dataset["test"]["label"]),
                    len(processed_dataset["test"]) - sum(processed_dataset["test"]["label"])
                ]
            },
        },
        "class_names": ["negative", "positive"],
    }

    logger.info("✅ IMDB prepared:")
    logger.info(f"   Train: {len(processed_dataset['train'])} samples")
    logger.info(f"   Val:   {len(processed_dataset['val'])} samples")
    logger.info(f"   Test:  {len(processed_dataset['test'])} samples")

    return metadata


def main() -> int:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Prepare datasets for training"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw datasets",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to save processed datasets",
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Starting data preparation")
    logger.info("="*60)

    # Prepare datasets
    try:
        cifar10_metadata = prepare_cifar10(args.raw_dir, args.processed_dir)
        imdb_metadata = prepare_imdb(args.raw_dir, args.processed_dir)

        # Save combined metadata
        metadata = {
            "cifar10": cifar10_metadata,
            "imdb": imdb_metadata,
        }

        metadata_file = args.processed_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("="*60)
        logger.info("✅ Data preparation complete!")
        logger.info(f"📁 Processed data: {args.processed_dir.absolute()}")
        logger.info(f"📄 Metadata: {metadata_file.absolute()}")
        logger.info("\nNext steps:")
        logger.info("  python scripts/train_image.py --config training/configs/image_config.yaml")
        return 0

    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
