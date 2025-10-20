"""
Download CIFAR-10 and IMDB datasets for training.

This script downloads public datasets used for:
- Image classification: CIFAR-10
- Text sentiment analysis: IMDB

Datasets are saved to data/raw/ directory.
"""

import argparse
import logging
import sys
from pathlib import Path

import torchvision.datasets as vision_datasets
from datasets import load_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/download.log"),
    ],
)
logger = logging.getLogger(__name__)


def download_cifar10(data_dir: Path) -> bool:
    """
    Download CIFAR-10 dataset.
    Args:
        data_dir: Directory to save the dataset
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Downloading CIFAR-10 dataset...")
        images_dir = data_dir / "images" / "cifar10"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Download training data
        logger.info("Downloading CIFAR-10 training set...")
        vision_datasets.CIFAR10(
            root=str(images_dir),
            train=True,
            download=True
        )

        # Download test data
        logger.info("Downloading CIFAR-10 test set...")
        vision_datasets.CIFAR10(
            root=str(images_dir),
            train=False,
            download=True
        )

        logger.info("✅ CIFAR-10 downloaded successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download CIFAR-10: {e}")
        return False


def download_imdb(data_dir: Path) -> bool:
    """
    Download IMDB dataset.
    Args:
        data_dir: Directory to save the dataset
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Downloading IMDB dataset...")
        text_dir = data_dir / "text" / "imdb"
        text_dir.mkdir(parents=True, exist_ok=True)

        # Download dataset
        logger.info("Downloading IMDB dataset from HuggingFace...")
        dataset = load_dataset("imdb")

        # Save to disk
        logger.info("Saving IMDB dataset to disk...")
        dataset.save_to_disk(str(text_dir))

        logger.info("✅ IMDB downloaded successfully")
        logger.info(f"   - Train samples: {len(dataset['train'])}")
        logger.info(f"   - Test samples: {len(dataset['test'])}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download IMDB: {e}")
        return False


def main() -> int:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Download CIFAR-10 and IMDB datasets"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save datasets (default: data/raw)",
    )
    parser.add_argument(
        "--cifar10-only",
        action="store_true",
        help="Download only CIFAR-10",
    )
    parser.add_argument(
        "--imdb-only",
        action="store_true",
        help="Download only IMDB",
    )

    args = parser.parse_args()

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Create data directory
    args.data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("Starting dataset download")
    logger.info(f"Data directory: {args.data_dir.absolute()}")
    logger.info("="*60)

    success = True

    # Download datasets
    if not args.imdb_only:
        success = download_cifar10(args.data_dir) and success

    if not args.cifar10_only:
        success = download_imdb(args.data_dir) and success

    # Summary
    logger.info("="*60)
    if success:
        logger.info("✅ All datasets downloaded successfully!")
        logger.info(f"📁 Location: {args.data_dir.absolute()}")
        logger.info("\nNext steps:")
        logger.info("  python scripts/prepare_data.py")
        return 0
    else:
        logger.error("❌ Some datasets failed to download")
        logger.error("Check logs/download.log for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
