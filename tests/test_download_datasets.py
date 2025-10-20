"""Tests for download_datasets script."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import download_datasets


def test_download_cifar10_success(tmp_path):
    """Test successful CIFAR-10 download."""
    with patch('download_datasets.vision_datasets.CIFAR10') as mock_cifar:
        result = download_datasets.download_cifar10(tmp_path)
        assert result is True
        assert mock_cifar.call_count == 2  # train and test


def test_download_imdb_success(tmp_path):
    """Test successful IMDB download."""
    with patch('download_datasets.load_dataset') as mock_load:
        mock_dataset = MagicMock()
        mock_load.return_value = mock_dataset

        result = download_datasets.download_imdb(tmp_path)
        assert result is True
        mock_load.assert_called_once_with("imdb")


def test_download_cifar10_failure(tmp_path):
    """Test CIFAR-10 download failure handling."""
    with patch('download_datasets.vision_datasets.CIFAR10', side_effect=Exception("Network error")):
        result = download_datasets.download_cifar10(tmp_path)
        assert result is False


def test_download_imdb_failure(tmp_path):
    """Test IMDB download failure handling."""
    with patch('download_datasets.load_dataset', side_effect=Exception("Network error")):
        result = download_datasets.download_imdb(tmp_path)
        assert result is False
