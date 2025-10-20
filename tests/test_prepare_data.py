"""Tests for prepare_data script."""

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_prepare_data_imports():
    """Test that prepare_data script imports correctly."""
    import prepare_data
    assert hasattr(prepare_data, 'prepare_cifar10')
    assert hasattr(prepare_data, 'prepare_imdb')
    assert hasattr(prepare_data, 'main')


# More tests will be added as we implement the scripts
