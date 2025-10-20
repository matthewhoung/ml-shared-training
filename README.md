# ML Shared Training Pipeline

[![Python 3.11.14](https://img.shields.io/badge/python-3.11.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

shared training pipeline for image and text models, separated from serving architecture.

## 🎯 Purpose

This repository contains the **training pipeline** for ML models used across multiple serving architectures:
- **Monolithic Architecture**
- **Service-Oriented Architecture (SOA)**
- **Microservices Architecture**

### Models
- **Image Model**: EfficientNet-B0 for image classification
- **Text Model**: DistilBERT for text processing

### Storage Strategy
- **Training**: Local experimentation with GCS upload
- **Versioning**: Semantic versioning (v1.0.0)
- **Environments**: Staging → Serving promotion workflow

---

## 📂 Project Structure
```
ml-shared-training/
├── data/                          # Datasets (gitignored)
│   ├── raw/                       # Original downloads
│   │   ├── images/                # CIFAR-10
│   │   └── text/                  # IMDB
│   └── processed/                 # After preparation
│       ├── train/
│       ├── val/
│       └── test/
│
├── scripts/                       # Executable scripts
│   ├── download_datasets.py       # Download CIFAR-10 & IMDB
│   ├── prepare_data.py            # Split into train/val/test
│   ├── train_image.py             # Train EfficientNet-B0
│   ├── train_text.py              # Train DistilBERT
│   ├── evaluate_model.py          # Model evaluation
│   ├── upload_to_gcs.py           # Upload to GCS staging
│   └── promote_to_serving.py      # Promote staging → serving
│
├── training/                      # Training modules
│   ├── configs/                   # YAML configurations
│   ├── datasets/                  # Dataset loaders
│   ├── models/                    # Model definitions
│   └── utils/                     # Utilities (GCS, metrics)
│
├── models/                        # Local storage (gitignored)
│   ├── experiments/               # Training runs
│   └── production/                # Ready for upload
│
├── notebooks/                     # Experimentation
└── credentials/                   # GCS keys (gitignored)
```

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/YOUR_USERNAME/ml-shared-training.git
cd ml-shared-training

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure GCS
```bash
# Copy environment template
cp .env.example .env

# Add your service account key
# Place JSON file in: credentials/gcs-service-account.json

# Edit .env with your GCS project details
```

### 3. Download & Prepare Data
```bash
# Download public datasets (~250MB)
python scripts/download_datasets.py

# Split into train/val/test
python scripts/prepare_data.py
```

### 4. Train Models
```bash
# Train image model (EfficientNet-B0)
python scripts/train_image.py --config training/configs/image_config.yaml

# Train text model (DistilBERT)
python scripts/train_text.py --config training/configs/text_config.yaml
```

### 5. Upload to GCS
```bash
# Upload to staging for validation
python scripts/upload_to_gcs.py \
  --model-path models/experiments/run_20241020_001 \
  --model-type image \
  --version v1.0.0 \
  --environment staging

# After validation, promote to serving
python scripts/promote_to_serving.py \
  --model-type image \
  --version v1.0.0
```

---

## 🗂️ GCS Bucket Structure
```
gs://ml-shared-training/
├── image/
│   ├── staging/                   # Testing environment
│   │   ├── v1.0.0/
│   │   │   ├── model.pth
│   │   │   ├── config.yaml
│   │   │   └── metrics.json
│   │   └── v1.1.0/
│   │
│   └── serving/                   # Production environment
│       ├── v1.0.0/
│       ├── latest/                # Symlink to current version
│       └── v1.1.0/
│
└── text/
    ├── staging/
    └── serving/
```

---

## 📝 Semantic Versioning

Format: `vMAJOR.MINOR.PATCH`

- **MAJOR** (v2.0.0): Breaking changes, incompatible with previous versions
- **MINOR** (v1.1.0): New features, backward compatible
- **PATCH** (v1.0.1): Bug fixes, backward compatible

Examples:
- `v1.0.0` - Initial release
- `v1.1.0` - Added new preprocessing step
- `v1.1.1` - Fixed training bug
- `v2.0.0` - Changed model architecture

---

## 👥 Team Workflow

### Data Science Team
1. Experiment in `notebooks/`
2. Train models locally using `scripts/train_*.py`
3. Evaluate results with `scripts/evaluate_model.py`
4. Upload successful models to **staging**: `scripts/upload_to_gcs.py`
5. Notify Ops team for validation

### Dev/Ops Team
1. Validate models in **staging** environment
2. Run integration tests with serving architecture
3. Promote to **serving**: `scripts/promote_to_serving.py`
4. Update serving configurations to new version
5. Monitor model performance

---

## 🔒 Security

- ❌ **Never commit** credentials to git
- ✅ Use `.env` for local configuration
- ✅ Service account keys in `credentials/` (gitignored)
- ✅ Use least-privilege IAM roles for GCS access

---

## 🧪 Development
```bash
# Run tests
pytest tests/

# Format code
black .

# Lint code
flake8 training/ scripts/

# Start Jupyter
jupyter notebook
```

---

## 📚 Documentation

Coming soon:
- Training Guide
- GCS Setup Guide
- Model Registry Documentation
- Troubleshooting Guide

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

**Status**: 🚧 Initial Setup  
**Last Updated**: 2025-10-20  
**Maintained by**: Matthew Hong

### Prerequisites

- **Python 3.11.14** (exact version for consistency)
- [uv](https://github.com/astral-sh/uv) package manager

**Install Python 3.11.14:**

Using [pyenv](https://github.com/pyenv/pyenv) (recommended):
```bash
# Install pyenv if not already installed
curl https://pyenv.run | bash

# Install Python 3.11.14
pyenv install 3.11.14
pyenv local 3.11.14

# Verify
python --version  # Should show: Python 3.11.14
```

**Install uv:**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```