#!/bin/bash
# ============================================
# Quick Start Setup Script
# سكربت الإعداد السريع
# ============================================
set -e

echo "🚀 Multimodal AI Production Setup"
echo "=================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install Python 3.11+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅ Python $PYTHON_VERSION found"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  No NVIDIA GPU detected. Training will be slow on CPU."
fi

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Setup .env
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created .env file — edit it with your credentials"
fi

# Create directories
mkdir -p models/{base,adapters,fine-tuned,merged}
mkdir -p data/{raw,processed,embeddings}
mkdir -p logs

# Create sample dataset
echo ""
echo "📊 Creating sample dataset..."
python3 -m src.data.prepare_dataset --create-sample --config config/config.yaml

echo ""
echo "============================================"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your credentials"
echo "  2. Edit config/config.yaml to customize"
echo "  3. Download model:"
echo "     python scripts/download_model.py --all"
echo "  4. Add your training data to data/processed/train.jsonl"
echo "  5. Train:"
echo "     python -m src.training.train --config config/config.yaml"
echo "  6. Start server:"
echo "     python -m src.api.server"
echo "============================================"
