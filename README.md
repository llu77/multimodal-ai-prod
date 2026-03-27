# 🚀 Multimodal AI Production System
# نظام ذكاء اصطناعي متعدد الوسائط - إنتاجي

## Overview | نظرة عامة
Production-grade multimodal AI system built on **Phi-4 Multimodal** with RAG integration.
Supports text, image, and audio inputs with retrieval-augmented generation.

## Architecture | البنية المعمارية
```
┌─────────────────────────────────────────────────────┐
│                   API Gateway (FastAPI)               │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Text     │  │  Image   │  │  Audio            │  │
│  │  Input    │  │  Input   │  │  Input (Whisper)  │  │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │
│       │              │                  │             │
│       └──────────────┼──────────────────┘             │
│                      ▼                                │
│         ┌────────────────────────┐                    │
│         │   RAG Retrieval Engine │                    │
│         │   (pgvector/Supabase)  │                    │
│         └───────────┬────────────┘                    │
│                     ▼                                 │
│         ┌────────────────────────┐                    │
│         │  Phi-4 Multimodal      │                    │
│         │  (QLoRA Fine-tuned)    │                    │
│         └───────────┬────────────┘                    │
│                     ▼                                 │
│         ┌────────────────────────┐                    │
│         │   Response Generator   │                    │
│         └────────────────────────┘                    │
└─────────────────────────────────────────────────────┘
```

## Quick Start | البداية السريعة

```bash
# 1. Clone and setup
git clone <your-repo>
cd multimodal-ai-prod

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your settings

# 4. Download base model
python scripts/download_model.py

# 5. Prepare data
python -m src.data.prepare_dataset --config config/config.yaml

# 6. Fine-tune
python -m src.training.train --config config/config.yaml

# 7. Start RAG indexing
python -m src.rag.indexer --config config/config.yaml

# 8. Launch API server
python -m src.api.server --config config/config.yaml
```

## 🪟 Windows Users | مستخدمي ويندوز
- **Setup Script**: Use `.\scripts\setup.ps1` in PowerShell instead of `setup.sh`
- **QLoRA (bitsandbytes)**: The `bitsandbytes` library used for 4-bit quantization does not officially support native Windows. To train the model, you must either:
  1. Run this project inside **WSL2** (Windows Subsystem for Linux) - *Recommended*, OR
  2. Disable 4-bit quantization in `config/config.yaml` (`quantization.enabled: false`)

## Hardware Requirements | متطلبات الأجهزة
- **Minimum**: 1x GPU with 16GB VRAM (RTX 4060 Ti 16GB)
- **Recommended**: 1x GPU with 24GB VRAM (RTX 3090/4090)
- **RAM**: 32GB+ system RAM
- **Storage**: 100GB+ SSD

## Project Structure | هيكل المشروع
```
multimodal-ai-prod/
├── config/
│   ├── config.yaml          # Main configuration
│   └── config.example.yaml  # Example config template
├── src/
│   ├── data/                # Data processing pipeline
│   ├── model/               # Model loading & management
│   ├── rag/                 # RAG engine (retrieval + indexing)
│   ├── training/            # Fine-tuning pipeline (LoRA/QLoRA)
│   ├── inference/           # Inference engine
│   ├── api/                 # FastAPI production server
│   └── utils/               # Shared utilities
├── scripts/                 # Setup & utility scripts
├── data/                    # Data directory
├── models/                  # Saved models & adapters
├── tests/                   # Test suite
├── requirements.txt
└── README.md
```
