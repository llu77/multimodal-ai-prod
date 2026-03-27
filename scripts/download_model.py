#!/usr/bin/env python3
"""
Download base model and required components.
تحميل النموذج الأساسي والمكونات المطلوبة
"""
import argparse
from loguru import logger


def download_model(model_name: str):
    """Download model from Hugging Face Hub."""
    from huggingface_hub import snapshot_download

    logger.info(f"Downloading model: {model_name}")
    path = snapshot_download(
        model_name,
        local_dir=f"./models/base/{model_name.split('/')[-1]}",
        ignore_patterns=["*.gguf", "*.ggml"],
    )
    logger.info(f"Model downloaded to: {path}")
    return path


def download_embedding_model(model_name: str):
    """Download embedding model for RAG."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Downloading embedding model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    logger.info("Embedding model ready")
    return model


def download_whisper(size: str = "base"):
    """Download Whisper model for audio transcription."""
    try:
        from faster_whisper import WhisperModel
        logger.info(f"Downloading faster-whisper model: {size}")
        model = WhisperModel(size, device="cpu", compute_type="int8")
        logger.info("faster-whisper model ready")
        return model
    except ImportError:
        logger.error("faster-whisper not installed. Please install it to download the model.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model-only", action="store_true", help="Download base model only")
    parser.add_argument("--all", action="store_true", help="Download all models")
    args = parser.parse_args()

    from src.utils.config import load_config
    cfg = load_config(args.config)

    # Always download base model
    download_model(cfg.model.base_model)

    if args.all or not args.model_only:
        download_embedding_model(cfg.rag.embedding_model)

        whisper_sizes = {
            "openai/whisper-large-v3": "large-v3",
            "openai/whisper-medium": "medium",
            "openai/whisper-small": "small",
            "openai/whisper-base": "base",
        }
        whisper_size = whisper_sizes.get(cfg.audio.model, "base")
        download_whisper(whisper_size)

    logger.info("✅ All downloads complete!")
