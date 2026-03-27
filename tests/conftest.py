"""
Shared test fixtures — no GPU required, no model downloads.
ملحقات اختبار مشتركة — لا تحتاج GPU ولا تحميل نماذج
"""
import os
import json
import tempfile
import pytest
from pathlib import Path


@pytest.fixture
def tmp_dir():
    """Create a temporary directory, cleaned up after test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_config_yaml(tmp_dir):
    """Write a minimal config YAML and return its path."""
    cfg = {
        "model": {
            "base_model": "test/model",
            "quantization": {"enabled": False, "bits": 4},
            "output_dir": str(tmp_dir / "out"),
            "adapter_dir": str(tmp_dir / "adapter"),
            "merged_dir": str(tmp_dir / "merged"),
            "max_length": 512,
        },
        "audio": {"model": "openai/whisper-base", "language": "ar"},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.0002,
            "dataset_path": str(tmp_dir / "data"),
        },
        "rag": {
            "embedding_model": "test/embed",
            "chunk_size": 100,
            "chunk_overlap": 20,
            "vector_store": "chromadb",
            "chromadb": {
                "persist_dir": str(tmp_dir / "chroma"),
                "collection_name": "test_collection",
            },
            "retrieval": {"top_k": 3, "similarity_threshold": 0.5, "rerank": False},
        },
        "server": {
            "host": "127.0.0.1",
            "port": 9999,
            "api_key": "test-key-123",
            "rate_limit": 100,
            "cors_origins": ["http://localhost:3000"],
        },
    }
    path = tmp_dir / "config.yaml"
    import yaml
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return str(path)


@pytest.fixture
def sample_env_config_yaml(tmp_dir):
    """Config with environment variable references."""
    cfg = {
        "server": {
            "api_key": "${TEST_API_KEY}",
        }
    }
    path = tmp_dir / "env_config.yaml"
    import yaml
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return str(path)


@pytest.fixture
def sample_train_jsonl(tmp_dir):
    """Create a sample training JSONL file and return its path."""
    data_dir = tmp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        {
            "messages": [
                {"role": "system", "content": "أنت مساعد ذكي."},
                {"role": "user", "content": "ما عاصمة السعودية؟"},
                {"role": "assistant", "content": "عاصمة المملكة العربية السعودية هي مدينة الرياض."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "أنت مساعد طبي."},
                {"role": "user", "content": [
                    {"type": "image", "image_path": "test.png"},
                    {"type": "text", "text": "صف هذه الصورة."},
                ]},
                {"role": "assistant", "content": "هذه صورة اختبار."},
            ]
        },
    ]

    path = data_dir / "train.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Create a tiny test image
    try:
        from PIL import Image
        img = Image.new("RGB", (32, 32), color=(100, 150, 200))
        img.save(str(data_dir / "test.png"))
    except ImportError:
        pass

    return str(path)


# ── Arabic text fixtures ──

ARABIC_PARAGRAPH = (
    "الذكاء الاصطناعي هو فرع من علوم الحاسوب. "
    "يهدف إلى إنشاء أنظمة قادرة على أداء مهام تتطلب ذكاءً بشرياً. "
    "تشمل هذه المهام التعلم والاستدلال وحل المشكلات. "
    "من أبرز تطبيقاته معالجة اللغة الطبيعية والرؤية الحاسوبية."
)

ENGLISH_PARAGRAPH = (
    "Artificial intelligence is a branch of computer science. "
    "It aims to create systems capable of performing tasks that require human intelligence. "
    "These tasks include learning, reasoning, and problem-solving. "
    "Notable applications include natural language processing and computer vision."
)

MIXED_PARAGRAPH = (
    "استخدمنا نموذج Transformer مع attention mechanism. "
    "تم التدريب باستخدام PyTorch على GPU واحد. "
    "النتائج أظهرت تحسناً بنسبة 15% في accuracy."
)
