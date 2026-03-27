"""
Tests for the API server — security, validation, rate limiting.
اختبارات خادم API — الأمان، التحقق، تحديد المعدل

These tests use FastAPI TestClient and mock the inference engine.
Requires: torch, transformers (will skip if not installed).
"""
import json
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

# Skip entire module if torch not available (needed by server imports)
torch = pytest.importorskip("torch", reason="API tests require torch")

from fastapi.testclient import TestClient


# ── Mock the engine before importing the app ──

@dataclass
class MockInferenceResponse:
    text: str = "إجابة تجريبية"
    audio_transcript: str = ""
    rag_context: str = ""
    rag_sources: list = None
    latency_ms: float = 50.0
    tokens_generated: int = 10

    def __post_init__(self):
        if self.rag_sources is None:
            self.rag_sources = []


@pytest.fixture
def mock_engine():
    """Create a mock inference engine."""
    engine = MagicMock()
    engine.generate.return_value = MockInferenceResponse()
    engine.rag = MagicMock()
    engine.rag.index_text.return_value = 5
    engine.rag.retrieve.return_value = []
    engine.transcriber = MagicMock()
    engine.transcriber.transcribe.return_value = "نص صوتي تجريبي"
    return engine


@pytest.fixture
def app_client(mock_engine, sample_config_yaml):
    """Create test client with mocked engine."""
    # Import after fixtures are ready
    import src.api.server as server_module
    from src.utils.config import load_config

    # Set global state
    server_module.cfg = load_config(sample_config_yaml)
    server_module.engine = mock_engine
    server_module.rag = mock_engine.rag

    client = TestClient(server_module.app)
    yield client

    # Cleanup
    server_module.engine = None
    server_module.rag = None
    server_module.cfg = None


# ═══════════════════════════════════════════
# Health Check
# ═══════════════════════════════════════════

class TestHealthEndpoint:

    def test_health_returns_200(self, app_client):
        resp = app_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_health_shows_model_info(self, app_client):
        resp = app_client.get("/health")
        data = resp.json()
        assert "model" in data
        assert "device" in data


# ═══════════════════════════════════════════
# Authentication
# ═══════════════════════════════════════════

class TestAuthentication:

    def test_valid_api_key_accepted(self, app_client):
        resp = app_client.post(
            "/chat",
            json={"message": "مرحبا"},
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200

    def test_invalid_api_key_rejected(self, app_client):
        resp = app_client.post(
            "/chat",
            json={"message": "مرحبا"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_missing_api_key_rejected(self, app_client):
        resp = app_client.post(
            "/chat",
            json={"message": "مرحبا"},
        )
        assert resp.status_code == 401


# ═══════════════════════════════════════════
# Chat Endpoint
# ═══════════════════════════════════════════

class TestChatEndpoint:

    def test_basic_chat(self, app_client):
        resp = app_client.post(
            "/chat",
            json={"message": "ما هو الذكاء الاصطناعي؟"},
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert data["response"] == "إجابة تجريبية"
        assert "latency_ms" in data

    def test_chat_with_custom_params(self, app_client):
        resp = app_client.post(
            "/chat",
            json={
                "message": "سؤال",
                "temperature": 0.5,
                "max_tokens": 100,
                "use_rag": False,
            },
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200

    def test_empty_message_rejected(self, app_client):
        resp = app_client.post(
            "/chat",
            json={},
            headers={"X-API-Key": "test-key-123"},
        )
        # Pydantic should reject missing 'message' field
        assert resp.status_code == 422

    def test_temperature_out_of_range(self, app_client):
        resp = app_client.post(
            "/chat",
            json={"message": "test", "temperature": 5.0},
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 422  # ge=0.0, le=2.0

    def test_timing_header_present(self, app_client):
        resp = app_client.post(
            "/chat",
            json={"message": "test"},
            headers={"X-API-Key": "test-key-123"},
        )
        assert "X-Response-Time-Ms" in resp.headers


# ═══════════════════════════════════════════
# RAG Endpoints
# ═══════════════════════════════════════════

class TestRAGEndpoints:

    def test_index_text(self, app_client):
        resp = app_client.post(
            "/rag/index",
            json={"text": "نص تجريبي للفهرسة", "metadata": {"source": "test"}},
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "indexed"
        assert "chunks" in data

    def test_search(self, app_client):
        resp = app_client.post(
            "/rag/search",
            json={"query": "بحث تجريبي", "top_k": 3},
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "query" in data
        assert "results" in data


# ═══════════════════════════════════════════
# Path Traversal Protection
# ═══════════════════════════════════════════

class TestPathTraversal:

    def test_directory_outside_data_rejected(self, app_client):
        resp = app_client.post(
            "/rag/index/directory",
            data={"dir_path": "/etc/passwd"},
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 403

    def test_relative_traversal_rejected(self, app_client):
        resp = app_client.post(
            "/rag/index/directory",
            data={"dir_path": "../../etc"},
            headers={"X-API-Key": "test-key-123"},
        )
        # Should be either 403 (blocked) or 404 (not found inside data/)
        assert resp.status_code in (403, 404)


# ═══════════════════════════════════════════
# File Upload Validation
# ═══════════════════════════════════════════

class TestFileUploadValidation:

    def test_transcribe_endpoint(self, app_client):
        # Create a tiny fake audio file
        fake_audio = b"\x00" * 100
        resp = app_client.post(
            "/transcribe",
            files={"audio": ("test.wav", fake_audio, "audio/wav")},
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "transcript" in data

    def test_multimodal_chat_with_image(self, app_client):
        # Create a tiny PNG
        from PIL import Image
        import io
        img = Image.new("RGB", (8, 8), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = app_client.post(
            "/chat/multimodal",
            data={"message": "صف هذه الصورة", "use_rag": "false"},
            files={"image": ("test.png", buf.getvalue(), "image/png")},
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200


# ═══════════════════════════════════════════
# Rate Limiting
# ═══════════════════════════════════════════

class TestRateLimiting:

    def test_rate_limit_not_triggered_under_limit(self, app_client):
        # Config has rate_limit=100, so 5 requests should be fine
        for _ in range(5):
            resp = app_client.post(
                "/chat",
                json={"message": "test"},
                headers={"X-API-Key": "test-key-123"},
            )
            assert resp.status_code == 200
