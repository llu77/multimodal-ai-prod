"""
Tests for multimodal dataset utilities.
اختبارات أدوات مجموعة البيانات متعددة الوسائط

Tests image loading, text building, and label masking.
Requires: torch (will skip if not installed).
"""
import io
import json
import base64
import pytest
from pathlib import Path
from PIL import Image

# Skip if torch not available
torch = pytest.importorskip("torch", reason="Multimodal dataset tests require torch")

from src.data.multimodal_dataset import (
    load_image_from_base64,
    load_image_from_path,
    extract_images_from_messages,
    build_text_from_messages,
)


# ═══════════════════════════════════════════
# Image Loading
# ═══════════════════════════════════════════

def _make_test_image_b64(width=32, height=32, color=(255, 0, 0)) -> str:
    """Create a small test image and return as base64."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TestLoadImageFromBase64:

    def test_valid_base64(self):
        b64 = _make_test_image_b64()
        img = load_image_from_base64(b64)
        assert img is not None
        assert isinstance(img, Image.Image)
        assert img.size == (32, 32)
        assert img.mode == "RGB"

    def test_data_uri_prefix_stripped(self):
        b64 = _make_test_image_b64()
        data_uri = f"data:image/png;base64,{b64}"
        img = load_image_from_base64(data_uri)
        assert img is not None

    def test_invalid_base64_returns_none(self):
        img = load_image_from_base64("not-valid-base64!!!")
        assert img is None

    def test_empty_string_returns_none(self):
        img = load_image_from_base64("")
        assert img is None


class TestLoadImageFromPath:

    def test_valid_path(self, tmp_dir):
        img_path = tmp_dir / "test.png"
        Image.new("RGB", (64, 64), color=(0, 255, 0)).save(str(img_path))

        img = load_image_from_path(str(img_path))
        assert img is not None
        assert img.size == (64, 64)

    def test_relative_path_with_root(self, tmp_dir):
        img_path = tmp_dir / "images" / "photo.png"
        img_path.parent.mkdir(parents=True)
        Image.new("RGB", (16, 16)).save(str(img_path))

        img = load_image_from_path("images/photo.png", data_root=str(tmp_dir))
        assert img is not None

    def test_nonexistent_path_returns_none(self):
        img = load_image_from_path("/nonexistent/path/image.png")
        assert img is None

    def test_corrupted_file_returns_none(self, tmp_dir):
        bad_path = tmp_dir / "bad.png"
        bad_path.write_text("this is not an image")
        img = load_image_from_path(str(bad_path))
        assert img is None


# ═══════════════════════════════════════════
# Image Extraction from Messages
# ═══════════════════════════════════════════

class TestExtractImages:

    def test_extracts_base64_image(self):
        b64 = _make_test_image_b64()
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": b64},
                {"type": "text", "text": "describe this"},
            ]}
        ]
        images = extract_images_from_messages(messages)
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)

    def test_extracts_path_image(self, tmp_dir):
        img_path = tmp_dir / "img.png"
        Image.new("RGB", (16, 16)).save(str(img_path))

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image_path": str(img_path)},
                {"type": "text", "text": "describe"},
            ]}
        ]
        images = extract_images_from_messages(messages)
        assert len(images) == 1

    def test_skips_placeholder_base64(self):
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": "<base64_encoded_image>"},
                {"type": "text", "text": "describe"},
            ]}
        ]
        images = extract_images_from_messages(messages)
        assert len(images) == 0

    def test_text_only_messages_return_empty(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        images = extract_images_from_messages(messages)
        assert len(images) == 0

    def test_multiple_images(self):
        b64_1 = _make_test_image_b64(color=(255, 0, 0))
        b64_2 = _make_test_image_b64(color=(0, 255, 0))
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": b64_1},
                {"type": "image", "image": b64_2},
                {"type": "text", "text": "compare these"},
            ]}
        ]
        images = extract_images_from_messages(messages)
        assert len(images) == 2

    def test_pil_image_direct(self):
        pil_img = Image.new("RGB", (8, 8), color=(0, 0, 255))
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": "what is this"},
            ]}
        ]
        images = extract_images_from_messages(messages)
        assert len(images) == 1
        assert images[0] is pil_img  # Same object


# ═══════════════════════════════════════════
# Text Building from Multimodal Messages
# ═══════════════════════════════════════════

class TestBuildTextFromMessages:

    def test_text_only_passthrough(self):
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "User question."},
            {"role": "assistant", "content": "Answer."},
        ]
        result = build_text_from_messages(messages)
        assert len(result) == 3
        assert result[0]["content"] == "System prompt."
        assert result[1]["content"] == "User question."

    def test_image_replaced_with_placeholder(self):
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": "some_base64"},
                {"type": "text", "text": "describe this"},
            ]}
        ]
        result = build_text_from_messages(messages)
        assert "<image>" in result[0]["content"]
        assert "describe this" in result[0]["content"]

    def test_multiple_images_multiple_placeholders(self):
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": "img1"},
                {"type": "text", "text": "compare"},
                {"type": "image", "image": "img2"},
            ]}
        ]
        result = build_text_from_messages(messages)
        content = result[0]["content"]
        assert content.count("<image>") == 2

    def test_preserves_role(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {"role": "assistant", "content": "a"},
        ]
        result = build_text_from_messages(messages)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_empty_content_handled(self):
        messages = [{"role": "user", "content": ""}]
        result = build_text_from_messages(messages)
        assert result[0]["content"] == ""

    def test_non_dict_content_parts(self):
        # Some formats have string parts in list
        messages = [
            {"role": "user", "content": ["plain string part", "another part"]}
        ]
        result = build_text_from_messages(messages)
        assert "plain string part" in result[0]["content"]
