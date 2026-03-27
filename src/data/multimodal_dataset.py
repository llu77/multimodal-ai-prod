"""
Multimodal Dataset & DataCollator for training with text + images.
مجموعة بيانات ومُجمّع بيانات متعدد الوسائط للتدريب بالنص والصور

This is the core fix for TODO-1: the old pipeline discarded images during
training and only used text. This module properly processes images through
the model's processor and feeds them alongside text tokens.

Architecture:
    JSONL sample → MultimodalDataset.__getitem__()
        ├─ Extract text messages → apply_chat_template → input_ids
        ├─ Extract images (base64/path) → processor → pixel_values
        └─ Build labels (mask non-assistant tokens with -100)
    
    Batch of samples → MultimodalDataCollator.__call__()
        ├─ Pad input_ids + attention_mask to max length
        ├─ Stack pixel_values (handle variable image counts)
        └─ Pad labels to match input_ids
"""
import io
import json
import base64
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass
from loguru import logger

import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image


# ────────────────── Image Loading Utilities ──────────────────

def load_image_from_base64(b64_string: str) -> Optional[Image.Image]:
    """Decode base64 string to PIL Image."""
    try:
        if b64_string.startswith("data:"):
            # Strip data URI prefix: data:image/png;base64,...
            b64_string = b64_string.split(",", 1)[1]
        raw = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to decode base64 image: {e}")
        return None


def load_image_from_path(image_path: str, data_root: str = "") -> Optional[Image.Image]:
    """Load PIL Image from file path."""
    path = Path(image_path)
    if not path.is_absolute() and data_root:
        path = Path(data_root) / path
    if path.exists():
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
    return None


def extract_images_from_messages(messages: list[dict], data_root: str = "") -> list[Image.Image]:
    """
    Extract all images from a conversation's messages.
    استخراج جميع الصور من رسائل المحادثة

    Supports:
    - base64 inline: {"type": "image", "image": "base64..."}
    - file path:     {"type": "image", "image_path": "path/to/img.jpg"}
    - PIL object:    {"type": "image", "image": <PIL.Image>}
    """
    images = []
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue

        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image":
                continue

            img = None

            # Case 1: Already a PIL Image
            if isinstance(part.get("image"), Image.Image):
                img = part["image"]

            # Case 2: base64 string
            elif isinstance(part.get("image"), str):
                b64 = part["image"]
                if b64 and b64 != "<base64_encoded_image>":  # Skip placeholder
                    img = load_image_from_base64(b64)

            # Case 3: File path
            if img is None and part.get("image_path"):
                img = load_image_from_path(part["image_path"], data_root)

            if img is not None:
                images.append(img)

    return images


def build_text_from_messages(messages: list[dict]) -> list[dict]:
    """
    Convert multimodal messages to text-only format for chat template.
    Replaces image parts with <image> placeholder token.
    يحوّل الرسائل متعددة الوسائط إلى نص فقط مع عناصر نائبة للصور
    """
    text_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if isinstance(content, str):
            text_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Multimodal: merge text parts, insert <image> for image parts
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(part["text"])
                    elif part.get("type") == "image":
                        parts.append("<image>")
                elif isinstance(part, str):
                    parts.append(part)
            text_messages.append({"role": role, "content": " ".join(parts)})
        else:
            text_messages.append({"role": role, "content": str(content)})

    return text_messages


# ────────────────── Label Masking ──────────────────

def create_labels_with_masking(
    input_ids: list[int],
    tokenizer,
    messages: list[dict],
) -> list[int]:
    """
    Create training labels that mask non-assistant tokens with -100.
    إنشاء ملصقات التدريب التي تحجب التوكنات غير المساعدة بـ -100

    Only compute loss on assistant responses, not on system/user prompts.
    This prevents the model from learning to generate user messages.

    Strategy:
    1. Encode the full conversation → input_ids
    2. Encode everything up to each assistant response → prefix_ids
    3. Mask all prefix tokens with -100
    """
    labels = list(input_ids)  # Copy

    # Find assistant response boundaries
    # Method: encode conversation incrementally to find where assistant content starts
    try:
        # Build prefix: everything except last assistant message
        text_msgs = build_text_from_messages(messages)
        prefix_msgs = []
        assistant_start = len(input_ids)  # Default: mask everything

        for i, msg in enumerate(text_msgs):
            prefix_msgs.append(msg)
            if msg["role"] == "assistant" and i == len(text_msgs) - 1:
                # This is the last assistant message — everything before is prefix
                prefix_without_last = text_msgs[:i]
                prefix_without_last.append({"role": "assistant", "content": ""})
                try:
                    prefix_text = tokenizer.apply_chat_template(
                        prefix_without_last,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
                    assistant_start = len(prefix_ids)
                except Exception:
                    # Fallback: mask first 60% as a rough heuristic
                    assistant_start = int(len(input_ids) * 0.6)

        # Mask everything before assistant response
        for i in range(min(assistant_start, len(labels))):
            labels[i] = -100

    except Exception as e:
        logger.debug(f"Label masking fallback: {e}")
        # Conservative fallback: mask first half
        mid = len(labels) // 2
        for i in range(mid):
            labels[i] = -100

    return labels


# ────────────────── Multimodal Dataset ──────────────────

class MultimodalDataset(TorchDataset):
    """
    Dataset that properly loads text + images for multimodal training.
    مجموعة بيانات تحمّل النص والصور بشكل صحيح للتدريب

    Each item returns:
    - input_ids: tokenized conversation with <image> placeholders
    - attention_mask: standard attention mask
    - pixel_values: processed image tensors (or None for text-only)
    - labels: input_ids with non-assistant tokens masked as -100
    - image_sizes: original image dimensions (needed by some models)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        processor,
        max_length: int = 4096,
        data_root: str = "",
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.data_root = data_root

        # Load all samples
        self.samples = []
        path = Path(data_path)

        if path.is_file():
            self.samples = self._load_jsonl(str(path))
        elif path.is_dir():
            for f in sorted(path.glob("*.jsonl")):
                self.samples.extend(self._load_jsonl(str(f)))

        # Separate into image and text-only for logging
        img_count = sum(1 for s in self.samples if self._has_images(s))
        logger.info(
            f"MultimodalDataset loaded: {len(self.samples)} total "
            f"({img_count} with images, {len(self.samples) - img_count} text-only)"
        )

    def _load_jsonl(self, path: str) -> list[dict]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line: {e}")
        return data

    def _has_images(self, sample: dict) -> bool:
        """Check if sample contains image content."""
        for msg in sample.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        img_val = part.get("image", "")
                        if img_val and img_val != "<base64_encoded_image>":
                            return True
                        if part.get("image_path"):
                            return True
        return False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Process a single sample into model-ready tensors.
        معالجة عينة واحدة إلى tensors جاهزة للنموذج
        """
        sample = self.samples[idx]
        messages = sample["messages"]

        # Extract images
        images = extract_images_from_messages(messages, self.data_root)

        # Build text version of messages (with <image> placeholders)
        text_messages = build_text_from_messages(messages)

        # Apply chat template to get formatted text
        try:
            formatted_text = self.tokenizer.apply_chat_template(
                text_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback formatting
            parts = []
            for msg in text_messages:
                parts.append(f"<|{msg['role']}|>\n{msg['content']}")
            formatted_text = "\n".join(parts)

        # Process with model's processor (handles both text and images)
        if images and self.processor:
            try:
                processed = self.processor(
                    text=formatted_text,
                    images=images,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                )
            except Exception as e:
                logger.warning(f"Processor failed for sample {idx} with images, falling back to text-only: {e}")
                images = []
                processed = self._tokenize_text_only(formatted_text)
        else:
            processed = self._tokenize_text_only(formatted_text)

        # Squeeze batch dimension (processor returns [1, seq_len])
        result = {}
        for key, val in processed.items():
            if isinstance(val, torch.Tensor):
                if val.dim() > 1 and val.size(0) == 1:
                    result[key] = val.squeeze(0)
                else:
                    result[key] = val
            else:
                result[key] = val

        # Create labels with assistant-only masking
        input_ids = result["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            ids_list = input_ids.tolist()
        else:
            ids_list = input_ids

        labels = create_labels_with_masking(ids_list, self.tokenizer, messages)
        result["labels"] = torch.tensor(labels, dtype=torch.long)

        return result

    def _tokenize_text_only(self, text: str) -> dict:
        """Tokenize text without images."""
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        return encoded


# ────────────────── Data Collator ──────────────────

@dataclass
class MultimodalDataCollator:
    """
    Collates multimodal samples into padded batches.
    يُجمّع العينات متعددة الوسائط في دفعات مبطّنة

    Handles:
    - Variable-length text sequences → pad to max in batch
    - Variable image counts per sample → stack or pad pixel_values
    - Labels padding with -100 (ignored in loss)
    - Mixed batches (some with images, some without)
    """
    tokenizer: Any
    max_length: int = 4096

    def __call__(self, features: list[dict]) -> dict:
        """Collate a batch of multimodal features."""
        batch = {}

        # ── Pad text sequences ──
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        max_len = min(
            max(f["input_ids"].size(-1) if isinstance(f["input_ids"], torch.Tensor)
                else len(f["input_ids"]) for f in features),
            self.max_length,
        )

        pad_token_id = self.tokenizer.pad_token_id or 0

        for f in features:
            ids = f["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            if ids.dim() > 1:
                ids = ids.squeeze(0)

            attn = f.get("attention_mask")
            if attn is not None:
                if not isinstance(attn, torch.Tensor):
                    attn = torch.tensor(attn, dtype=torch.long)
                if attn.dim() > 1:
                    attn = attn.squeeze(0)
            else:
                attn = torch.ones_like(ids)

            labels = f.get("labels")
            if labels is not None:
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=torch.long)
                if labels.dim() > 1:
                    labels = labels.squeeze(0)
            else:
                labels = ids.clone()

            # Truncate if needed
            ids = ids[:max_len]
            attn = attn[:max_len]
            labels = labels[:max_len]

            # Pad
            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
                attn = torch.cat([attn, torch.zeros(pad_len, dtype=torch.long)])
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

            input_ids_list.append(ids)
            attention_mask_list.append(attn)
            labels_list.append(labels)

        batch["input_ids"] = torch.stack(input_ids_list)
        batch["attention_mask"] = torch.stack(attention_mask_list)
        batch["labels"] = torch.stack(labels_list)

        # ── Handle pixel_values (images) ──
        # Collect all image-related keys from features
        image_keys = [k for k in features[0].keys()
                      if k not in ("input_ids", "attention_mask", "labels")]

        for key in image_keys:
            values = [f[key] for f in features if key in f]
            if not values:
                continue

            if all(isinstance(v, torch.Tensor) for v in values):
                # All same shape → stack
                shapes = [v.shape for v in values]
                if len(set(shapes)) == 1:
                    batch[key] = torch.stack(values)
                else:
                    # Variable shapes — pad to max
                    # Common for pixel_values with different image sizes
                    max_shape = [max(s[i] for s in shapes) for i in range(len(shapes[0]))]
                    padded = []
                    for v in values:
                        pad_sizes = []
                        for dim in range(len(max_shape) - 1, -1, -1):
                            pad_sizes.extend([0, max_shape[dim] - v.shape[dim]])
                        padded.append(torch.nn.functional.pad(v, pad_sizes))
                    batch[key] = torch.stack(padded)
            elif all(isinstance(v, (list, tuple)) for v in values):
                # Lists (e.g., image_sizes) — keep as list
                batch[key] = values
            # Skip non-tensor, non-list values

        return batch


# ────────────────── Validation ──────────────────

def validate_dataset(dataset: MultimodalDataset, num_samples: int = 3) -> dict:
    """
    Validate dataset by loading a few samples and checking shapes.
    تحقق من مجموعة البيانات عبر تحميل عينات وفحص الأشكال
    """
    stats = {
        "total": len(dataset),
        "valid": 0,
        "text_only": 0,
        "with_images": 0,
        "errors": [],
    }

    check_count = min(num_samples, len(dataset))
    for i in range(check_count):
        try:
            sample = dataset[i]
            stats["valid"] += 1

            has_pixels = "pixel_values" in sample
            if has_pixels:
                stats["with_images"] += 1
            else:
                stats["text_only"] += 1

            logger.info(
                f"Sample {i}: input_ids={sample['input_ids'].shape}, "
                f"labels={sample['labels'].shape}, "
                f"has_image={'pixel_values' in sample}"
            )
        except Exception as e:
            stats["errors"].append(f"Sample {i}: {str(e)}")
            logger.error(f"Sample {i} failed: {e}")

    logger.info(f"Validation: {stats['valid']}/{check_count} valid, "
                f"{stats['with_images']} with images, {len(stats['errors'])} errors")
    return stats
