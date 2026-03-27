"""
Multimodal Inference Engine — Text, Image, Audio with RAG.
محرك الاستنتاج متعدد الوسائط — نص وصورة وصوت مع RAG
"""
import io
import base64
import time
from pathlib import Path
from typing import Optional, Generator
from dataclasses import dataclass
from loguru import logger

import torch
from PIL import Image

from src.model.loader import load_inference_model
from src.rag.engine import RAGEngine
from src.utils.config import AppConfig


@dataclass
class InferenceRequest:
    """Multimodal inference request."""
    text: str = ""
    image: Optional[bytes] = None       # Raw image bytes
    image_path: Optional[str] = None    # Or path to image file
    audio: Optional[bytes] = None       # Raw audio bytes
    audio_path: Optional[str] = None    # Or path to audio file
    use_rag: bool = True
    system_prompt: str = ""
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


@dataclass
class InferenceResponse:
    """Inference response with metadata."""
    text: str
    audio_transcript: str = ""
    rag_context: str = ""
    rag_sources: list = None
    latency_ms: float = 0.0
    tokens_generated: int = 0

    def __post_init__(self):
        if self.rag_sources is None:
            self.rag_sources = []


class AudioTranscriber:
    """
    Audio transcription with faster-whisper (preferred) or openai-whisper (fallback).
    تحويل الصوت لنص بـ faster-whisper (مفضّل) أو openai-whisper (بديل)

    TODO-8 FIX: faster-whisper uses CTranslate2 backend:
    - 4x faster inference than openai-whisper
    - ~50% less VRAM (critical when sharing GPU with LLM)
    - Supports int8 quantization for even more savings
    Falls back to openai-whisper if faster-whisper not installed.
    """

    def __init__(self, model_name: str = "openai/whisper-large-v3", language: str = "ar"):
        self.model = None
        self.model_name = model_name
        self.language = language
        self._backend = None  # "faster" or "openai"

    def _load(self):
        """Lazy load — try faster-whisper first, fallback to openai."""
        if self.model is not None:
            return

        size_map = {
            "openai/whisper-large-v3": "large-v3",
            "openai/whisper-medium": "medium",
            "openai/whisper-small": "small",
            "openai/whisper-base": "base",
        }
        size = size_map.get(self.model_name, "base")

        # Try faster-whisper first
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                size,
                device="cuda" if torch.cuda.is_available() else "cpu",
                compute_type="int8" if torch.cuda.is_available() else "float32",
            )
            self._backend = "faster"
            logger.info(f"Loaded faster-whisper: {size} (int8)")
            return
        except ImportError:
            pass

        # Fallback: openai-whisper
        try:
            import whisper
            self.model = whisper.load_model(size)
            self._backend = "openai"
            logger.info(f"Loaded openai-whisper: {size} (fallback)")
        except ImportError:
            logger.error("No whisper library found. Install: pip install faster-whisper  or  pip install openai-whisper")
            raise

    def transcribe(self, audio_bytes: Optional[bytes] = None, audio_path: Optional[str] = None) -> str:
        """Transcribe audio to text."""
        self._load()
        import tempfile
        import os

        # Resolve to file path
        temp_path = None
        target_path = None

        if audio_bytes:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name
            target_path = temp_path
        elif audio_path and Path(audio_path).exists():
            target_path = audio_path

        if not target_path:
            return ""

        try:
            if self._backend == "faster":
                segments, info = self.model.transcribe(
                    target_path,
                    language=self.language,
                    beam_size=5,
                    vad_filter=True,  # Skip silence — faster
                )
                text = " ".join(seg.text for seg in segments)
                return text.strip()
            else:
                # openai-whisper
                result = self.model.transcribe(target_path, language=self.language)
                return result["text"].strip()
        finally:
            if temp_path:
                os.unlink(temp_path)


class MultimodalInferenceEngine:
    """
    Production Multimodal Inference Engine.
    محرك الاستنتاج الإنتاجي متعدد الوسائط

    Combines: LLM + Vision + Audio (Whisper) + RAG
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load LLM
        logger.info("Initializing inference engine...")
        self.model, self.tokenizer, self.processor = load_inference_model(cfg)
        self.model.eval()

        # Initialize RAG
        self.rag = RAGEngine(cfg)
        logger.info("RAG engine initialized")

        # Initialize audio transcriber (lazy loaded)
        self.transcriber = AudioTranscriber(
            model_name=cfg.audio.model,
            language=cfg.audio.language,
        )

        logger.info(f"Inference engine ready on {self.device}")

    def _process_image(self, image_bytes: Optional[bytes] = None, image_path: Optional[str] = None) -> Optional[Image.Image]:
        """Process image input."""
        if image_bytes:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif image_path and Path(image_path).exists():
            return Image.open(image_path).convert("RGB")
        return None

    def _build_messages(self, request: InferenceRequest, audio_text: str = "") -> tuple:
        """
        Build chat messages with optional RAG context.
        بناء رسائل المحادثة مع سياق RAG اختياري

        FIX: Returns retrieved docs to avoid double-retrieval.
        Old code called rag.build_context() here then rag.retrieve() again
        in generate() — wasting 2x latency and compute.

        Returns:
            (messages, image, rag_context_str, rag_docs)
        """
        query = request.text
        if audio_text:
            query = f"[نص صوتي]: {audio_text}\n\n{query}" if query else audio_text

        # RAG: single retrieval, reuse results
        rag_context = ""
        rag_docs = []
        if request.use_rag and query:
            rag_docs = self.rag.retrieve(query)
            if rag_docs:
                context_parts = []
                for i, doc in enumerate(rag_docs, 1):
                    source = doc.metadata.get("source", doc.metadata.get("filename", "unknown"))
                    context_parts.append(
                        f"[مرجع {i}] (مصدر: {source}, درجة الصلة: {doc.score:.2f})\n{doc.content}"
                    )
                rag_context = "\n\n---\n\n".join(context_parts)

        system = request.system_prompt or (
            "أنت مساعد ذكي متعدد الوسائط. تفهم النصوص والصور والأصوات وتقدم إجابات دقيقة ومفيدة باللغة العربية."
        )

        if rag_context:
            system += f"\n\nاستخدم المراجع التالية في إجابتك:\n{rag_context}"

        messages = [{"role": "system", "content": system}]

        # Build user message
        user_content = []

        # Add image if present
        image = self._process_image(request.image, request.image_path)
        if image and self.processor:
            user_content.append({"type": "image"})

        # Add text
        user_content.append({"type": "text", "text": query})

        messages.append({"role": "user", "content": user_content if image else query})

        return messages, image, rag_context, rag_docs

    @torch.inference_mode()
    def generate(self, request: InferenceRequest) -> InferenceResponse:
        """
        Generate a response for a multimodal request.
        توليد إجابة لطلب متعدد الوسائط
        """
        start_time = time.time()

        # Transcribe audio if present
        audio_text = ""
        if request.audio or request.audio_path:
            try:
                audio_text = self.transcriber.transcribe(request.audio, request.audio_path)
                logger.info(f"Audio transcribed: {audio_text[:100]}...")
            except Exception as e:
                logger.error(f"Audio transcription failed: {e}")
                audio_text = "[فشل تحويل الصوت إلى نص]"

        # Build messages (RAG docs retrieved ONCE here, reused below)
        messages, image, rag_context, rag_docs = self._build_messages(request, audio_text)

        # Tokenize
        try:
            if self.processor and image:
                inputs = self.processor(
                    text=self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                    images=image,
                    return_tensors="pt",
                ).to(self.device)
            else:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True,
                    max_length=self.cfg.model.max_length,
                ).to(self.device)
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            return InferenceResponse(
                text=f"خطأ في معالجة المدخلات: {str(e)}",
                latency_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Generate with timeout protection
        # TODO-4 FIX: Added ThreadPoolExecutor timeout — prevents infinite GPU lock
        generation_kwargs = {
            "max_new_tokens": request.max_new_tokens,
            "temperature": max(request.temperature, 0.01),
            "top_p": request.top_p,
            "do_sample": request.temperature > 0.01,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
            timeout_sec = 120  # 2 minutes max per generation

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.model.generate, **inputs, **generation_kwargs)
                try:
                    outputs = future.result(timeout=timeout_sec)
                except FutureTimeout:
                    logger.error(f"Generation timed out after {timeout_sec}s")
                    return InferenceResponse(
                        text="خطأ: انتهت مهلة التوليد. حاول تقليل max_tokens أو تبسيط السؤال.",
                        latency_ms=round((time.time() - start_time) * 1000, 2),
                    )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error("GPU OOM during generation")
            return InferenceResponse(
                text="خطأ: ذاكرة GPU غير كافية. حاول تقليل max_tokens.",
                latency_ms=round((time.time() - start_time) * 1000, 2),
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return InferenceResponse(
                text=f"خطأ في التوليد: {str(e)}",
                latency_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Decode — only the new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        latency = (time.time() - start_time) * 1000

        # Reuse already-retrieved docs (FIX: no second retrieval)
        rag_sources = [
            {"source": d.metadata.get("source", ""), "score": round(d.score, 3)}
            for d in rag_docs
        ]

        return InferenceResponse(
            text=response_text.strip(),
            audio_transcript=audio_text,
            rag_context=rag_context[:500] if rag_context else "",
            rag_sources=rag_sources,
            latency_ms=round(latency, 2),
            tokens_generated=len(generated_ids),
        )

    @torch.inference_mode()
    def generate_stream(self, request: InferenceRequest) -> Generator[str, None, None]:
        """
        Stream generation token by token.
        توليد متدفق توكن بتوكن
        """
        from transformers import TextIteratorStreamer
        from threading import Thread

        audio_text = ""
        if request.audio or request.audio_path:
            audio_text = self.transcriber.transcribe(request.audio, request.audio_path)

        messages, image, _, _ = self._build_messages(request, audio_text)

        if self.processor and image:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device)
        else:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = {
            **inputs,
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "do_sample": request.temperature > 0,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for text_chunk in streamer:
            yield text_chunk

        thread.join()
