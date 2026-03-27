"""
Production API Server — FastAPI with Multimodal + RAG endpoints.
خادم API الإنتاجي — FastAPI مع نقاط نهاية متعددة الوسائط + RAG
"""
import time
import asyncio
import secrets
import torch
from typing import Optional
from collections import defaultdict
from contextlib import asynccontextmanager
from loguru import logger

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from src.inference.engine import MultimodalInferenceEngine, InferenceRequest
from src.rag.engine import RAGEngine
from src.utils.config import load_config, AppConfig

# ─────────────────── Global State ───────────────────

engine: Optional[MultimodalInferenceEngine] = None
rag: Optional[RAGEngine] = None
cfg: Optional[AppConfig] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup, cleanup on shutdown."""
    global engine, rag, cfg
    logger.info("Starting Multimodal AI Server...")

    cfg = load_config("config/config.yaml")

    # Update CORS origins from config (after config is loaded)
    cors_origins = cfg.server.cors_origins if cfg.server.cors_origins != ["*"] else [
        "http://localhost:3000", "http://localhost:8000"
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    engine = MultimodalInferenceEngine(cfg)
    rag = engine.rag

    logger.info("Server ready!")
    yield

    logger.info("Shutting down...")
    del engine
    torch.cuda.empty_cache()


app = FastAPI(
    title="Multimodal AI API",
    description="نظام ذكاء اصطناعي متعدد الوسائط مع RAG — Production API",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────── Middleware ───────────────────
# NOTE: CORS middleware is now added in lifespan() after config is loaded.
# This ensures config values are applied correctly.


# ─────────────────── Rate Limiter ───────────────────

_rate_limit_store: dict[str, list[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Simple in-memory rate limiter.
    FIX: Config had rate_limit but it was never enforced.
    """
    if cfg and cfg.server.rate_limit > 0:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = 60.0  # 1 minute window

        # Clean old entries
        _rate_limit_store[client_ip] = [
            t for t in _rate_limit_store[client_ip] if now - t < window
        ]

        if len(_rate_limit_store[client_ip]) >= cfg.server.rate_limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded / تجاوز حد الطلبات"},
                headers={"Retry-After": "60"},
            )

        _rate_limit_store[client_ip].append(now)

    return await call_next(request)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add response timing header."""
    start = time.time()
    response = await call_next(request)
    response.headers["X-Response-Time-Ms"] = str(round((time.time() - start) * 1000, 2))
    return response


# ─────────────────── Auth ───────────────────

async def verify_api_key(request: Request):
    """
    API key verification — timing-safe comparison.
    FIX: Old code used == which is vulnerable to timing attacks.
    """
    if cfg and cfg.server.api_key:
        api_key = request.headers.get("X-API-Key", "")
        if not secrets.compare_digest(api_key, cfg.server.api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")


# ─────────────────── Request/Response Models ───────────────────

class ChatRequest(BaseModel):
    """Text chat request."""
    message: str = Field(..., description="User message / رسالة المستخدم")
    system_prompt: str = Field(default="", description="System prompt / تعليمات النظام")
    use_rag: bool = Field(default=True, description="Enable RAG retrieval / تفعيل RAG")
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False, description="Stream response / بث الإجابة")


class ChatResponse(BaseModel):
    """Chat response with metadata."""
    response: str
    audio_transcript: str = ""
    rag_sources: list = []
    latency_ms: float = 0.0
    tokens_generated: int = 0


class IndexRequest(BaseModel):
    """Document indexing request."""
    text: str = Field(..., description="Text content to index")
    metadata: dict = Field(default_factory=dict)


class SearchRequest(BaseModel):
    """RAG search request."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=20)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model: str = ""
    device: str = ""
    rag_store: str = ""


# ─────────────────── Endpoints ───────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint / فحص الصحة"""
    import torch
    return HealthResponse(
        status="healthy" if engine else "loading",
        model=cfg.model.base_model if cfg else "",
        device="cuda" if torch.cuda.is_available() else "cpu",
        rag_store=cfg.rag.vector_store if cfg else "",
    )


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest):
    """
    Text chat endpoint with RAG.
    نقطة نهاية المحادثة النصية مع RAG
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.stream:
        return StreamingResponse(
            _stream_chat(request),
            media_type="text/event-stream",
        )

    req = InferenceRequest(
        text=request.message,
        use_rag=request.use_rag,
        system_prompt=request.system_prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )

    # Run inference in thread pool to not block event loop
    result = await asyncio.to_thread(engine.generate, req)

    return ChatResponse(
        response=result.text,
        audio_transcript=result.audio_transcript,
        rag_sources=result.rag_sources,
        latency_ms=result.latency_ms,
        tokens_generated=result.tokens_generated,
    )


async def _stream_chat(request: ChatRequest):
    """SSE streaming for chat."""
    import json
    req = InferenceRequest(
        text=request.message,
        use_rag=request.use_rag,
        system_prompt=request.system_prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=True,
    )
    for chunk in engine.generate_stream(req):
        yield f"data: {json.dumps({'text': chunk})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/chat/multimodal", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat_multimodal(
    message: str = Form(default=""),
    system_prompt: str = Form(default=""),
    use_rag: bool = Form(default=True),
    max_tokens: int = Form(default=2048),
    temperature: float = Form(default=0.7),
    image: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
):
    """
    Multimodal chat with image and/or audio upload.
    محادثة متعددة الوسائط مع رفع صور و/أو صوت
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # FIX: Validate upload sizes
    max_size = cfg.server.max_upload_size if cfg else 52428800  # 50MB default

    image_bytes = None
    audio_bytes = None

    if image:
        image_bytes = await image.read()
        if len(image_bytes) > max_size:
            raise HTTPException(status_code=413, detail=f"Image too large ({len(image_bytes)} bytes, max {max_size})")
        logger.info(f"Image received: {image.filename} ({len(image_bytes)} bytes)")

    if audio:
        audio_bytes = await audio.read()
        if len(audio_bytes) > max_size:
            raise HTTPException(status_code=413, detail=f"Audio too large ({len(audio_bytes)} bytes, max {max_size})")
        logger.info(f"Audio received: {audio.filename} ({len(audio_bytes)} bytes)")

    req = InferenceRequest(
        text=message,
        image=image_bytes,
        audio=audio_bytes,
        use_rag=use_rag,
        system_prompt=system_prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )

    result = await asyncio.to_thread(engine.generate, req)

    return ChatResponse(
        response=result.text,
        audio_transcript=result.audio_transcript,
        rag_sources=result.rag_sources,
        latency_ms=result.latency_ms,
        tokens_generated=result.tokens_generated,
    )


# ─────────────────── RAG Management Endpoints ───────────────────

@app.post("/rag/index", dependencies=[Depends(verify_api_key)])
async def rag_index_text(request: IndexRequest):
    """
    Index a text document into RAG.
    فهرسة مستند نصي في RAG
    """
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    count = rag.index_text(request.text, request.metadata)
    return {"status": "indexed", "chunks": count}


@app.post("/rag/index/file", dependencies=[Depends(verify_api_key)])
async def rag_index_file(file: UploadFile = File(...)):
    """
    Upload and index a file into RAG.
    رفع وفهرسة ملف في RAG
    """
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    content = await file.read()
    text = content.decode("utf-8")
    metadata = {"filename": file.filename, "content_type": file.content_type}
    count = rag.index_text(text, metadata)

    return {"status": "indexed", "filename": file.filename, "chunks": count}


@app.post("/rag/search", dependencies=[Depends(verify_api_key)])
async def rag_search(request: SearchRequest):
    """
    Search RAG knowledge base.
    البحث في قاعدة المعرفة RAG
    """
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    docs = rag.retrieve(request.query, request.top_k)
    return {
        "query": request.query,
        "results": [
            {
                "content": doc.content[:500],
                "score": round(doc.score, 4),
                "metadata": doc.metadata,
            }
            for doc in docs
        ],
    }


@app.post("/rag/index/directory", dependencies=[Depends(verify_api_key)])
async def rag_index_directory(dir_path: str = Form(...)):
    """
    Index all files in a directory.
    FIX: Validate path to prevent Path Traversal (/../../../etc/passwd)
    Only allows indexing within ./data/ directory.
    """
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    from pathlib import Path
    import os

    # Resolve to absolute and verify it's within allowed base
    allowed_base = Path("./data").resolve()
    target = Path(dir_path).resolve()

    if not str(target).startswith(str(allowed_base)):
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: indexing only allowed within {allowed_base}",
        )

    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {dir_path}")

    count = await asyncio.to_thread(rag.index_directory, str(target))
    return {"status": "indexed", "directory": str(target), "total_chunks": count}


# ─────────────────── Audio Transcription ───────────────────

@app.post("/transcribe", dependencies=[Depends(verify_api_key)])
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio file to text.
    تحويل ملف صوتي إلى نص
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    audio_bytes = await audio.read()
    text = engine.transcriber.transcribe(audio_bytes=audio_bytes)

    return {"filename": audio.filename, "transcript": text}


# ─────────────────── Server Entry Point ───────────────────

def start_server():
    """Start the production server."""
    import uvicorn
    config = load_config("config/config.yaml")
    uvicorn.run(
        "src.api.server:app",
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
        log_level="info",
    )


if __name__ == "__main__":
    start_server()
