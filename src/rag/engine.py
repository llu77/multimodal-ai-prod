"""
RAG Engine — Retrieval-Augmented Generation with Vector Search.
محرك RAG — التوليد المعزز بالاسترجاع مع البحث المتجهي
Supports: Supabase (pgvector), ChromaDB (local)
"""
import json
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from loguru import logger

from src.utils.config import AppConfig, RAGConfig


@dataclass
class Document:
    """A document chunk with metadata."""
    content: str
    metadata: dict
    doc_id: str = ""
    score: float = 0.0

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


# ─────────────────────────── Text Chunker ───────────────────────────

import re


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences — supports Arabic and English.
    تقسيم النص إلى جمل — يدعم العربية والإنجليزية

    Handles: Arabic period/question/exclamation, English punctuation,
    newline-delimited paragraphs, and numbered lists.
    """
    # Normalize whitespace but preserve newlines as potential boundaries
    text = re.sub(r"[ \t]+", " ", text)

    # First: split on newlines (always a boundary)
    lines = text.split("\n")
    
    # Then: split each line on sentence-ending punctuation
    pattern = r'(?<=[.!?؟।])\s+'
    sentences = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        raw_splits = re.split(pattern, line)
        for s in raw_splits:
            s = s.strip()
            if not s:
                continue
            # If a "sentence" is still very long (no punctuation found),
            # fall back to splitting on commas / semicolons / Arabic comma
            if len(s) > 500:
                sub_parts = re.split(r'(?<=[,;،])\s+', s)
                sentences.extend(p.strip() for p in sub_parts if p.strip())
            else:
                sentences.append(s)

    return sentences


class TextChunker:
    """
    Token-aware, sentence-respecting text chunker.
    مُقسّم نصوص ذكي يحترم حدود الجمل ويحسب بالتوكنات

    Strategy:
    1. Split text into sentences (Arabic + English aware)
    2. Greedily group sentences until token budget is reached
    3. Apply overlap by re-including trailing sentences from previous chunk

    FIX over old version:
    - Old: split by whitespace words → cuts mid-sentence, wrong unit
    - New: split by sentences → groups by token count → clean boundaries
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        tokenizer=None,
    ):
        self.chunk_size = chunk_size  # in tokens
        self.overlap = overlap        # in tokens
        self._tokenizer = tokenizer

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tokenizer or fallback heuristic."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        # Heuristic: ~1.5 tokens per whitespace-word for Arabic,
        # ~1.3 for English. Use 1.5 as safe upper bound.
        return int(len(text.split()) * 1.5)

    def set_tokenizer(self, tokenizer) -> None:
        """Attach a real tokenizer for accurate counting."""
        self._tokenizer = tokenizer

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Document]:
        """
        Split text into overlapping chunks respecting sentence boundaries.
        تقسيم النص إلى أجزاء متداخلة مع احترام حدود الجمل
        """
        metadata = metadata or {}
        sentences = _split_sentences(text)

        if not sentences:
            return []

        chunks: list[Document] = []
        current_sentences: list[str] = []
        current_tokens: int = 0

        for sentence in sentences:
            sent_tokens = self._count_tokens(sentence)

            # If single sentence exceeds budget, force-split it
            if sent_tokens > self.chunk_size:
                # Flush current buffer first
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(Document(
                        content=chunk_text,
                        metadata={**metadata, "chunk_index": len(chunks)},
                    ))
                    current_sentences = []
                    current_tokens = 0

                # Force-split the long sentence by words
                words = sentence.split()
                buf = []
                buf_tokens = 0
                for w in words:
                    w_tok = self._count_tokens(w)
                    if buf_tokens + w_tok > self.chunk_size and buf:
                        chunks.append(Document(
                            content=" ".join(buf),
                            metadata={**metadata, "chunk_index": len(chunks)},
                        ))
                        # Overlap: keep last few words
                        overlap_words = max(1, int(self.overlap / 2))
                        buf = buf[-overlap_words:]
                        buf_tokens = self._count_tokens(" ".join(buf))
                    buf.append(w)
                    buf_tokens += w_tok
                if buf:
                    current_sentences = [" ".join(buf)]
                    current_tokens = buf_tokens
                continue

            # Normal case: would this sentence overflow the budget?
            if current_tokens + sent_tokens > self.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(Document(
                    content=chunk_text,
                    metadata={**metadata, "chunk_index": len(chunks)},
                ))

                # Overlap: carry trailing sentences that fit within overlap budget
                overlap_sents: list[str] = []
                overlap_tok = 0
                for s in reversed(current_sentences):
                    s_tok = self._count_tokens(s)
                    if overlap_tok + s_tok > self.overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_tok += s_tok

                current_sentences = overlap_sents
                current_tokens = overlap_tok

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        # Flush remaining
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(Document(
                content=chunk_text,
                metadata={**metadata, "chunk_index": len(chunks)},
            ))

        logger.debug(f"Chunked into {len(chunks)} parts (budget={self.chunk_size} tokens)")
        return chunks


# ─────────────────────────── Embedding Model ───────────────────────────

class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct"):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model_name = model_name

    def embed(self, texts: list[str], prefix: str = "passage: ") -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        prefixed = [f"{prefix}{t}" for t in texts]
        embeddings = self.model.encode(prefixed, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query."""
        return self.embed([query], prefix="query: ")[0]


# ─────────────────────────── Vector Store Interface ───────────────────────────

class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_documents(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Document]:
        pass

    @abstractmethod
    def delete_all(self) -> None:
        pass


# ─────────────────────────── ChromaDB Store ───────────────────────────

class ChromaDBStore(VectorStore):
    """Local ChromaDB vector store."""

    def __init__(self, persist_dir: str, collection_name: str):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB initialized: {persist_dir}/{collection_name}")

    def add_documents(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        self.collection.add(
            ids=[d.doc_id for d in documents],
            embeddings=embeddings,
            documents=[d.content for d in documents],
            metadatas=[d.metadata for d in documents],
        )
        logger.info(f"Added {len(documents)} documents to ChromaDB")

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Document]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = []
        for i in range(len(results["documents"][0])):
            docs.append(Document(
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                doc_id=results["ids"][0][i],
                score=1.0 - results["distances"][0][i],  # cosine distance to similarity
            ))
        return docs

    def delete_all(self) -> None:
        self.client.delete_collection(self.collection.name)
        logger.info("ChromaDB collection deleted")


# ─────────────────────────── Supabase (pgvector) Store ───────────────────────────

class SupabaseStore(VectorStore):
    """Supabase pgvector store for production."""

    def __init__(self, url: str, key: str, table_name: str):
        from supabase import create_client
        self.client = create_client(url, key)
        self.table_name = table_name
        logger.info(f"Supabase connected: {table_name}")

    def add_documents(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        rows = []
        for doc, emb in zip(documents, embeddings):
            rows.append({
                "id": doc.doc_id,
                "content": doc.content,
                # FIX: Pass dict directly — JSONB column handles serialization.
                # Old code: json.dumps() → stored as JSON string inside JSONB = double-encoding.
                "metadata": doc.metadata,
                "embedding": emb,
            })
        # Batch insert with error handling
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            try:
                self.client.table(self.table_name).upsert(batch).execute()
            except Exception as e:
                logger.error(f"Supabase upsert failed (batch {i // batch_size}): {e}")
                raise
        logger.info(f"Added {len(documents)} documents to Supabase")

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Document]:
        # Uses pgvector cosine similarity via Supabase RPC
        result = self.client.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.5,
                "match_count": top_k,
            }
        ).execute()

        docs = []
        for row in result.data:
            # FIX: metadata is already a dict from JSONB — no json.loads needed.
            # Handle both cases for backward compatibility with old double-encoded data.
            raw_meta = row.get("metadata", {})
            if isinstance(raw_meta, str):
                try:
                    raw_meta = json.loads(raw_meta)
                except (json.JSONDecodeError, TypeError):
                    raw_meta = {}

            docs.append(Document(
                content=row["content"],
                metadata=raw_meta,
                doc_id=row["id"],
                score=row.get("similarity", 0.0),
            ))
        return docs

    def delete_all(self) -> None:
        self.client.table(self.table_name).delete().neq("id", "").execute()
        logger.info("Supabase table cleared")


# ─────────────────────────── BM25 Keyword Search (TODO-6) ───────────────────────────

class BM25Index:
    """
    Simple BM25 keyword search index — no external dependencies.
    فهرس بحث كلمات BM25 بسيط — بدون مكتبات خارجية

    Used for hybrid search: vector similarity + keyword matching.
    Important for technical terms, numbers, and exact phrases
    that vector search can miss.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[Document] = []
        self.doc_tokens: list[list[str]] = []
        self.avg_dl: float = 0.0
        self.df: dict[str, int] = {}  # document frequency
        self.N: int = 0

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer for BM25."""
        import re
        # Remove diacritics
        text = re.sub(r'[\u064B-\u065F\u0610-\u061A\u0670]', '', text)
        return re.findall(r'\w+', text.lower())

    def add_documents(self, documents: list[Document]) -> None:
        """Index documents for BM25 search."""
        for doc in documents:
            tokens = self._tokenize(doc.content)
            self.docs.append(doc)
            self.doc_tokens.append(tokens)

            # Update document frequency
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

        self.N = len(self.docs)
        total_tokens = sum(len(t) for t in self.doc_tokens)
        self.avg_dl = total_tokens / max(self.N, 1)

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        """Search using BM25 scoring."""
        import math

        if not self.docs:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for i, doc_tokens in enumerate(self.doc_tokens):
            score = 0.0
            dl = len(doc_tokens)
            tf_map = {}
            for t in doc_tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for term in query_tokens:
                if term not in tf_map:
                    continue
                tf = tf_map[term]
                df = self.df.get(term, 0)
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
                score += idf * tf_norm

            scores.append((score, i))

        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            if score > 0:
                doc = self.docs[idx]
                doc_copy = Document(
                    content=doc.content,
                    metadata=doc.metadata,
                    doc_id=doc.doc_id,
                    score=score,
                )
                results.append(doc_copy)

        return results


# ─────────────────────────── Query Cache (TODO-5) ───────────────────────────

class QueryCache:
    """
    Simple TTL cache for RAG queries — avoids re-retrieving identical queries.
    ذاكرة مؤقتة بسيطة لاستعلامات RAG — تتجنب إعادة الاسترجاع

    Uses LRU eviction with time-based expiry.
    """

    def __init__(self, max_size: int = 256, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: dict[str, tuple[float, list[Document]]] = {}  # key → (timestamp, docs)

    def _make_key(self, query: str, top_k: int) -> str:
        """Create cache key from query + top_k."""
        return f"{query.strip().lower()}::{top_k}"

    def get(self, query: str, top_k: int) -> Optional[list[Document]]:
        """Get cached results if fresh enough."""
        import time
        key = self._make_key(query, top_k)
        if key in self._cache:
            ts, docs = self._cache[key]
            if time.time() - ts < self.ttl:
                return docs
            else:
                del self._cache[key]  # Expired
        return None

    def put(self, query: str, top_k: int, docs: list[Document]) -> None:
        """Cache retrieval results."""
        import time
        # Evict oldest if full
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]

        key = self._make_key(query, top_k)
        self._cache[key] = (time.time(), docs)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


# ─────────────────────────── RAG Engine ───────────────────────────

class RAGEngine:
    """
    Production RAG Engine with hybrid search and caching.
    محرك RAG الإنتاجي مع بحث هجين وذاكرة مؤقتة

    Features:
    - Vector search (embedding similarity)
    - BM25 keyword search (TODO-6) for technical terms / exact phrases
    - Reciprocal Rank Fusion to merge vector + keyword results
    - Query cache (TODO-5) to avoid re-retrieving identical queries
    - Reranking for precision improvement
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg.rag
        self.chunker = TextChunker(
            chunk_size=self.cfg.chunk_size,
            overlap=self.cfg.chunk_overlap,
        )
        self.embedder = EmbeddingModel(self.cfg.embedding_model)
        self.store = self._init_store()

        # TODO-6: BM25 keyword index
        self.bm25 = BM25Index()

        # TODO-5: Query cache
        self.cache = QueryCache(max_size=256, ttl_seconds=300)

        self.reranker = None
        if self.cfg.rerank:
            self._init_reranker()

    def _init_store(self) -> VectorStore:
        """Initialize the appropriate vector store."""
        if self.cfg.vector_store == "supabase" and self.cfg.supabase_url:
            return SupabaseStore(
                url=self.cfg.supabase_url,
                key=self.cfg.supabase_key,
                table_name=self.cfg.supabase_table,
            )
        else:
            return ChromaDBStore(
                persist_dir=self.cfg.chromadb_persist_dir,
                collection_name=self.cfg.chromadb_collection,
            )

    def _init_reranker(self):
        """Initialize reranker model for improved retrieval quality."""
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(self.cfg.rerank_model)
            logger.info(f"Reranker loaded: {self.cfg.rerank_model}")
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}")

    def index_text(self, text: str, metadata: Optional[dict] = None) -> int:
        """
        Index a text document (chunk + embed + store + BM25).
        TODO-6: Also adds to BM25 index for hybrid search.
        TODO-5: Clears cache when new data is indexed.
        """
        chunks = self.chunker.chunk(text, metadata)
        if not chunks:
            return 0

        embeddings = self.embedder.embed([c.content for c in chunks])
        self.store.add_documents(chunks, embeddings)

        # Also index in BM25 for keyword search
        self.bm25.add_documents(chunks)

        # Invalidate cache since knowledge base changed
        self.cache.clear()

        return len(chunks)

    def index_file(self, file_path: str) -> int:
        """
        Index any supported file (txt, md, pdf, docx, html, csv, json).
        فهرسة أي ملف مدعوم
        TODO-7 FIX: Was text-only, now uses document_parser for PDF/DOCX/HTML.
        """
        from src.rag.document_parser import extract_text

        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return 0

        text = extract_text(str(path))
        if not text:
            logger.warning(f"No text extracted from: {path.name}")
            return 0

        metadata = {"source": str(path), "filename": path.name, "filetype": path.suffix}
        return self.index_text(text, metadata)

    def index_directory(self, dir_path: str, extensions: list[str] = None) -> int:
        """
        Index all supported files in a directory.
        TODO-7 FIX: Now supports PDF, DOCX, HTML in addition to text files.
        """
        from src.rag.document_parser import supported_extensions
        extensions = extensions or supported_extensions()
        path = Path(dir_path)
        total = 0

        for ext in extensions:
            for file_path in sorted(path.rglob(f"*{ext}")):
                count = self.index_file(str(file_path))
                total += count
                if count > 0:
                    logger.info(f"Indexed {file_path.name}: {count} chunks")

        logger.info(f"Total indexed: {total} chunks from {dir_path}")
        return total

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[Document]:
        """
        Retrieve relevant documents using hybrid search with caching.
        استرجاع المستندات ذات الصلة باستخدام بحث هجين مع ذاكرة مؤقتة

        Pipeline:
        1. Check cache (TODO-5) → return immediately if fresh hit
        2. Vector search (embedding similarity)
        3. BM25 keyword search (TODO-6) for exact terms
        4. Reciprocal Rank Fusion to merge both result sets
        5. Filter by similarity threshold
        6. Rerank if available
        7. Cache results before returning
        """
        top_k = top_k or self.cfg.top_k

        # Step 1: Cache check
        cached = self.cache.get(query, top_k)
        if cached is not None:
            logger.debug(f"Cache hit for query (cache size={self.cache.size})")
            return cached

        # Step 2: Vector search
        query_embedding = self.embedder.embed_query(query)
        fetch_k = top_k * 3 if self.reranker else top_k * 2
        vector_results = self.store.search(query_embedding, top_k=fetch_k)

        # Step 3: BM25 keyword search
        bm25_results = self.bm25.search(query, top_k=fetch_k)

        # Step 4: Reciprocal Rank Fusion
        if bm25_results:
            candidates = self._reciprocal_rank_fusion(vector_results, bm25_results, k=60)
        else:
            candidates = vector_results

        # Step 5: Filter by threshold
        candidates = [d for d in candidates if d.score >= self.cfg.similarity_threshold]

        # Step 6: Rerank if available
        if self.reranker and candidates:
            pairs = [[query, doc.content] for doc in candidates]
            scores = self.reranker.predict(pairs)
            for doc, score in zip(candidates, scores):
                doc.score = float(score)
            candidates.sort(key=lambda d: d.score, reverse=True)

        final = candidates[:top_k]

        # Step 7: Cache results
        self.cache.put(query, top_k, final)

        return final

    @staticmethod
    def _reciprocal_rank_fusion(
        vector_results: list[Document],
        keyword_results: list[Document],
        k: int = 60,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ) -> list[Document]:
        """
        Merge vector and keyword results using Reciprocal Rank Fusion.
        دمج نتائج البحث المتجهي والكلمات باستخدام RRF

        RRF score = sum(weight / (k + rank)) across result lists.
        k=60 is the standard constant from the original RRF paper.
        """
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(vector_results):
            rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0) + vector_weight / (k + rank + 1)
            doc_map[doc.doc_id] = doc

        for rank, doc in enumerate(keyword_results):
            rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0) + keyword_weight / (k + rank + 1)
            if doc.doc_id not in doc_map:
                doc_map[doc.doc_id] = doc

        # Sort by RRF score and assign as doc.score
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        results = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id]
            doc.score = rrf_scores[doc_id]
            results.append(doc)

        return results

    def build_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Build RAG context string from retrieved documents.
        بناء سياق RAG من المستندات المسترجعة

        Returns formatted context ready to be injected into the prompt.
        """
        docs = self.retrieve(query, top_k)
        if not docs:
            return ""

        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", doc.metadata.get("filename", "unknown"))
            context_parts.append(
                f"[مرجع {i}] (مصدر: {source}, درجة الصلة: {doc.score:.2f})\n{doc.content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def augmented_prompt(
        self,
        query: str,
        system_prompt: str = "",
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Build a complete RAG-augmented prompt.
        بناء prompt معزز بالسياق المسترجع

        Returns messages list ready for model input.
        """
        context = self.build_context(query, top_k)

        system = system_prompt or (
            "أنت مساعد ذكي متعدد الوسائط. استخدم المراجع المقدمة للإجابة بدقة. "
            "إذا لم تجد الإجابة في المراجع، أخبر المستخدم بذلك."
        )

        if context:
            user_content = (
                f"السياق والمراجع:\n{context}\n\n"
                f"---\n\n"
                f"السؤال: {query}\n\n"
                f"أجب بناءً على المراجع المقدمة أعلاه."
            )
        else:
            user_content = query

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]


# ─────────────────────────── SQL for Supabase Setup ───────────────────────────

SUPABASE_SETUP_SQL = """
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1024),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Similarity search function
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1024),
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id TEXT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.content,
        d.metadata,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents d
    WHERE 1 - (d.embedding <=> query_embedding) > match_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Index for fast similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_idx
ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG Indexer")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--index-dir", type=str, help="Directory to index")
    parser.add_argument("--query", type=str, help="Test query")
    parser.add_argument("--print-sql", action="store_true", help="Print Supabase setup SQL")
    args = parser.parse_args()

    if args.print_sql:
        print(SUPABASE_SETUP_SQL)
    else:
        from src.utils.config import load_config
        cfg = load_config(args.config)
        engine = RAGEngine(cfg)

        if args.index_dir:
            engine.index_directory(args.index_dir)
        if args.query:
            docs = engine.retrieve(args.query)
            for doc in docs:
                print(f"[{doc.score:.3f}] {doc.content[:200]}...")
