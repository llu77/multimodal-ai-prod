"""
Tests for new RAG components: BM25, QueryCache, DocumentParser.
اختبارات مكونات RAG الجديدة
"""
import time
import pytest
from src.rag.engine import BM25Index, QueryCache, Document, _split_sentences


# ═══════════════════════════════════════════
# BM25 Keyword Search
# ═══════════════════════════════════════════

class TestBM25Index:

    @pytest.fixture
    def bm25_with_docs(self):
        bm25 = BM25Index()
        docs = [
            Document(content="القلب عضو عضلي يضخ الدم في جسم الإنسان", metadata={}, doc_id="d1"),
            Document(content="الكبد أكبر عضو داخلي وينظف السموم من الدم", metadata={}, doc_id="d2"),
            Document(content="الرئتان مسؤولتان عن تبادل الأكسجين وثاني أكسيد الكربون", metadata={}, doc_id="d3"),
            Document(content="الكلى تصفي الدم وتنتج البول للتخلص من الفضلات", metadata={}, doc_id="d4"),
            Document(content="Python programming language is used for machine learning", metadata={}, doc_id="d5"),
        ]
        bm25.add_documents(docs)
        return bm25

    def test_empty_index_returns_empty(self):
        bm25 = BM25Index()
        assert bm25.search("أي شيء") == []

    def test_finds_relevant_doc(self, bm25_with_docs):
        results = bm25_with_docs.search("القلب يضخ الدم", top_k=3)
        assert len(results) > 0
        assert results[0].doc_id == "d1"

    def test_english_search(self, bm25_with_docs):
        results = bm25_with_docs.search("Python machine learning", top_k=3)
        assert len(results) > 0
        assert results[0].doc_id == "d5"

    def test_no_match_returns_empty(self, bm25_with_docs):
        results = bm25_with_docs.search("فيزياء كمية فلكية", top_k=3)
        # May return empty or low-score results
        high_score = [r for r in results if r.score > 0.5]
        assert len(high_score) == 0

    def test_respects_top_k(self, bm25_with_docs):
        results = bm25_with_docs.search("الدم", top_k=2)
        assert len(results) <= 2

    def test_scores_are_positive(self, bm25_with_docs):
        results = bm25_with_docs.search("القلب", top_k=5)
        for r in results:
            assert r.score > 0

    def test_results_sorted_by_score(self, bm25_with_docs):
        results = bm25_with_docs.search("الدم", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_incremental_add(self):
        bm25 = BM25Index()
        bm25.add_documents([Document(content="أول مستند", metadata={}, doc_id="a1")])
        bm25.add_documents([Document(content="ثاني مستند", metadata={}, doc_id="a2")])
        assert bm25.N == 2
        results = bm25.search("أول", top_k=5)
        assert len(results) >= 1


# ═══════════════════════════════════════════
# Query Cache
# ═══════════════════════════════════════════

class TestQueryCache:

    def test_cache_miss_returns_none(self):
        cache = QueryCache()
        assert cache.get("query", 5) is None

    def test_cache_hit_returns_docs(self):
        cache = QueryCache()
        docs = [Document(content="test", metadata={}, doc_id="x")]
        cache.put("my query", 5, docs)
        result = cache.get("my query", 5)
        assert result is not None
        assert len(result) == 1
        assert result[0].doc_id == "x"

    def test_different_top_k_is_different_key(self):
        cache = QueryCache()
        docs3 = [Document(content="a", metadata={}, doc_id="a")]
        docs5 = [Document(content="a", metadata={}, doc_id="a"), Document(content="b", metadata={}, doc_id="b")]
        cache.put("query", 3, docs3)
        cache.put("query", 5, docs5)
        assert len(cache.get("query", 3)) == 1
        assert len(cache.get("query", 5)) == 2

    def test_case_insensitive_key(self):
        cache = QueryCache()
        docs = [Document(content="t", metadata={}, doc_id="t")]
        cache.put("Hello World", 5, docs)
        assert cache.get("hello world", 5) is not None

    def test_ttl_expiry(self):
        cache = QueryCache(ttl_seconds=0.1)  # 100ms TTL
        docs = [Document(content="t", metadata={}, doc_id="t")]
        cache.put("q", 5, docs)
        assert cache.get("q", 5) is not None
        time.sleep(0.15)
        assert cache.get("q", 5) is None  # Expired

    def test_max_size_eviction(self):
        cache = QueryCache(max_size=2)
        cache.put("q1", 5, [])
        cache.put("q2", 5, [])
        cache.put("q3", 5, [])  # Should evict q1
        assert cache.size == 2
        assert cache.get("q1", 5) is None  # Evicted

    def test_clear(self):
        cache = QueryCache()
        cache.put("q1", 5, [])
        cache.put("q2", 5, [])
        cache.clear()
        assert cache.size == 0


# ═══════════════════════════════════════════
# Document Parser
# ═══════════════════════════════════════════

class TestDocumentParser:

    def test_extract_txt(self, tmp_dir):
        from src.rag.document_parser import extract_text
        path = tmp_dir / "test.txt"
        path.write_text("محتوى نصي بسيط", encoding="utf-8")
        text = extract_text(str(path))
        assert text == "محتوى نصي بسيط"

    def test_extract_md(self, tmp_dir):
        from src.rag.document_parser import extract_text
        path = tmp_dir / "test.md"
        path.write_text("# عنوان\nفقرة نصية.", encoding="utf-8")
        text = extract_text(str(path))
        assert "عنوان" in text

    def test_extract_html(self, tmp_dir):
        from src.rag.document_parser import extract_text
        path = tmp_dir / "test.html"
        path.write_text("<html><body><p>محتوى HTML</p><script>var x=1;</script></body></html>", encoding="utf-8")
        text = extract_text(str(path))
        assert "محتوى HTML" in text
        assert "var x" not in text  # Script stripped

    def test_unsupported_extension(self, tmp_dir):
        from src.rag.document_parser import extract_text
        path = tmp_dir / "file.xyz"
        path.write_text("data")
        assert extract_text(str(path)) is None

    def test_nonexistent_file(self):
        from src.rag.document_parser import extract_text
        assert extract_text("/nonexistent/file.txt") is None

    def test_supported_extensions(self):
        from src.rag.document_parser import supported_extensions
        exts = supported_extensions()
        assert ".pdf" in exts
        assert ".docx" in exts
        assert ".html" in exts
        assert ".txt" in exts
        assert ".csv" in exts


# ═══════════════════════════════════════════
# RRF (Reciprocal Rank Fusion)
# ═══════════════════════════════════════════

class TestReciprocalRankFusion:

    def test_basic_fusion(self):
        from src.rag.engine import RAGEngine
        vector = [
            Document(content="a", metadata={}, doc_id="d1", score=0.9),
            Document(content="b", metadata={}, doc_id="d2", score=0.8),
        ]
        keyword = [
            Document(content="b", metadata={}, doc_id="d2", score=5.0),
            Document(content="c", metadata={}, doc_id="d3", score=3.0),
        ]
        merged = RAGEngine._reciprocal_rank_fusion(vector, keyword)
        ids = [d.doc_id for d in merged]
        # d2 should rank high (appears in both)
        assert "d2" in ids[:2]
        # All three docs present
        assert set(ids) == {"d1", "d2", "d3"}

    def test_scores_are_positive(self):
        from src.rag.engine import RAGEngine
        vector = [Document(content="x", metadata={}, doc_id="d1", score=0.5)]
        keyword = [Document(content="y", metadata={}, doc_id="d2", score=1.0)]
        merged = RAGEngine._reciprocal_rank_fusion(vector, keyword)
        for doc in merged:
            assert doc.score > 0

    def test_empty_keyword_results(self):
        from src.rag.engine import RAGEngine
        vector = [Document(content="x", metadata={}, doc_id="d1", score=0.5)]
        merged = RAGEngine._reciprocal_rank_fusion(vector, [])
        assert len(merged) == 1

    def test_empty_vector_results(self):
        from src.rag.engine import RAGEngine
        keyword = [Document(content="x", metadata={}, doc_id="d1", score=1.0)]
        merged = RAGEngine._reciprocal_rank_fusion([], keyword)
        assert len(merged) == 1

    def test_duplicate_doc_boosted(self):
        from src.rag.engine import RAGEngine
        # Same doc in both lists should score higher
        both = Document(content="common", metadata={}, doc_id="shared", score=0.5)
        only_vec = Document(content="vec only", metadata={}, doc_id="vec", score=0.9)
        only_kw = Document(content="kw only", metadata={}, doc_id="kw", score=5.0)

        merged = RAGEngine._reciprocal_rank_fusion(
            [both, only_vec],
            [both, only_kw],
        )
        # shared should be first (boosted by appearing in both lists)
        assert merged[0].doc_id == "shared"
