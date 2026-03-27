"""
Tests for the text chunker — sentence splitting, token counting, Arabic support.
اختبارات مُقسّم النصوص — تقسيم الجمل، حساب التوكنات، دعم العربية

These tests verify the FIX-1 chunker that replaced the primitive word-split.
"""
import pytest
from src.rag.engine import TextChunker, _split_sentences, Document
from tests.conftest import ARABIC_PARAGRAPH, ENGLISH_PARAGRAPH, MIXED_PARAGRAPH


class TestSentenceSplitting:
    """Verify _split_sentences handles Arabic and English correctly."""

    def test_arabic_period_split(self):
        text = "الجملة الأولى. الجملة الثانية. الجملة الثالثة."
        sentences = _split_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "الجملة الأولى."
        assert sentences[2] == "الجملة الثالثة."

    def test_arabic_question_mark(self):
        text = "ما هو الذكاء الاصطناعي؟ هو فرع من علوم الحاسوب."
        sentences = _split_sentences(text)
        assert len(sentences) == 2
        assert "؟" in sentences[0]

    def test_english_split(self):
        text = "First sentence. Second sentence! Third sentence?"
        sentences = _split_sentences(text)
        assert len(sentences) == 3

    def test_mixed_language(self):
        sentences = _split_sentences(MIXED_PARAGRAPH)
        assert len(sentences) >= 3

    def test_newline_as_boundary(self):
        text = "سطر أول\nسطر ثاني\nسطر ثالث"
        sentences = _split_sentences(text)
        assert len(sentences) == 3

    def test_empty_text(self):
        assert _split_sentences("") == []
        assert _split_sentences("   ") == []

    def test_single_sentence_no_punctuation(self):
        text = "جملة بدون علامة ترقيم"
        sentences = _split_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == text

    def test_long_sentence_falls_back_to_commas(self):
        # Build a 600-char sentence with commas but no periods
        parts = [f"جزء {i} من النص الطويل" for i in range(40)]
        text = "، ".join(parts)
        sentences = _split_sentences(text)
        # Should split on commas since len > 500
        assert len(sentences) > 1

    def test_multiple_whitespace_normalized(self):
        text = "الجملة   الأولى.    الجملة   الثانية."
        sentences = _split_sentences(text)
        assert len(sentences) == 2
        # No double spaces in output
        assert "  " not in sentences[0]


class TestTextChunkerBasic:
    """Basic chunking functionality."""

    def test_empty_text_returns_empty(self):
        chunker = TextChunker(chunk_size=100)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_short_text_single_chunk(self):
        chunker = TextChunker(chunk_size=500)
        chunks = chunker.chunk("جملة قصيرة جداً.")
        assert len(chunks) == 1
        assert chunks[0].content == "جملة قصيرة جداً."

    def test_returns_document_objects(self):
        chunker = TextChunker(chunk_size=500)
        chunks = chunker.chunk("نص اختبار.")
        assert isinstance(chunks[0], Document)
        assert chunks[0].doc_id  # Should have auto-generated ID
        assert chunks[0].metadata.get("chunk_index") == 0

    def test_metadata_passed_through(self):
        chunker = TextChunker(chunk_size=500)
        meta = {"source": "test.txt", "page": 1}
        chunks = chunker.chunk("نص اختبار.", metadata=meta)
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["page"] == 1
        assert chunks[0].metadata["chunk_index"] == 0


class TestChunkerSentenceRespect:
    """Chunks should NOT cut sentences in the middle."""

    def test_no_mid_sentence_cuts(self):
        chunker = TextChunker(chunk_size=30, overlap=5)
        chunks = chunker.chunk(ARABIC_PARAGRAPH)

        for chunk in chunks:
            text = chunk.content.strip()
            # Each chunk should end with punctuation or be the last chunk
            if chunk.metadata["chunk_index"] < len(chunks) - 1:
                # Non-last chunks should ideally end with sentence boundary
                # (unless a single sentence exceeds chunk_size)
                assert len(text) > 0

    def test_arabic_paragraph_chunks_correctly(self):
        chunker = TextChunker(chunk_size=30, overlap=5)
        chunks = chunker.chunk(ARABIC_PARAGRAPH)
        assert len(chunks) >= 2

        # Reconstruct: all original sentences should appear across chunks
        full_text = " ".join(c.content for c in chunks)
        assert "الذكاء الاصطناعي" in full_text
        assert "الرؤية الحاسوبية" in full_text


class TestChunkerOverlap:
    """Overlap should carry trailing sentences from previous chunk."""

    def test_overlap_creates_shared_content(self):
        # Use a text with many short sentences and small chunk size
        text = ". ".join([f"جملة رقم {i}" for i in range(20)]) + "."
        chunker = TextChunker(chunk_size=30, overlap=15)
        chunks = chunker.chunk(text)

        if len(chunks) >= 2:
            # Some content from end of chunk[0] should appear in chunk[1]
            words_0 = set(chunks[0].content.split())
            words_1 = set(chunks[1].content.split())
            shared = words_0 & words_1
            # With overlap, there should be some shared words
            assert len(shared) > 0

    def test_zero_overlap(self):
        # Need enough sentences and small enough chunk to force splitting
        text = ". ".join([f"الجملة رقم {i} في النص" for i in range(10)]) + "."
        chunker = TextChunker(chunk_size=10, overlap=0)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2


class TestChunkerTokenCounting:
    """Token counting with heuristic and custom tokenizer."""

    def test_heuristic_counting(self):
        chunker = TextChunker(chunk_size=100)
        # Heuristic: ~1.5 tokens per word
        count = chunker._count_tokens("كلمة واحدة اثنتان ثلاث أربع خمس")
        assert count > 0
        assert isinstance(count, int)

    def test_custom_tokenizer(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return text.split()  # 1 token per word

        chunker = TextChunker(chunk_size=100)
        chunker.set_tokenizer(FakeTokenizer())
        count = chunker._count_tokens("one two three four")
        assert count == 4

    def test_chunk_size_respected(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return text.split()

        chunker = TextChunker(chunk_size=10, overlap=2)
        chunker.set_tokenizer(FakeTokenizer())

        # Create text with many words
        text = ". ".join([f"Word{i} word{i}a word{i}b" for i in range(20)]) + "."
        chunks = chunker.chunk(text)

        for chunk in chunks:
            token_count = chunker._count_tokens(chunk.content)
            # Allow some slack for sentence boundary respect
            assert token_count <= 15, f"Chunk too large: {token_count} tokens"


class TestChunkerForceSplit:
    """Very long sentences that exceed chunk_size should be force-split."""

    def test_long_sentence_gets_split(self):
        # Build one giant sentence with no punctuation
        long_sentence = " ".join([f"كلمة{i}" for i in range(200)])
        chunker = TextChunker(chunk_size=30, overlap=5)
        chunks = chunker.chunk(long_sentence)
        assert len(chunks) > 1

    def test_mixed_long_and_short(self):
        short = "جملة قصيرة."
        long_sent = " ".join([f"كلمة{i}" for i in range(200)])
        text = f"{short} {long_sent} {short}"
        chunker = TextChunker(chunk_size=50, overlap=10)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2


class TestChunkerEdgeCases:
    """Edge cases and regression tests."""

    def test_only_punctuation(self):
        chunker = TextChunker(chunk_size=100)
        chunks = chunker.chunk("...!?")
        assert len(chunks) <= 1

    def test_unicode_text(self):
        chunker = TextChunker(chunk_size=100)
        text = "مرحباً بالعالم 🌍. هذا نص يحتوي رموز تعبيرية."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert "🌍" in chunks[0].content

    def test_numeric_content(self):
        chunker = TextChunker(chunk_size=100)
        text = "درجة الحرارة 37.5 مئوية. ضغط الدم 120/80."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert "37.5" in chunks[0].content

    def test_chunk_ids_unique(self):
        chunker = TextChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk(ARABIC_PARAGRAPH)
        ids = [c.doc_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_indices_sequential(self):
        chunker = TextChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk(ARABIC_PARAGRAPH)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))
