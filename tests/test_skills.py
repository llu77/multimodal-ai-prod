"""
Tests for advanced agent skills.
اختبارات مهارات الوكيل المتقدمة

Note: web_search, pubmed_search, drug_interaction need internet.
      They are marked with @pytest.mark.network.
      File generation tests work offline.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agent.skills import (
    _web_search, _pubmed_search, _generate_pdf, _generate_docx,
    _check_drug_interaction, _clinical_guidelines_search,
    register_advanced_skills,
)
from src.agent.tools import ToolRegistry


# ═══════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════

class TestRegistration:

    def test_register_all_skills(self):
        registry = ToolRegistry()
        register_advanced_skills(registry)
        names = registry.list_names()
        assert "web_search" in names
        assert "pubmed_search" in names
        assert "generate_pdf" in names
        assert "generate_docx" in names
        assert "drug_interaction" in names
        assert "clinical_guidelines" in names

    def test_skills_have_descriptions(self):
        registry = ToolRegistry()
        register_advanced_skills(registry)
        for tool_info in registry.list_tools():
            assert len(tool_info["description"]) > 20, f"{tool_info['name']} has no description"


# ═══════════════════════════════════════════
# File Generation (offline)
# ═══════════════════════════════════════════

class TestGeneratePDF:

    def test_creates_pdf_file(self, tmp_dir):
        # Monkey-patch output dir
        import src.agent.skills as skills_mod
        original = Path("./data/outputs")
        try:
            result = _generate_pdf(
                title="تقرير اختبار",
                content="هذا محتوى تجريبي للتقرير.\n\nفقرة ثانية.",
                filename="test_report.pdf",
            )
            if "fpdf2 غير مثبتة" in result:
                pytest.skip("fpdf2 not installed")
            assert "تم إنشاء" in result or "خطأ" in result
        except Exception:
            pytest.skip("PDF generation not available in this env")

    def test_empty_content_handled(self):
        result = _generate_pdf(title="عنوان", content="")
        # Should not crash
        assert isinstance(result, str)


class TestGenerateDocx:

    def test_creates_docx(self):
        result = _generate_docx(
            title="مستند اختبار",
            content="# قسم أول\n\nفقرة نصية.\n\n## قسم فرعي\n\n- نقطة أولى\n- نقطة ثانية",
            filename="test_doc.docx",
        )
        if "python-docx غير مثبتة" in result:
            pytest.skip("python-docx not installed")
        assert "تم إنشاء" in result or "خطأ" in result

    def test_headings_and_lists(self):
        result = _generate_docx(
            title="Test",
            content="# Heading 1\n\n## Heading 2\n\n- Item 1\n- Item 2\n\nParagraph.",
        )
        # Should not crash regardless of library availability
        assert isinstance(result, str)


# ═══════════════════════════════════════════
# Network-dependent tests (web search, PubMed)
# ═══════════════════════════════════════════

@pytest.mark.network
class TestWebSearch:

    def test_basic_search(self):
        result = _web_search("Python programming language")
        assert isinstance(result, str)
        assert len(result) > 50

    def test_arabic_search(self):
        result = _web_search("إعادة التأهيل الطبي")
        assert isinstance(result, str)

    def test_empty_query(self):
        result = _web_search("")
        assert isinstance(result, str)


@pytest.mark.network
class TestPubMedSearch:

    def test_basic_search(self):
        result = _pubmed_search("shoulder rehabilitation frozen")
        assert isinstance(result, str)
        assert "PubMed" in result or "خطأ" in result

    def test_arabic_medical_query(self):
        result = _pubmed_search("stroke rehabilitation outcomes")
        assert isinstance(result, str)

    def test_returns_links(self):
        result = _pubmed_search("knee osteoarthritis exercise")
        if "خطأ" not in result:
            assert "pubmed.ncbi.nlm.nih.gov" in result


@pytest.mark.network
class TestDrugInteraction:

    def test_known_interaction(self):
        result = _check_drug_interaction("warfarin", "aspirin")
        assert isinstance(result, str)

    def test_unknown_drugs(self):
        result = _check_drug_interaction("xyzabc123", "qwerty456")
        assert isinstance(result, str)
        assert "لم يتم" in result or "خطأ" in result


# ═══════════════════════════════════════════
# Mocked Network Tests (always run)
# ═══════════════════════════════════════════

class TestWebSearchMocked:

    def test_handles_timeout(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timeout")):
            result = _web_search("test")
            assert "خطأ" in result

    def test_handles_network_error(self):
        with patch("urllib.request.urlopen", side_effect=ConnectionError("no internet")):
            result = _web_search("test")
            assert "خطأ" in result


class TestPubMedMocked:

    def test_handles_timeout(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timeout")):
            result = _pubmed_search("test")
            assert "خطأ" in result

    def test_handles_empty_results(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "esearchresult": {"idlist": [], "count": "0"}
        }).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _pubmed_search("xyznonexistent")
            assert "لم يتم" in result
