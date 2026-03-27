"""
Tests for extended agent tools — web search, PubMed, image gen, report gen.
اختبارات الأدوات المتقدمة
"""
import pytest
from pathlib import Path
from src.agent.tools_extended import (
    web_search, pubmed_search, generate_rehab_image, generate_report,
)


class TestGenerateRehabImage:

    def test_shoulder_pendulum(self, tmp_dir):
        result = generate_rehab_image("shoulder_pendulum", output_dir=str(tmp_dir))
        assert "تم توليد" in result
        assert "SVG" in result
        svg_files = list(tmp_dir.glob("*.svg"))
        assert len(svg_files) == 1
        content = svg_files[0].read_text()
        assert "<svg" in content
        assert "البندول" in content

    def test_knee_extension(self, tmp_dir):
        result = generate_rehab_image("knee_extension", title="بسط الركبة", output_dir=str(tmp_dir))
        assert "تم توليد" in result
        svg_files = list(tmp_dir.glob("*.svg"))
        assert len(svg_files) == 1

    def test_fallback_to_general(self, tmp_dir):
        result = generate_rehab_image("unknown_exercise", output_dir=str(tmp_dir))
        assert "تم توليد" in result  # Should fallback to general_stretch

    def test_body_part_match(self, tmp_dir):
        result = generate_rehab_image("تمرين", body_part="back", output_dir=str(tmp_dir))
        assert "تم توليد" in result


class TestGenerateReport:

    def test_txt_report(self, tmp_dir):
        result = generate_report(
            title="تقرير متابعة",
            content="المريض يتحسن بشكل ملحوظ.\nنطاق الحركة تحسن 20 درجة.",
            output_format="txt",
            patient_name="أحمد محمد",
            output_dir=str(tmp_dir),
        )
        assert "تم توليد" in result
        assert "TXT" in result
        txt_files = list(tmp_dir.glob("*.txt"))
        assert len(txt_files) == 1
        content = txt_files[0].read_text(encoding="utf-8")
        assert "أحمد" in content
        assert "يتحسن" in content
        assert "Symbol Rehab AI" in content

    def test_docx_fallback_to_txt(self, tmp_dir):
        # python-docx may not be installed — should fallback
        result = generate_report(
            title="تقرير",
            content="محتوى",
            output_format="docx",
            output_dir=str(tmp_dir),
        )
        assert "تم توليد" in result

    def test_pdf_fallback_to_txt(self, tmp_dir):
        result = generate_report(
            title="تقرير",
            content="محتوى",
            output_format="pdf",
            output_dir=str(tmp_dir),
        )
        assert "تم توليد" in result

    def test_empty_content(self, tmp_dir):
        result = generate_report(
            title="فارغ",
            content="",
            output_dir=str(tmp_dir),
        )
        assert "تم توليد" in result


class TestRegistryWithExtendedTools:

    def test_all_tools_registered(self):
        from src.agent.tools import build_default_registry
        registry = build_default_registry()
        names = registry.list_names()
        assert "web_search" in names
        assert "pubmed_search" in names
        assert "generate_image" in names
        assert "generate_report" in names
        assert "rag_search" in names
        assert "medical_calculator" in names
        assert "rehab_exercises" in names
        assert len(names) == 9  # 5 original + 4 new

    def test_tool_descriptions_are_bilingual(self):
        from src.agent.tools import build_default_registry
        registry = build_default_registry()
        for tool_info in registry.list_tools():
            desc = tool_info["description"]
            # Should have some Arabic content
            has_arabic = any('\u0600' <= c <= '\u06FF' for c in desc)
            assert has_arabic, f"Tool {tool_info['name']} has no Arabic description"
