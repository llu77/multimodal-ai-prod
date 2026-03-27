"""
Agent Tools — Functions the agent can decide to call.
أدوات الوكيل — الدوال التي يقرر الوكيل استدعاءها

Each tool is a simple function with:
- name: unique identifier
- description: Arabic+English (the agent reads this to decide when to use it)
- parameters: JSON schema
- execute: the actual function

The agent READS the descriptions and DECIDES which tool to use.
This is what separates an agent from a fixed pipeline.
"""
import re
import json
import math
from datetime import datetime, date
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Tool:
    """A tool the agent can invoke."""
    name: str
    description: str        # Agent reads this to decide when to use
    parameters: dict         # JSON schema of expected input
    execute: Callable        # The actual function
    category: str = "general"


class ToolRegistry:
    """
    Central registry of all available tools.
    السجل المركزي لجميع الأدوات المتاحة

    The agent receives this list and decides which tool to call.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.debug(f"Tool registered: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        """Return tool descriptions for the agent's system prompt."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def execute(self, name: str, **kwargs) -> dict:
        """Execute a tool by name. Returns {"result": ..., "error": ...}."""
        tool = self._tools.get(name)
        if not tool:
            return {"result": None, "error": f"Tool '{name}' not found"}
        try:
            result = tool.execute(**kwargs)
            return {"result": result, "error": None}
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}")
            return {"result": None, "error": str(e)}


# ═══════════════════════════════════════════
# Built-in Tools
# ═══════════════════════════════════════════

def _rag_search(query: str, top_k: int = 5, **kwargs) -> str:
    """RAG search — called by agent when it needs to look up information."""
    rag_engine = kwargs.get("_rag_engine")
    if not rag_engine:
        return "خطأ: محرك RAG غير متصل"

    docs = rag_engine.retrieve(query, top_k=top_k)
    if not docs:
        return "لم يتم العثور على معلومات ذات صلة في قاعدة المعرفة."

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", doc.metadata.get("filename", "مصدر غير معروف"))
        results.append(f"[{i}] (مصدر: {source})\n{doc.content[:500]}")

    return "\n\n---\n\n".join(results)


def _medical_calculator(calculation: str, **kwargs) -> str:
    """
    Medical calculator — BMI, dosage, GFR, etc.
    حاسبة طبية — مؤشر كتلة الجسم، الجرعات، معدل الترشيح الكبيبي
    """
    calc_type = kwargs.get("calc_type", "").lower()

    if calc_type == "bmi":
        weight = float(kwargs.get("weight_kg", 0))
        height = float(kwargs.get("height_cm", 0)) / 100
        if height <= 0 or weight <= 0:
            return "خطأ: الوزن والطول يجب أن يكونا أكبر من صفر"
        bmi = weight / (height ** 2)
        category = (
            "نقص وزن" if bmi < 18.5 else
            "وزن طبيعي" if bmi < 25 else
            "وزن زائد" if bmi < 30 else
            "سمنة"
        )
        return f"مؤشر كتلة الجسم: {bmi:.1f} ({category})"

    elif calc_type == "gfr":
        # CKD-EPI equation (simplified)
        creatinine = float(kwargs.get("creatinine", 0))
        age = int(kwargs.get("age", 0))
        is_female = kwargs.get("is_female", False)
        if creatinine <= 0 or age <= 0:
            return "خطأ: القيم يجب أن تكون أكبر من صفر"
        k = 0.7 if is_female else 0.9
        alpha = -0.329 if is_female else -0.411
        gfr = 141 * min(creatinine / k, 1) ** alpha * max(creatinine / k, 1) ** (-1.209) * 0.993 ** age
        if is_female:
            gfr *= 1.018
        stage = (
            "طبيعي (≥90)" if gfr >= 90 else
            "المرحلة 2 (60-89)" if gfr >= 60 else
            "المرحلة 3 (30-59)" if gfr >= 30 else
            "المرحلة 4 (15-29)" if gfr >= 15 else
            "المرحلة 5 (<15)"
        )
        return f"معدل الترشيح الكبيبي: {gfr:.1f} مل/دقيقة — {stage}"

    elif calc_type == "dosage":
        weight = float(kwargs.get("weight_kg", 0))
        dose_per_kg = float(kwargs.get("dose_per_kg", 0))
        frequency = int(kwargs.get("frequency", 1))
        if weight <= 0 or dose_per_kg <= 0:
            return "خطأ: القيم يجب أن تكون أكبر من صفر"
        single_dose = weight * dose_per_kg
        daily_dose = single_dose * frequency
        return f"الجرعة الواحدة: {single_dose:.1f} ملغ | الجرعة اليومية: {daily_dose:.1f} ملغ ({frequency}x/يوم)"

    return f"نوع الحساب غير مدعوم: {calc_type}. الأنواع المتاحة: bmi, gfr, dosage"


def _rehab_exercise_lookup(body_part: str, condition: str = "", **kwargs) -> str:
    """
    Rehabilitation exercise database lookup.
    البحث في قاعدة بيانات تمارين إعادة التأهيل
    """
    # Structured exercise database
    exercises = {
        "الكتف": {
            "default": [
                "تمارين البندول (Pendulum) — 3 مجموعات × 10 تكرارات",
                "تمارين نطاق الحركة السلبية (PROM) — 5 دقائق لكل اتجاه",
                "تمارين التقوية بالمطاط (Theraband) — 3 مجموعات × 12 تكرار",
                "تمارين الدوران الخارجي (External Rotation) — 3 × 10",
            ],
            "تيبس": [
                "تمارين الشد المتدرج (Progressive Stretching)",
                "تعبئة المفصل اليدوية (Joint Mobilization) — درجة II-III",
                "تمارين الحبل والبكرة (Pulley exercises)",
                "الكمادات الحرارية قبل التمارين — 15 دقيقة",
            ],
        },
        "الركبة": {
            "default": [
                "تمارين رفع الساق المستقيمة (SLR) — 3 × 15",
                "تمارين ضغط الركبة (Quad Sets) — 3 × 10 لمدة 5 ثوانٍ",
                "تمارين ثني وبسط الركبة — 3 × 15",
                "تمارين الخطوة (Step-ups) — 3 × 10 لكل جانب",
            ],
        },
        "الظهر": {
            "default": [
                "تمارين ماكنزي (McKenzie Extensions) — 10 تكرارات",
                "تمارين تقوية عضلات الجذع (Core Stabilization)",
                "تمارين Cat-Cow — 3 × 10",
                "تمارين Bird-Dog — 3 × 10 لكل جانب",
            ],
        },
    }

    body_lower = body_part.strip()
    for key, data in exercises.items():
        if key in body_lower:
            condition_lower = condition.strip().lower() if condition else ""
            if condition_lower and condition_lower in data:
                ex_list = data[condition_lower]
            else:
                ex_list = data["default"]
            header = f"تمارين إعادة تأهيل {key}"
            if condition:
                header += f" — حالة: {condition}"
            return header + "\n" + "\n".join(f"  • {e}" for e in ex_list)

    available = "، ".join(exercises.keys())
    return f"لم يتم العثور على تمارين لـ '{body_part}'. المناطق المتاحة: {available}"


def _get_current_datetime(**kwargs) -> str:
    """Return current date and time."""
    now = datetime.now()
    return f"التاريخ: {now.strftime('%Y-%m-%d')} | الوقت: {now.strftime('%H:%M')} | اليوم: {now.strftime('%A')}"


def _summarize_text(text: str, max_sentences: int = 3, **kwargs) -> str:
    """Simple extractive summarizer — picks most important sentences."""
    sentences = re.split(r'[.!?؟]\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if not sentences:
        return text[:500]
    # Score by length + position
    scored = [(i, len(s), s) for i, s in enumerate(sentences)]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = sorted(scored[:max_sentences], key=lambda x: x[0])
    return ". ".join(s[2] for s in top) + "."


# ═══════════════════════════════════════════
# Build Default Registry
# ═══════════════════════════════════════════

def build_default_registry() -> ToolRegistry:
    """Create registry with all built-in tools."""
    registry = ToolRegistry()

    # ── Core Tools ──

    registry.register(Tool(
        name="rag_search",
        description=(
            "البحث في قاعدة المعرفة الطبية المحلية. استخدم هذه الأداة عندما تحتاج معلومات "
            "من المراجع الطبية أو البروتوكولات السريرية أو إرشادات إعادة التأهيل المخزنة محلياً. "
            "Search the LOCAL medical knowledge base for clinical guidelines and rehab protocols."
        ),
        parameters={"query": "string — search query", "top_k": "int — number of results (default 5)"},
        execute=_rag_search,
        category="retrieval",
    ))

    # ── Extended Tools ──
    from src.agent.tools_extended import web_search, pubmed_search, generate_rehab_image, generate_report

    registry.register(Tool(
        name="web_search",
        description=(
            "البحث في الإنترنت عن معلومات حديثة. استخدمها عندما تحتاج معلومات "
            "غير موجودة في قاعدة المعرفة المحلية، أو لمعرفة آخر الأبحاث والأخبار الطبية. "
            "Search the internet for recent information not in local knowledge base."
        ),
        parameters={"query": "string — search query", "max_results": "int — max results (default 5)"},
        execute=web_search,
        category="retrieval",
    ))

    registry.register(Tool(
        name="pubmed_search",
        description=(
            "البحث في PubMed عن أبحاث طبية محكّمة. استخدمها عندما تحتاج أدلة علمية، "
            "دراسات سريرية، أو مراجعات منهجية لدعم قرار طبي. "
            "Search PubMed for peer-reviewed medical research papers and clinical evidence."
        ),
        parameters={"query": "string — medical search query", "max_results": "int — max results (default 5)"},
        execute=pubmed_search,
        category="retrieval",
    ))

    registry.register(Tool(
        name="medical_calculator",
        description=(
            "حاسبة طبية للمؤشرات الصحية. تحسب: مؤشر كتلة الجسم (BMI)، "
            "معدل الترشيح الكبيبي (GFR)، جرعات الأدوية حسب الوزن. "
            "Medical calculator: BMI, GFR (kidney function), weight-based dosing."
        ),
        parameters={
            "calc_type": "string — 'bmi', 'gfr', or 'dosage'",
            "weight_kg": "float", "height_cm": "float",
            "creatinine": "float", "age": "int", "is_female": "bool",
            "dose_per_kg": "float", "frequency": "int",
        },
        execute=_medical_calculator,
        category="calculation",
    ))

    registry.register(Tool(
        name="rehab_exercises",
        description=(
            "البحث في قاعدة بيانات تمارين إعادة التأهيل. يعطي تمارين محددة "
            "حسب المنطقة (الكتف، الركبة، الظهر) والحالة المرضية. "
            "Look up rehab exercises by body part and condition."
        ),
        parameters={"body_part": "string — body region", "condition": "string — diagnosis (optional)"},
        execute=_rehab_exercise_lookup,
        category="medical",
    ))

    registry.register(Tool(
        name="datetime",
        description="الحصول على التاريخ والوقت الحالي. Get current date and time.",
        parameters={},
        execute=_get_current_datetime,
        category="utility",
    ))

    registry.register(Tool(
        name="summarize",
        description=(
            "تلخيص نص طويل إلى نقاط رئيسية. استخدمها عندما تحتاج اختصار "
            "مستند أو نتائج بحث طويلة. Summarize long text into key points."
        ),
        parameters={"text": "string — text to summarize", "max_sentences": "int — max sentences (default 3)"},
        execute=_summarize_text,
        category="utility",
    ))

    registry.register(Tool(
        name="generate_image",
        description=(
            "توليد رسوم توضيحية لتمارين إعادة التأهيل. تنتج صور SVG طبية "
            "قابلة للطباعة بأي حجم. الأنواع: shoulder_pendulum, shoulder_flexion, "
            "knee_extension, back_extension, general_stretch. "
            "Generate rehab exercise illustrations as SVG images."
        ),
        parameters={
            "exercise_type": "string — type of exercise",
            "body_part": "string — body region (optional)",
            "title": "string — image title (optional)",
        },
        execute=generate_rehab_image,
        category="generation",
    ))

    registry.register(Tool(
        name="generate_report",
        description=(
            "توليد تقرير طبي أو تأهيلي كملف. يدعم: TXT (دائماً)، DOCX (Word)، PDF. "
            "استخدمها عندما يطلب المستخدم تقرير مكتوب أو ملخص رسمي. "
            "Generate a medical/rehab report as a file (TXT, DOCX, or PDF)."
        ),
        parameters={
            "title": "string — report title",
            "content": "string — report body text",
            "output_format": "string — 'txt', 'docx', or 'pdf' (default: txt)",
            "patient_name": "string — patient name (optional)",
        },
        execute=generate_report,
        category="generation",
    ))

    return registry
