"""
Tests for the Agent layer — tools, memory, orchestrator.
اختبارات طبقة الوكيل — الأدوات، الذاكرة، المنسّق
"""
import json
import time
import pytest
from src.agent.tools import (
    ToolRegistry, Tool, build_default_registry,
    _medical_calculator, _rehab_exercise_lookup, _get_current_datetime, _summarize_text,
)
from src.agent.memory import ConversationMemory, PersistentMemory, Message
from src.agent.orchestrator import RehabAgent, AgentConfig, AgentResponse


# ═══════════════════════════════════════════
# Tool Registry
# ═══════════════════════════════════════════

class TestToolRegistry:

    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = Tool(name="test", description="desc", parameters={}, execute=lambda: "ok")
        registry.register(tool)
        assert registry.get("test") is tool

    def test_execute_success(self):
        registry = ToolRegistry()
        registry.register(Tool(name="add", description="", parameters={}, execute=lambda a, b: a + b))
        result = registry.execute("add", a=2, b=3)
        assert result["result"] == 5
        assert result["error"] is None

    def test_execute_missing_tool(self):
        registry = ToolRegistry()
        result = registry.execute("nonexistent")
        assert result["error"] is not None
        assert "not found" in result["error"]

    def test_execute_tool_failure(self):
        registry = ToolRegistry()
        registry.register(Tool(name="crash", description="", parameters={}, execute=lambda: 1 / 0))
        result = registry.execute("crash")
        assert result["error"] is not None

    def test_list_tools(self):
        registry = build_default_registry()
        tools = registry.list_tools()
        assert len(tools) >= 5
        names = [t["name"] for t in tools]
        assert "rag_search" in names
        assert "medical_calculator" in names
        assert "rehab_exercises" in names


# ═══════════════════════════════════════════
# Medical Calculator
# ═══════════════════════════════════════════

class TestMedicalCalculator:

    def test_bmi_normal(self):
        result = _medical_calculator("", calc_type="bmi", weight_kg=70, height_cm=175)
        assert "22.9" in result
        assert "طبيعي" in result

    def test_bmi_obese(self):
        result = _medical_calculator("", calc_type="bmi", weight_kg=120, height_cm=170)
        assert "سمنة" in result

    def test_bmi_invalid(self):
        result = _medical_calculator("", calc_type="bmi", weight_kg=0, height_cm=0)
        assert "خطأ" in result

    def test_gfr_normal(self):
        result = _medical_calculator("", calc_type="gfr", creatinine=0.9, age=35)
        assert "الترشيح" in result

    def test_dosage(self):
        result = _medical_calculator("", calc_type="dosage", weight_kg=70, dose_per_kg=10, frequency=3)
        assert "700" in result  # 70 * 10

    def test_unknown_type(self):
        result = _medical_calculator("", calc_type="xyz")
        assert "غير مدعوم" in result


# ═══════════════════════════════════════════
# Rehab Exercises
# ═══════════════════════════════════════════

class TestRehabExercises:

    def test_shoulder_default(self):
        result = _rehab_exercise_lookup("الكتف")
        assert "الكتف" in result
        assert "البندول" in result or "Pendulum" in result

    def test_shoulder_frozen(self):
        result = _rehab_exercise_lookup("الكتف", "تيبس")
        assert "الشد" in result or "Stretching" in result

    def test_knee(self):
        result = _rehab_exercise_lookup("الركبة")
        assert "الركبة" in result

    def test_unknown_body_part(self):
        result = _rehab_exercise_lookup("الأذن")
        assert "لم يتم العثور" in result

    def test_datetime(self):
        result = _get_current_datetime()
        assert "التاريخ" in result

    def test_summarize(self):
        text = "الجملة الأولى طويلة جداً وتحتوي معلومات مهمة. الجملة الثانية أقصر. الجملة الثالثة طويلة جداً أيضاً ومليئة بالتفاصيل المهمة."
        result = _summarize_text(text, max_sentences=2)
        assert len(result) < len(text)


# ═══════════════════════════════════════════
# Conversation Memory
# ═══════════════════════════════════════════

class TestConversationMemory:

    def test_add_and_retrieve(self):
        mem = ConversationMemory()
        mem.add_user("مرحبا")
        mem.add_assistant("أهلاً!")
        assert len(mem) == 2

    def test_get_messages_for_llm(self):
        mem = ConversationMemory()
        mem.set_system_prompt("أنت مساعد.")
        mem.add_user("سؤال")
        mem.add_assistant("إجابة")
        messages = mem.get_messages_for_llm()
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_tool_call_formatting(self):
        mem = ConversationMemory()
        mem.add_tool_call("rag_search", '{"query": "test"}', "نتيجة البحث")
        messages = mem.get_messages_for_llm()
        assert "rag_search" in messages[0]["content"]
        assert "نتيجة البحث" in messages[0]["content"]

    def test_trim_old_messages(self):
        mem = ConversationMemory(max_turns=5)
        for i in range(10):
            mem.add_user(f"msg {i}")
        assert len(mem) <= 5

    def test_turn_count(self):
        mem = ConversationMemory()
        mem.add_user("q1")
        mem.add_assistant("a1")
        mem.add_user("q2")
        assert mem.turn_count == 2

    def test_clear(self):
        mem = ConversationMemory()
        mem.add_user("test")
        mem.clear()
        assert len(mem) == 0


# ═══════════════════════════════════════════
# Persistent Memory
# ═══════════════════════════════════════════

class TestPersistentMemory:

    def test_store_and_recall(self, tmp_dir):
        mem = PersistentMemory(storage_path=str(tmp_dir / "mem.json"))
        mem.store("name", "عمر")
        assert mem.recall("name") == "عمر"

    def test_recall_missing_key(self, tmp_dir):
        mem = PersistentMemory(storage_path=str(tmp_dir / "mem.json"))
        assert mem.recall("nonexistent") is None

    def test_search(self, tmp_dir):
        mem = PersistentMemory(storage_path=str(tmp_dir / "mem.json"))
        mem.store("allergy_penicillin", "حساسية شديدة")
        mem.store("allergy_aspirin", "حساسية خفيفة")
        mem.store("blood_type", "O+")
        results = mem.search("allergy")
        assert len(results) == 2

    def test_forget(self, tmp_dir):
        mem = PersistentMemory(storage_path=str(tmp_dir / "mem.json"))
        mem.store("temp", "value")
        assert mem.forget("temp") is True
        assert mem.recall("temp") is None

    def test_persistence(self, tmp_dir):
        path = str(tmp_dir / "persist.json")
        mem1 = PersistentMemory(storage_path=path)
        mem1.store("key", "value")
        # Load fresh instance from same file
        mem2 = PersistentMemory(storage_path=path)
        assert mem2.recall("key") == "value"

    def test_context_summary(self, tmp_dir):
        mem = PersistentMemory(storage_path=str(tmp_dir / "mem.json"))
        mem.store("patient_name", "أحمد")
        mem.store("allergies", ["بنسلين", "أسبرين"])
        summary = mem.get_context_summary()
        assert "أحمد" in summary
        assert "بنسلين" in summary

    def test_list_all(self, tmp_dir):
        mem = PersistentMemory(storage_path=str(tmp_dir / "mem.json"))
        mem.store("a", 1)
        mem.store("b", 2)
        all_facts = mem.list_all()
        assert all_facts == {"a": 1, "b": 2}


# ═══════════════════════════════════════════
# Agent Orchestrator
# ═══════════════════════════════════════════

class TestAgentOrchestrator:

    def _make_agent(self, responses: list[str], tmp_dir) -> RehabAgent:
        """Create agent with a fake model that returns predefined responses."""
        response_iter = iter(responses)

        def fake_generate(messages):
            return next(response_iter, "لا أعرف.")

        return RehabAgent(
            generate_fn=fake_generate,
            config=AgentConfig(
                max_steps=3,
                memory_path=str(tmp_dir / "test_mem.json"),
                verbose=False,
            ),
        )

    def test_direct_answer(self, tmp_dir):
        """Agent answers directly without tools."""
        agent = self._make_agent(["مرحباً! كيف أقدر أساعدك؟"], tmp_dir)
        response = agent.run("مرحبا")
        assert isinstance(response, AgentResponse)
        assert "مرحباً" in response.answer
        assert len(response.tools_used) == 0

    def test_tool_call_then_answer(self, tmp_dir):
        """Agent calls a tool, gets result, then answers."""
        agent = self._make_agent([
            # First call: model decides to use a tool
            '<tool_call>\n{"tool": "rehab_exercises", "input": {"body_part": "الكتف"}}\n</tool_call>',
            # Second call: model sees tool result and gives final answer
            "بناءً على التمارين المتاحة، أنصح بتمارين البندول وتمارين نطاق الحركة.",
        ], tmp_dir)

        response = agent.run("ما تمارين الكتف؟")
        assert "rehab_exercises" in response.tools_used
        assert response.total_steps >= 2
        assert "البندول" in response.answer or "تمارين" in response.answer

    def test_calculator_tool(self, tmp_dir):
        """Agent uses medical calculator."""
        agent = self._make_agent([
            '<tool_call>\n{"tool": "medical_calculator", "input": {"calc_type": "bmi", "weight_kg": 85, "height_cm": 175}}\n</tool_call>',
            "مؤشر كتلة الجسم هو 27.8 — وزن زائد. أنصح بممارسة الرياضة.",
        ], tmp_dir)

        response = agent.run("وزني 85 كيلو وطولي 175، كم مؤشر كتلة الجسم؟")
        assert "medical_calculator" in response.tools_used

    def test_max_steps_limit(self, tmp_dir):
        """Agent stops after max_steps even if still calling tools."""
        # All responses are tool calls — should stop at max_steps
        agent = self._make_agent([
            '<tool_call>\n{"tool": "datetime", "input": {}}\n</tool_call>',
            '<tool_call>\n{"tool": "datetime", "input": {}}\n</tool_call>',
            '<tool_call>\n{"tool": "datetime", "input": {}}\n</tool_call>',
            '<tool_call>\n{"tool": "datetime", "input": {}}\n</tool_call>',
            "الجواب النهائي.",
        ], tmp_dir)

        response = agent.run("ما الوقت الآن؟")
        assert response.total_steps <= 4  # max_steps=3 + 1 forced answer

    def test_extract_tool_call_valid(self):
        text = 'بعض النص <tool_call>\n{"tool": "rag_search", "input": {"query": "كتف"}}\n</tool_call> نص آخر'
        result = RehabAgent._extract_tool_call(text)
        assert result is not None
        assert result["tool"] == "rag_search"

    def test_extract_tool_call_none(self):
        result = RehabAgent._extract_tool_call("نص عادي بدون أي أداة")
        assert result is None

    def test_extract_tool_call_malformed(self):
        result = RehabAgent._extract_tool_call("<tool_call>not json</tool_call>")
        assert result is None

    def test_memory_integration(self, tmp_dir):
        """Agent remembers facts across calls."""
        agent = self._make_agent(["تم الحفظ.", "أحمد."], tmp_dir)
        agent.remember("patient_name", "أحمد")
        assert agent.recall("patient_name") == "أحمد"

    def test_conversation_persists(self, tmp_dir):
        """Conversation memory accumulates across turns."""
        agent = self._make_agent(["إجابة 1", "إجابة 2"], tmp_dir)
        agent.run("سؤال 1")
        agent.run("سؤال 2")
        assert agent.conversation.turn_count == 2

    def test_reset_conversation(self, tmp_dir):
        agent = self._make_agent(["إجابة"], tmp_dir)
        agent.run("سؤال")
        agent.reset_conversation()
        assert len(agent.conversation) == 0

    def test_response_has_latency(self, tmp_dir):
        agent = self._make_agent(["إجابة"], tmp_dir)
        response = agent.run("سؤال")
        assert response.latency_ms > 0

    def test_reasoning_trace(self, tmp_dir):
        agent = self._make_agent([
            '<tool_call>\n{"tool": "datetime", "input": {}}\n</tool_call>',
            "الوقت الآن كذا.",
        ], tmp_dir)
        response = agent.run("ما الوقت؟")
        trace = response.reasoning_trace()
        assert "datetime" in trace
