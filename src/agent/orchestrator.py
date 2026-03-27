"""
Agent Orchestrator — The brain that thinks, decides, acts, observes.
منسّق الوكيل — العقل الذي يفكّر، يقرر، ينفّذ، يراقب

This is the core that transforms a "text generator" into an "agent".

ReAct Loop:
    1. THINK: What does the user need? Do I need more info?
    2. DECIDE: Which tool (if any) should I use?
    3. ACT: Call the tool
    4. OBSERVE: Read the tool's result
    5. REPEAT or ANSWER: Do I need another step, or can I answer now?

Compatible with:
    - Local model (our Phi-4 Multimodal)
    - External API (Claude, GPT) as orchestrator
    - LangGraph (can be wrapped as a node)
"""
import re
import json
import time
from typing import Optional, Callable
from dataclasses import dataclass, field
from loguru import logger

from src.agent.tools import ToolRegistry, build_default_registry
from src.agent.memory import ConversationMemory, PersistentMemory


def _build_full_registry(rag_engine=None) -> ToolRegistry:
    """Build registry with base + advanced skills."""
    registry = build_default_registry()
    try:
        from src.agent.skills import register_advanced_skills
        # Only register skills not already in the registry (avoid overwriting
        # higher-quality implementations from tools_extended.py)
        register_advanced_skills(registry, skip_existing=True)
    except ImportError:
        logger.warning("Advanced skills not available")
    return registry


# ═══════════════════════════════════════════
# Agent Configuration
# ═══════════════════════════════════════════

@dataclass
class AgentConfig:
    """Agent behavior configuration."""
    max_steps: int = 5                  # Max tool calls per turn (prevents infinite loops)
    temperature: float = 0.3            # Low = more focused decisions
    max_tokens: int = 2048
    language: str = "ar"                # Primary language
    persona: str = "أخصائي إعادة تأهيل طبي"
    memory_path: str = "./data/memory/persistent.json"
    verbose: bool = True                # Log agent reasoning


# ═══════════════════════════════════════════
# System Prompts
# ═══════════════════════════════════════════

AGENT_SYSTEM_PROMPT = """أنت {persona} — وكيل ذكي يساعد في المجال الطبي وإعادة التأهيل.

لديك القدرة على استخدام أدوات للبحث والحساب والاستعلام. لا تخمّن — إذا لم تكن متأكداً، استخدم أداة.

## الأدوات المتاحة:
{tools_description}

## طريقة العمل:
عندما تتلقى سؤالاً:
1. فكّر: ما الذي يحتاجه المستخدم بالضبط؟
2. قرر: هل أحتاج أداة أم أستطيع الإجابة مباشرة؟
3. إذا احتجت أداة، استدعِها بالتنسيق:
   <tool_call>
   {{"tool": "اسم_الأداة", "input": {{"param1": "value1"}}}}
   </tool_call>
4. بعد الحصول على النتيجة، قرر: هل أحتاج أداة أخرى أم أستطيع الإجابة؟
5. عندما تكون جاهزاً للإجابة النهائية، اكتب إجابتك مباشرة بدون tool_call.

## قواعد مهمة:
- أجب دائماً باللغة العربية إلا إذا طُلب غير ذلك
- لا تستدعِ أكثر من {max_steps} أدوات في نفس السؤال
- إذا لم تجد معلومات كافية، اعترف بذلك بصراحة
- لا تختلق معلومات طبية — استخدم rag_search للتحقق

{memory_context}"""


# ═══════════════════════════════════════════
# Agent Response
# ═══════════════════════════════════════════

@dataclass
class AgentStep:
    """A single step in the agent's reasoning."""
    step_type: str          # "think", "tool_call", "tool_result", "answer"
    content: str
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResponse:
    """Complete agent response with reasoning trace."""
    answer: str                         # Final answer to the user
    steps: list[AgentStep] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    total_steps: int = 0
    latency_ms: float = 0.0

    def reasoning_trace(self) -> str:
        """Human-readable reasoning trace."""
        lines = []
        for i, step in enumerate(self.steps, 1):
            if step.step_type == "tool_call":
                lines.append(f"  الخطوة {i}: استدعاء أداة [{step.tool_name}]")
                lines.append(f"    المدخل: {json.dumps(step.tool_input, ensure_ascii=False)}")
            elif step.step_type == "tool_result":
                lines.append(f"    النتيجة: {step.content[:200]}...")
            elif step.step_type == "think":
                lines.append(f"  التفكير: {step.content[:200]}")
        return "\n".join(lines)


# ═══════════════════════════════════════════
# The Agent
# ═══════════════════════════════════════════

class RehabAgent:
    """
    Medical Rehabilitation Agent — thinks, searches, calculates, answers.
    وكيل إعادة التأهيل الطبي — يفكّر، يبحث، يحسب، يجيب

    This is the core class that transforms a language model into an agent.

    Usage:
        agent = RehabAgent(generate_fn=my_model_generate)
        response = agent.run("ما خطة إعادة تأهيل مريض بعد جلطة دماغية؟")
        print(response.answer)
        print(response.reasoning_trace())
    """

    def __init__(
        self,
        generate_fn: Callable[[list[dict]], str],
        config: Optional[AgentConfig] = None,
        rag_engine=None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        """
        Args:
            generate_fn: Function(messages: list[dict]) → str
                         Takes chat messages, returns model response text.
                         Can be local model OR external API (Claude/GPT).
            config: Agent configuration
            rag_engine: RAG engine instance for search tool
            tool_registry: Custom tool registry (or uses defaults)
        """
        self.generate = generate_fn
        self.config = config or AgentConfig()
        self.tools = tool_registry or _build_full_registry()
        self.rag_engine = rag_engine

        # Memory
        self.conversation = ConversationMemory(max_turns=50)
        self.memory = PersistentMemory(storage_path=self.config.memory_path)

        # Build system prompt
        self._build_system_prompt()

        logger.info(f"RehabAgent initialized — {len(self.tools.list_names())} tools available")

    def _build_system_prompt(self) -> None:
        """Build the agent's system prompt with tools and memory."""
        tools_desc = ""
        for tool_info in self.tools.list_tools():
            tools_desc += f"\n### {tool_info['name']}\n{tool_info['description']}\n"
            tools_desc += f"المدخلات: {json.dumps(tool_info['parameters'], ensure_ascii=False)}\n"

        memory_context = self.memory.get_context_summary()
        if memory_context:
            memory_context = f"\n## معلومات محفوظة:\n{memory_context}\n"

        self._system_prompt = AGENT_SYSTEM_PROMPT.format(
            persona=self.config.persona,
            tools_description=tools_desc,
            max_steps=self.config.max_steps,
            memory_context=memory_context,
        )
        self.conversation.set_system_prompt(self._system_prompt)

    def run(self, user_message: str) -> AgentResponse:
        """
        Process a user message through the full agent loop.
        معالجة رسالة المستخدم عبر حلقة الوكيل الكاملة

        This is the main entry point. It:
        1. Adds the message to conversation memory
        2. Runs the ReAct loop (think → act → observe → repeat)
        3. Returns the final answer with reasoning trace
        """
        start_time = time.time()
        steps = []
        tools_used = []

        # Add user message to memory
        self.conversation.add_user(user_message)

        for step_num in range(self.config.max_steps + 1):
            # Get model response
            messages = self.conversation.get_messages_for_llm()
            raw_response = self.generate(messages)

            # Check if response contains a tool call
            tool_call = self._extract_tool_call(raw_response)

            if tool_call:
                tool_name = tool_call["tool"]
                tool_input = tool_call.get("input", {})

                steps.append(AgentStep(
                    step_type="tool_call",
                    content=f"Calling {tool_name}",
                    tool_name=tool_name,
                    tool_input=tool_input,
                ))

                if self.config.verbose:
                    logger.info(f"  🔧 Step {step_num + 1}: {tool_name}({json.dumps(tool_input, ensure_ascii=False)[:100]})")

                # Inject RAG engine if needed
                if tool_name == "rag_search" and self.rag_engine:
                    tool_input["_rag_engine"] = self.rag_engine

                # Execute tool
                result = self.tools.execute(tool_name, **tool_input)
                result_text = str(result.get("result", result.get("error", "خطأ غير معروف")))

                steps.append(AgentStep(
                    step_type="tool_result",
                    content=result_text,
                    tool_name=tool_name,
                ))
                tools_used.append(tool_name)

                # Add tool result to conversation so model can see it
                self.conversation.add_tool_call(tool_name, json.dumps(tool_input, ensure_ascii=False, default=str), result_text)

            else:
                # No tool call — this is the final answer
                # Clean up any partial formatting
                answer = self._clean_answer(raw_response)

                steps.append(AgentStep(step_type="answer", content=answer))
                self.conversation.add_assistant(answer)

                latency = round((time.time() - start_time) * 1000, 2)

                if self.config.verbose:
                    logger.info(f"  ✅ Answer ready ({len(tools_used)} tools, {latency}ms)")

                return AgentResponse(
                    answer=answer,
                    steps=steps,
                    tools_used=tools_used,
                    total_steps=step_num + 1,
                    latency_ms=latency,
                )

        # Max steps exceeded — force an answer
        messages = self.conversation.get_messages_for_llm()
        messages.append({"role": "user", "content": "أجب الآن مباشرة بناءً على المعلومات المتاحة."})
        final = self.generate(messages)
        answer = self._clean_answer(final)
        self.conversation.add_assistant(answer)

        return AgentResponse(
            answer=answer,
            steps=steps,
            tools_used=tools_used,
            total_steps=self.config.max_steps,
            latency_ms=round((time.time() - start_time) * 1000, 2),
        )

    def remember(self, key: str, value: any) -> None:
        """Store a fact in long-term memory."""
        self.memory.store(key, value)
        self._build_system_prompt()  # Refresh prompt with new memory

    def recall(self, key: str) -> any:
        """Recall a fact from long-term memory."""
        return self.memory.recall(key)

    def reset_conversation(self) -> None:
        """Clear conversation history (keep long-term memory)."""
        self.conversation.clear()
        self._build_system_prompt()

    @staticmethod
    def _extract_tool_call(text: str) -> Optional[dict]:
        """
        Extract tool call from model response.
        استخراج استدعاء الأداة من استجابة النموذج

        Looks for: <tool_call>{"tool": "name", "input": {...}}</tool_call>
        """
        match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool_call JSON: {match.group(1)[:100]}")
            return None

    @staticmethod
    def _clean_answer(text: str) -> str:
        """Remove tool_call artifacts from final answer."""
        # Remove any lingering tool_call tags
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
        return text.strip()


# ═══════════════════════════════════════════
# Factory Functions
# ═══════════════════════════════════════════

def create_local_agent(cfg, inference_engine=None) -> RehabAgent:
    """
    Create agent using the local model as brain.
    إنشاء وكيل يستخدم النموذج المحلي كعقل

    Args:
        cfg: AppConfig
        inference_engine: Our MultimodalInferenceEngine
    """
    def local_generate(messages: list[dict]) -> str:
        if inference_engine is None:
            return "خطأ: محرك الاستنتاج غير متصل"
        from src.inference.engine import InferenceRequest
        # Use the last user message as the query
        last_user = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user = m["content"]
                break
        request = InferenceRequest(text=last_user)
        response = inference_engine.generate(request)
        return response.text

    return RehabAgent(
        generate_fn=local_generate,
        config=AgentConfig(memory_path="./data/memory/persistent.json"),
        rag_engine=inference_engine.rag if inference_engine else None,
    )


def create_api_agent(
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
) -> RehabAgent:
    """
    Create agent using external API (Claude/GPT) as brain.
    إنشاء وكيل يستخدم API خارجي (Claude/GPT) كعقل

    The external model handles the thinking/planning.
    Our local tools handle the domain-specific execution.

    Args:
        api_key: API key for the provider
        model: Model name
        provider: "anthropic" or "openai"
    """
    if provider == "anthropic":
        def anthropic_generate(messages: list[dict]) -> str:
            import httpx
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 2048,
                    "messages": [m for m in messages if m["role"] != "system"],
                    "system": next((m["content"] for m in messages if m["role"] == "system"), ""),
                },
                timeout=60.0,
            )
            data = response.json()
            return data["content"][0]["text"]

        return RehabAgent(
            generate_fn=anthropic_generate,
            config=AgentConfig(persona="وكيل إعادة تأهيل طبي ذكي (مدعوم بـ Claude)"),
        )

    elif provider == "openai":
        def openai_generate(messages: list[dict]) -> str:
            import httpx
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "messages": messages, "max_tokens": 2048},
                timeout=60.0,
            )
            data = response.json()
            return data["choices"][0]["message"]["content"]

        return RehabAgent(
            generate_fn=openai_generate,
            config=AgentConfig(persona="وكيل إعادة تأهيل طبي ذكي (مدعوم بـ GPT)"),
        )

    raise ValueError(f"Unknown provider: {provider}")
