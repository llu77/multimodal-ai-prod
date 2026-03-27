"""
Agent Memory — Conversation history + long-term persistent memory.
ذاكرة الوكيل — سجل المحادثات + ذاكرة طويلة المدى

Two types of memory:
1. ConversationMemory: Current session messages (short-term)
2. PersistentMemory: Cross-session facts about the user (long-term)

This is what makes the agent "remember" between turns and sessions.
"""
import json
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Message:
    """A single message in conversation."""
    role: str           # "system", "user", "assistant", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_name: str = ""     # If role="tool", which tool produced this
    tool_input: str = ""    # What was passed to the tool
    metadata: dict = field(default_factory=dict)


class ConversationMemory:
    """
    Short-term conversation memory — current session.
    ذاكرة المحادثة القصيرة — الجلسة الحالية

    Keeps the last N turns. Automatically summarizes older turns
    to stay within context window limits.
    """

    def __init__(self, max_turns: int = 50, max_tokens_estimate: int = 8000):
        self.messages: list[Message] = []
        self.max_turns = max_turns
        self.max_tokens = max_tokens_estimate
        self._system_prompt: str = ""

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt

    def add_user(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))
        self._trim()

    def add_assistant(self, content: str) -> None:
        self.messages.append(Message(role="assistant", content=content))
        self._trim()

    def add_tool_call(self, tool_name: str, tool_input: str, result: str) -> None:
        """Record a tool invocation and its result."""
        self.messages.append(Message(
            role="tool",
            content=result,
            tool_name=tool_name,
            tool_input=tool_input,
        ))
        self._trim()

    def get_messages_for_llm(self) -> list[dict]:
        """
        Format messages for LLM input.
        تنسيق الرسائل لإدخال النموذج

        Returns list of {"role": ..., "content": ...} dicts.
        Tool messages are formatted as assistant observations.
        """
        formatted = []

        if self._system_prompt:
            formatted.append({"role": "system", "content": self._system_prompt})

        for msg in self.messages:
            if msg.role == "tool":
                # Format tool results as part of assistant's thinking
                formatted.append({
                    "role": "assistant",
                    "content": f"[أداة: {msg.tool_name}]\nالمدخل: {msg.tool_input}\nالنتيجة: {msg.content}",
                })
            else:
                formatted.append({"role": msg.role, "content": msg.content})

        return formatted

    def get_last_n(self, n: int = 5) -> list[Message]:
        return self.messages[-n:]

    def clear(self) -> None:
        self.messages.clear()

    def _trim(self) -> None:
        """Remove oldest messages if exceeding limits."""
        if len(self.messages) > self.max_turns:
            # Keep first 2 (system context) and last max_turns-2
            overflow = len(self.messages) - self.max_turns
            self.messages = self.messages[overflow:]

    def _estimate_tokens(self) -> int:
        return sum(len(m.content.split()) * 2 for m in self.messages)

    @property
    def turn_count(self) -> int:
        return len([m for m in self.messages if m.role == "user"])

    def __len__(self) -> int:
        return len(self.messages)


class PersistentMemory:
    """
    Long-term memory — persists across sessions.
    ذاكرة طويلة المدى — تستمر عبر الجلسات

    Stores structured facts about the user/context.
    Saved as JSON file — simple and portable.

    Usage:
        memory.store("patient_allergies", ["بنسلين", "أسبرين"])
        memory.store("preferred_language", "ar")
        allergies = memory.recall("patient_allergies")
    """

    def __init__(self, storage_path: str = "./data/memory/persistent.json"):
        self._path = Path(storage_path)
        self._data: dict[str, dict] = {}
        self._load()

    def store(self, key: str, value: any, category: str = "general") -> None:
        """Store a fact in long-term memory."""
        self._data[key] = {
            "value": value,
            "category": category,
            "updated_at": time.time(),
            "access_count": self._data.get(key, {}).get("access_count", 0),
        }
        self._save()

    def recall(self, key: str) -> Optional[any]:
        """Recall a fact from long-term memory."""
        entry = self._data.get(key)
        if entry:
            entry["access_count"] = entry.get("access_count", 0) + 1
            return entry["value"]
        return None

    def search(self, keyword: str) -> list[tuple[str, any]]:
        """Search memory by keyword in keys and string values."""
        results = []
        keyword_lower = keyword.lower()
        for key, entry in self._data.items():
            val = entry["value"]
            if keyword_lower in key.lower():
                results.append((key, val))
            elif isinstance(val, str) and keyword_lower in val.lower():
                results.append((key, val))
        return results

    def list_all(self) -> dict[str, any]:
        """Return all stored facts."""
        return {k: v["value"] for k, v in self._data.items()}

    def forget(self, key: str) -> bool:
        """Remove a fact from memory."""
        if key in self._data:
            del self._data[key]
            self._save()
            return True
        return False

    def get_context_summary(self) -> str:
        """
        Generate a summary of stored facts for the agent's system prompt.
        توليد ملخص الحقائق المخزنة لـ system prompt الوكيل
        """
        if not self._data:
            return ""

        lines = ["معلومات محفوظة عن المستخدم/السياق:"]
        for key, entry in self._data.items():
            val = entry["value"]
            if isinstance(val, list):
                val = "، ".join(str(v) for v in val)
            lines.append(f"  • {key}: {val}")

        return "\n".join(lines)

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.debug(f"Loaded {len(self._data)} memory entries")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load memory: {e}")
                self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        return len(self._data)
