"""
GRPO Verifiable Reward Functions — standalone module for testability.
دوال مكافأة GRPO القابلة للتحقق — وحدة مستقلة للاختبار

Extracted from train.py closures so they can be:
1. Unit tested independently
2. Reused across different training pipelines
3. Extended without modifying training logic
"""
import re
from loguru import logger


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for comparison.
    تطبيع النص العربي للمقارنة

    - Removes tashkeel (diacritics)
    - Normalizes alef variants (إأآا → ا)
    - Normalizes taa marbuta (ة → ه)
    - Lowercases and normalizes whitespace
    """
    # Remove tashkeel (diacritics)
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670]', '', text)
    # Normalize alef variants
    text = re.sub(r'[إأآا]', 'ا', text)
    # Normalize taa marbuta
    text = text.replace('ة', 'ه')
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def accuracy_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Verifiable accuracy reward — compares against ground-truth answers.
    مكافأة الدقة القابلة للتحقق — تقارن مع الإجابات المرجعية

    Args:
        completions: Model-generated responses
        **kwargs: Must include 'answer' (list[str]) for verification

    Returns:
        List of reward scores [0.0 - 1.0]
    """
    expected_answers = kwargs.get("answer", [])
    rewards = []

    for i, completion in enumerate(completions):
        if not expected_answers or i >= len(expected_answers):
            rewards.append(0.0)
            continue

        expected = expected_answers[i] if i < len(expected_answers) else ""
        if not expected:
            rewards.append(0.0)
            continue

        # Normalize both for comparison
        comp_norm = normalize_arabic(completion)
        exp_norm = normalize_arabic(expected)

        # Exact match (expected found within completion)
        if exp_norm in comp_norm:
            rewards.append(1.0)
        else:
            # Partial match — key terms overlap
            exp_terms = set(exp_norm.split())
            comp_terms = set(comp_norm.split())
            if exp_terms:
                overlap = len(exp_terms & comp_terms) / len(exp_terms)
                rewards.append(round(overlap, 2))
            else:
                rewards.append(0.0)

    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Format compliance reward — verifiable structural checks.
    مكافأة الالتزام بالتنسيق — فحوصات هيكلية قابلة للتحقق

    Updated with Nemotron-style XML reasoning format support.
    Checks for both <think>/<reasoning> tag formats.

    Returns:
        List of reward scores [0.0 - 1.0]
    """
    rewards = []
    for completion in completions:
        score = 0.0
        text = completion.strip()

        if not text:
            rewards.append(0.0)
            continue

        # Reasonable length
        word_count = len(text.split())
        if 5 <= word_count <= 2000:
            score += 0.25
        elif word_count > 0:
            score += 0.1

        # No degenerate repetition
        words = text.split()
        has_repetition = False
        if len(words) > 30:
            for j in range(len(words) - 30):
                seq = " ".join(words[j:j + 10])
                if text.count(seq) >= 3:
                    has_repetition = True
                    break
        if not has_repetition:
            score += 0.25

        # Reasoning structure — supports both Nemotron XML and DeepSeek think formats
        has_reasoning = _check_reasoning_tags(text)
        if has_reasoning == "valid":
            score += 0.3  # Well-structured reasoning
        elif has_reasoning == "malformed":
            score -= 0.1

        # Ends with complete sentence
        if text[-1] in ".!?؟。":
            score += 0.2

        rewards.append(max(0.0, min(1.0, score)))

    return rewards


def _check_reasoning_tags(text: str) -> str:
    """
    Check for valid reasoning tag structure.
    يفحص صحة هيكل علامات الاستدلال

    Supports:
    - Nemotron XML: <reasoning>...</reasoning><answer>...</answer>
    - DeepSeek think: <think>...</think>
    - Plain: no tags (neutral)

    Returns: "valid", "malformed", or "none"
    """
    # Check Nemotron XML format
    has_reasoning_open = "<reasoning>" in text
    has_reasoning_close = "</reasoning>" in text
    has_answer_open = "<answer>" in text
    has_answer_close = "</answer>" in text

    if has_reasoning_open or has_answer_open:
        if (has_reasoning_open and has_reasoning_close and
            has_answer_open and has_answer_close):
            r_open = text.index("<reasoning>")
            r_close = text.index("</reasoning>")
            a_open = text.index("<answer>")
            a_close = text.index("</answer>")
            if r_open < r_close < a_open < a_close:
                return "valid"
        return "malformed"

    # Check DeepSeek think format
    has_think_open = "<think>" in text
    has_think_close = "</think>" in text

    if has_think_open:
        if has_think_close:
            t_open = text.index("<think>")
            t_close = text.index("</think>")
            if t_close > t_open:
                return "valid"
        return "malformed"

    return "none"


def xml_format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Strict XML reasoning format reward — Nemotron style.
    مكافأة تنسيق XML صارمة — نمط Nemotron

    Expects: <reasoning>...</reasoning><answer>...</answer>
    Used when config grpo.reasoning_format = "xml"

    Scores:
    - 1.0: Perfect XML format with non-empty reasoning and answer
    - 0.5: Has tags but empty content
    - 0.0: Missing or malformed tags
    """
    import re
    rewards = []
    for completion in completions:
        text = completion.strip()

        # Extract reasoning and answer
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)

        if reasoning_match and answer_match:
            reasoning = reasoning_match.group(1).strip()
            answer = answer_match.group(1).strip()

            if reasoning and answer:
                rewards.append(1.0)  # Perfect format
            elif answer:
                rewards.append(0.5)  # Has answer but empty reasoning
            else:
                rewards.append(0.25)  # Has structure but empty
        elif answer_match:
            rewards.append(0.3)  # Answer without reasoning
        else:
            rewards.append(0.0)  # No structure

    return rewards


def coherence_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Language coherence reward — checks language consistency and output quality.
    مكافأة تماسك اللغة — تفحص اتساق اللغة وجودة المخرجات

    Checks:
    - Arabic content ratio (rewards consistency)
    - Pure English acceptable for technical content
    - Penalizes garbage (excessive symbols)

    Returns:
        List of reward scores [0.0 - 1.0]
    """
    rewards = []
    for completion in completions:
        text = completion.strip()
        if not text:
            rewards.append(0.0)
            continue

        score = 0.5  # Neutral baseline

        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        total_alpha = arabic_chars + latin_chars

        if total_alpha > 0:
            arabic_ratio = arabic_chars / total_alpha
            if arabic_ratio > 0.6:
                score += 0.3
            elif arabic_ratio > 0.3:
                score += 0.1
            elif latin_chars > 0 and arabic_ratio < 0.1:
                score += 0.2

        # Penalize garbage
        symbol_ratio = len(re.findall(r'[^\w\s\u0600-\u06FF.,!?؟;:(){}\[\]]', text)) / max(len(text), 1)
        if symbol_ratio > 0.3:
            score -= 0.3

        rewards.append(max(0.0, min(1.0, score)))

    return rewards
