"""
Tests for GRPO verifiable reward functions.
اختبارات دوال مكافأة GRPO القابلة للتحقق

These tests verify that rewards produce correct scores for known inputs.
Critical because bad rewards → bad model training → garbage outputs.
"""
import pytest
from src.training.rewards import (
    normalize_arabic,
    accuracy_reward,
    format_reward,
    coherence_reward,
)


# ═══════════════════════════════════════════
# Arabic Normalization
# ═══════════════════════════════════════════

class TestNormalizeArabic:
    """Normalization is the foundation of accuracy_reward — must be correct."""

    def test_removes_tashkeel(self):
        assert normalize_arabic("كِتَابٌ") == normalize_arabic("كتاب")

    def test_normalizes_alef_variants(self):
        # إ أ آ ا should all become ا
        assert normalize_arabic("إبراهيم") == normalize_arabic("ابراهيم")
        assert normalize_arabic("أحمد") == normalize_arabic("احمد")
        # آ → ا (single alef, not double)
        assert normalize_arabic("آمنة") == normalize_arabic("امنه")

    def test_normalizes_taa_marbuta(self):
        assert normalize_arabic("مدرسة") == normalize_arabic("مدرسه")

    def test_lowercases(self):
        assert normalize_arabic("ABC") == "abc"

    def test_normalizes_whitespace(self):
        assert normalize_arabic("كلمة   أخرى") == "كلمه اخرى"

    def test_empty_string(self):
        assert normalize_arabic("") == ""

    def test_english_passthrough(self):
        assert normalize_arabic("hello world") == "hello world"


# ═══════════════════════════════════════════
# Accuracy Reward
# ═══════════════════════════════════════════

class TestAccuracyReward:
    """Accuracy reward should score based on ground-truth matching."""

    def test_exact_match_scores_1(self):
        completions = ["الإجابة هي القلب"]
        scores = accuracy_reward(completions, answer=["القلب"])
        assert scores[0] == 1.0

    def test_answer_with_tashkeel_still_matches(self):
        completions = ["الجواب هو القَلْب"]
        scores = accuracy_reward(completions, answer=["القلب"])
        assert scores[0] == 1.0

    def test_no_match_scores_low(self):
        completions = ["الطقس جميل اليوم"]
        scores = accuracy_reward(completions, answer=["القلب"])
        assert scores[0] < 0.5

    def test_partial_match(self):
        completions = ["الشريان ينقل الدم بعيداً عن القلب"]
        scores = accuracy_reward(completions, answer=["الشريان ينقل الدم من القلب والوريد ينقل الدم إلى القلب"])
        assert 0.0 < scores[0] < 1.0  # Partial overlap

    def test_no_answer_provided_scores_zero(self):
        completions = ["أي إجابة هنا"]
        scores = accuracy_reward(completions)  # No answer kwarg
        assert scores[0] == 0.0

    def test_empty_completion_with_answer(self):
        completions = [""]
        scores = accuracy_reward(completions, answer=["القلب"])
        assert scores[0] == 0.0  # Can't match empty

    def test_multiple_completions(self):
        completions = ["القلب هو الجواب", "لا أعرف", "القلب يضخ الدم"]
        answers = ["القلب", "القلب", "القلب"]
        scores = accuracy_reward(completions, answer=answers)
        assert len(scores) == 3
        assert scores[0] == 1.0  # Exact match
        assert scores[1] < scores[0]  # Doesn't contain answer
        assert scores[2] == 1.0  # Contains answer

    def test_numeric_answer(self):
        completions = ["عدد العظام هو 206 عظمة"]
        scores = accuracy_reward(completions, answer=["206"])
        assert scores[0] == 1.0

    def test_mismatched_lengths_handled(self):
        # More completions than answers
        completions = ["إجابة 1", "إجابة 2", "إجابة 3"]
        scores = accuracy_reward(completions, answer=["القلب"])
        assert len(scores) == 3
        assert scores[1] == 0.0  # No answer for index 1
        assert scores[2] == 0.0


# ═══════════════════════════════════════════
# Format Reward
# ═══════════════════════════════════════════

class TestFormatReward:
    """Format reward checks structural quality of output."""

    def test_empty_scores_zero(self):
        scores = format_reward([""])
        assert scores[0] == 0.0

    def test_whitespace_only_scores_zero(self):
        scores = format_reward(["   \n\t  "])
        assert scores[0] == 0.0

    def test_good_response_scores_high(self):
        text = "هذا نص جيد يتكون من عدة كلمات ويشكل إجابة مفيدة ومنطقية."
        scores = format_reward([text])
        assert scores[0] >= 0.7  # Good length + no repetition + ends with period

    def test_single_word_scores_low(self):
        scores = format_reward(["كلمة"])
        assert scores[0] < 0.5

    def test_proper_think_tags_rewarded(self):
        text = "<think>لنفكر في هذا السؤال بعمق وبشكل منطقي</think> الإجابة هي كذا."
        scores = format_reward([text])
        assert scores[0] >= 0.5

    def test_malformed_think_tags_penalized(self):
        # Opening without closing
        text_open = "<think>تفكير بدون إغلاق والنص يستمر بشكل طويل نسبياً"
        # Reversed order
        text_reversed = "</think>خطأ<think> والنص يستمر بشكل طويل نسبياً هنا"

        scores_open = format_reward([text_open])
        scores_reversed = format_reward([text_reversed])

        good_text = "نص عادي بدون think tags ويتكون من كلمات كافية ومنطقية."
        scores_good = format_reward([good_text])

        # Malformed should score lower than clean text
        assert scores_open[0] < scores_good[0]

    def test_degenerate_repetition_penalized(self):
        # Create text with 10-word sequence repeated 3+ times
        repeated = "كلمة واحدة اثنتان ثلاث أربع خمس ست سبع ثمان تسع عشر "
        text = repeated * 5  # Repeat the whole sequence
        scores = format_reward([text])
        # Should not get the +0.3 for no-repetition
        # Compare with non-repetitive text of similar length
        unique_text = " ".join([f"كلمة_فريدة_{i}" for i in range(50)]) + "."
        scores_unique = format_reward([unique_text])
        assert scores[0] <= scores_unique[0]

    def test_ends_with_punctuation_rewarded(self):
        with_period = "إجابة مكونة من عدة كلمات مفيدة ومنطقية."
        without = "إجابة مكونة من عدة كلمات مفيدة ومنطقية"
        scores_with = format_reward([with_period])
        scores_without = format_reward([without])
        assert scores_with[0] > scores_without[0]

    def test_arabic_question_mark_counts(self):
        text = "هل هذا صحيح؟ نعم بالتأكيد هذا هو الجواب الصحيح؟"
        scores = format_reward([text])
        assert scores[0] > 0  # ؟ should count as sentence-ending

    def test_score_clamped_0_to_1(self):
        # Even worst case should be >= 0.0
        scores = format_reward(["x"])
        assert 0.0 <= scores[0] <= 1.0

        # Good case should be <= 1.0
        scores = format_reward(["نص ممتاز جداً مع <think>تفكير</think> وإجابة شاملة ومفصلة."])
        assert 0.0 <= scores[0] <= 1.0


# ═══════════════════════════════════════════
# Coherence Reward
# ═══════════════════════════════════════════

class TestCoherenceReward:
    """Coherence reward checks language consistency."""

    def test_empty_scores_zero(self):
        scores = coherence_reward([""])
        assert scores[0] == 0.0

    def test_pure_arabic_scores_high(self):
        text = "هذا نص عربي كامل بدون أي كلمات أجنبية ويتحدث عن موضوع مهم"
        scores = coherence_reward([text])
        assert scores[0] >= 0.7  # 0.5 base + 0.3 arabic ratio

    def test_pure_english_acceptable(self):
        text = "This is a purely English technical response about machine learning"
        scores = coherence_reward([text])
        assert scores[0] >= 0.5  # 0.5 base + 0.2 for pure English

    def test_mixed_moderate_score(self):
        text = "استخدمنا PyTorch و TensorFlow في المشروع وكانت النتائج ممتازة"
        scores = coherence_reward([text])
        assert scores[0] > 0.0

    def test_garbage_symbols_penalized(self):
        text = "###$$$%%%^^^&&&***!!!@@@" * 5
        scores = coherence_reward([text])
        assert scores[0] < 0.5

    def test_score_always_0_to_1(self):
        test_cases = [
            "نص عربي",
            "English text",
            "مختلط mixed",
            "!!!???###",
            "a",
            "ك" * 1000,
        ]
        for text in test_cases:
            scores = coherence_reward([text])
            assert 0.0 <= scores[0] <= 1.0, f"Failed for: {text[:20]}"

    def test_multiple_completions(self):
        completions = ["نص عربي جيد", "Good English", "###garbage###"]
        scores = coherence_reward(completions)
        assert len(scores) == 3
        assert scores[0] > scores[2]  # Arabic > garbage


# ═══════════════════════════════════════════
# Integration: Combined Reward Scoring
# ═══════════════════════════════════════════

class TestCombinedRewards:
    """Test that reward functions work correctly together (as GRPO uses them)."""

    def test_all_rewards_same_length(self):
        completions = ["إجابة أولى.", "إجابة ثانية.", "إجابة ثالثة."]
        answers = ["أولى", "ثانية", "ثالثة"]

        acc = accuracy_reward(completions, answer=answers)
        fmt = format_reward(completions)
        coh = coherence_reward(completions)

        assert len(acc) == len(fmt) == len(coh) == 3

    def test_good_response_scores_well_on_all(self):
        text = "القلب هو العضو المسؤول عن ضخ الدم في جسم الإنسان."
        acc = accuracy_reward([text], answer=["القلب"])
        fmt = format_reward([text])
        coh = coherence_reward([text])

        assert acc[0] == 1.0  # Contains answer
        assert fmt[0] >= 0.5  # Good format
        assert coh[0] >= 0.7  # Pure Arabic

    def test_bad_response_scores_poorly(self):
        text = "###"
        acc = accuracy_reward([text], answer=["القلب"])
        fmt = format_reward([text])
        coh = coherence_reward([text])

        assert acc[0] == 0.0
        assert fmt[0] < 0.5
        assert coh[0] < 0.5


# ═══════════════════════════════════════════
# Reasoning Tags (Nemotron XML + DeepSeek Think)
# ═══════════════════════════════════════════

class TestCheckReasoningTags:
    """Test _check_reasoning_tags helper for both XML and think formats."""

    def test_valid_xml_format(self):
        from src.training.rewards import _check_reasoning_tags
        text = "<reasoning>تحليل خطوة بخطوة</reasoning><answer>الإجابة النهائية</answer>"
        assert _check_reasoning_tags(text) == "valid"

    def test_valid_think_format(self):
        from src.training.rewards import _check_reasoning_tags
        text = "<think>أفكر في الموضوع</think>الإجابة هي كذا."
        assert _check_reasoning_tags(text) == "valid"

    def test_malformed_xml_missing_close(self):
        from src.training.rewards import _check_reasoning_tags
        text = "<reasoning>بداية بدون نهاية<answer>إجابة</answer>"
        assert _check_reasoning_tags(text) == "malformed"

    def test_malformed_think_no_close(self):
        from src.training.rewards import _check_reasoning_tags
        text = "<think>تفكير بدون إغلاق"
        assert _check_reasoning_tags(text) == "malformed"

    def test_no_tags_returns_none(self):
        from src.training.rewards import _check_reasoning_tags
        text = "نص عادي بدون أي علامات"
        assert _check_reasoning_tags(text) == "none"

    def test_reversed_xml_order_malformed(self):
        from src.training.rewards import _check_reasoning_tags
        text = "<answer>إجابة</answer><reasoning>تحليل</reasoning>"
        assert _check_reasoning_tags(text) == "malformed"


class TestXmlFormatReward:
    """Test the strict Nemotron-style XML format reward."""

    def test_perfect_xml(self):
        from src.training.rewards import xml_format_reward
        text = "<reasoning>القلب عضو عضلي يضخ الدم</reasoning><answer>القلب</answer>"
        scores = xml_format_reward([text])
        assert scores[0] == 1.0

    def test_empty_reasoning(self):
        from src.training.rewards import xml_format_reward
        text = "<reasoning></reasoning><answer>القلب</answer>"
        scores = xml_format_reward([text])
        assert scores[0] == 0.5  # Has answer but empty reasoning

    def test_no_tags(self):
        from src.training.rewards import xml_format_reward
        text = "إجابة بدون أي تنسيق"
        scores = xml_format_reward([text])
        assert scores[0] == 0.0

    def test_answer_only_no_reasoning(self):
        from src.training.rewards import xml_format_reward
        text = "<answer>القلب</answer>"
        scores = xml_format_reward([text])
        assert scores[0] == 0.3

    def test_multiple_completions(self):
        from src.training.rewards import xml_format_reward
        completions = [
            "<reasoning>تحليل</reasoning><answer>جواب</answer>",
            "بدون تنسيق",
        ]
        scores = xml_format_reward(completions)
        assert len(scores) == 2
        assert scores[0] > scores[1]
