"""
Tests for evaluation metrics and evaluators.
اختبارات مقاييس التقييم والمُقيّمات

All tests run WITHOUT GPU, models, or external dependencies.
"""
import pytest

from src.evaluation.metrics import (
    precision_at_k, recall_at_k, mrr, ndcg_at_k, hit_rate,
    rouge_l, bleu_simple, exact_match, f1_token,
    faithfulness, answer_relevance,
)
from src.evaluation.evaluator import (
    RetrievalEvaluator, GenerationEvaluator, E2EEvaluator,
    RetrievalSample, GenerationSample, E2ESample,
    EvalResult, load_eval_dataset, create_sample_eval_dataset,
)


# ═══════════════════════════════════════════
# Retrieval Metrics
# ═══════════════════════════════════════════

class TestPrecisionAtK:

    def test_perfect(self):
        assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["x", "y", "z"], ["a", "b"], k=3) == 0.0

    def test_partial(self):
        assert precision_at_k(["a", "x", "b"], ["a", "b"], k=3) == pytest.approx(2 / 3)

    def test_k_larger_than_retrieved(self):
        assert precision_at_k(["a"], ["a", "b"], k=5) == pytest.approx(1 / 5)

    def test_empty_inputs(self):
        assert precision_at_k([], ["a"], k=3) == 0.0
        assert precision_at_k(["a"], [], k=3) == 0.0


class TestRecallAtK:

    def test_perfect(self):
        assert recall_at_k(["a", "b"], ["a", "b"], k=5) == 1.0

    def test_partial(self):
        assert recall_at_k(["a", "x"], ["a", "b"], k=5) == 0.5

    def test_none_found(self):
        assert recall_at_k(["x", "y"], ["a", "b"], k=5) == 0.0

    def test_empty(self):
        assert recall_at_k([], ["a"], k=3) == 0.0


class TestMRR:

    def test_first_is_relevant(self):
        assert mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_second_is_relevant(self):
        assert mrr(["x", "a", "c"], ["a"]) == 0.5

    def test_third_is_relevant(self):
        assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_none_relevant(self):
        assert mrr(["x", "y", "z"], ["a"]) == 0.0


class TestNDCG:

    def test_perfect_ranking(self):
        assert ndcg_at_k(["a", "b", "x"], ["a", "b"], k=3) == pytest.approx(1.0)

    def test_reversed_ranking(self):
        # Relevant docs at bottom → lower score than at top
        score_top = ndcg_at_k(["a", "x", "y"], ["a"], k=3)
        score_bottom = ndcg_at_k(["x", "y", "a"], ["a"], k=3)
        assert score_top > score_bottom

    def test_no_relevant(self):
        assert ndcg_at_k(["x", "y"], ["a"], k=2) == 0.0

    def test_empty(self):
        assert ndcg_at_k([], ["a"], k=3) == 0.0


class TestHitRate:

    def test_hit(self):
        assert hit_rate(["x", "a"], ["a"]) == 1.0

    def test_miss(self):
        assert hit_rate(["x", "y"], ["a"]) == 0.0

    def test_empty(self):
        assert hit_rate([], ["a"]) == 0.0


# ═══════════════════════════════════════════
# Generation Metrics
# ═══════════════════════════════════════════

class TestExactMatch:

    def test_identical(self):
        assert exact_match("القلب", "القلب") == 1.0

    def test_different(self):
        assert exact_match("القلب", "الكبد") == 0.0

    def test_case_insensitive(self):
        assert exact_match("Hello World", "hello world") == 1.0

    def test_ignores_diacritics(self):
        assert exact_match("كِتَاب", "كتاب") == 1.0


class TestF1Token:

    def test_perfect(self):
        assert f1_token("القلب يضخ الدم", "القلب يضخ الدم") == 1.0

    def test_partial_overlap(self):
        score = f1_token("القلب عضو مهم", "القلب يضخ الدم")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        assert f1_token("الطقس جميل", "القلب يضخ الدم") == 0.0

    def test_empty(self):
        assert f1_token("", "something") == 0.0
        assert f1_token("something", "") == 0.0


class TestRougeL:

    def test_identical(self):
        assert rouge_l("القلب يضخ الدم", "القلب يضخ الدم") == 1.0

    def test_subsequence(self):
        score = rouge_l(
            "القلب هو العضو الذي يضخ الدم",
            "القلب يضخ الدم"
        )
        assert score > 0.5

    def test_no_overlap(self):
        assert rouge_l("أحمر أخضر أزرق", "واحد اثنان ثلاثة") == 0.0

    def test_empty(self):
        assert rouge_l("", "text") == 0.0


class TestBleuSimple:

    def test_identical(self):
        text = "هذا نص اختبار بسيط للتحقق من المقياس"
        score = bleu_simple(text, text)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_completely_different(self):
        score = bleu_simple("واحد اثنان ثلاثة أربعة", "خمسة ستة سبعة ثمانية")
        assert score == 0.0

    def test_partial(self):
        # BLEU needs sufficient n-gram overlap across all orders (1-4)
        # Short texts often get 0 because 4-grams don't overlap
        score = bleu_simple(
            "القلب هو العضو الذي يضخ الدم في جسم الإنسان ويوزعه على الأعضاء",
            "القلب هو العضو المسؤول عن ضخ الدم في جسم الإنسان وتوزيعه",
        )
        assert 0.0 < score < 1.0

    def test_brevity_penalty(self):
        # Short prediction penalized
        short = bleu_simple("القلب", "القلب هو العضو المسؤول عن ضخ الدم")
        long = bleu_simple("القلب هو العضو المسؤول عن ضخ الدم", "القلب هو العضو المسؤول عن ضخ الدم")
        assert long > short


class TestFaithfulness:

    def test_fully_grounded(self):
        context = "القلب عضو عضلي يضخ الدم إلى جميع أنحاء الجسم"
        answer = "القلب يضخ الدم إلى الجسم"
        score = faithfulness(answer, context)
        assert score > 0.5

    def test_not_grounded(self):
        context = "القلب عضو عضلي"
        answer = "الكبد يفرز الصفراء ويزيل السموم"
        score = faithfulness(answer, context)
        assert score < 0.3

    def test_empty_context(self):
        assert faithfulness("some answer", "") == 0.0

    def test_empty_answer(self):
        assert faithfulness("", "some context") == 0.0


class TestAnswerRelevance:

    def test_relevant(self):
        score = answer_relevance("القلب يضخ الدم", "ما وظيفة القلب")
        assert score > 0.3

    def test_irrelevant(self):
        score = answer_relevance("الطقس مشمس اليوم", "ما وظيفة القلب")
        assert score < 0.3

    def test_empty(self):
        assert answer_relevance("", "query") == 0.0
        assert answer_relevance("answer", "") == 0.0


# ═══════════════════════════════════════════
# Retrieval Evaluator
# ═══════════════════════════════════════════

class TestRetrievalEvaluator:

    def test_perfect_retrieval(self):
        evaluator = RetrievalEvaluator(k=3)
        samples = [
            RetrievalSample(query="q1", relevant_doc_ids=["a", "b"]),
            RetrievalSample(query="q2", relevant_doc_ids=["c"]),
        ]

        def perfect_retrieve(query):
            return {"q1": ["a", "b", "x"], "q2": ["c", "x", "y"]}[query]

        result = evaluator.evaluate(samples, perfect_retrieve)
        assert isinstance(result, EvalResult)
        assert result.total_samples == 2
        assert result.aggregate["hit_rate"] == 1.0
        assert result.aggregate["mrr"] == 1.0

    def test_no_retrieval(self):
        evaluator = RetrievalEvaluator(k=3)
        samples = [RetrievalSample(query="q1", relevant_doc_ids=["a"])]

        result = evaluator.evaluate(samples, lambda q: ["x", "y", "z"])
        assert result.aggregate["hit_rate"] == 0.0
        assert result.aggregate["mrr"] == 0.0

    def test_per_sample_details(self):
        evaluator = RetrievalEvaluator(k=2)
        samples = [RetrievalSample(query="test", relevant_doc_ids=["a"])]
        result = evaluator.evaluate(samples, lambda q: ["a", "b"])
        assert len(result.per_sample) == 1
        assert result.per_sample[0]["query"] == "test"


# ═══════════════════════════════════════════
# Generation Evaluator
# ═══════════════════════════════════════════

class TestGenerationEvaluator:

    def test_perfect_generation(self):
        evaluator = GenerationEvaluator()
        samples = [
            GenerationSample(
                query="ما هو القلب؟",
                reference_answer="القلب عضو يضخ الدم",
                context="القلب عضو عضلي يضخ الدم في الجسم",
            ),
        ]

        def perfect_gen(query, context):
            return "القلب عضو يضخ الدم"

        result = evaluator.evaluate(samples, perfect_gen)
        assert result.aggregate["exact_match"] == 1.0
        assert result.aggregate["f1_token"] == 1.0
        assert result.aggregate["rouge_l"] == 1.0

    def test_wrong_generation(self):
        evaluator = GenerationEvaluator()
        samples = [
            GenerationSample(
                query="ما هو القلب؟",
                reference_answer="القلب عضو يضخ الدم",
            ),
        ]

        result = evaluator.evaluate(samples, lambda q, c: "الطقس جميل")
        assert result.aggregate["exact_match"] == 0.0
        assert result.aggregate["f1_token"] < 0.3

    def test_generation_failure_handled(self):
        evaluator = GenerationEvaluator()
        samples = [
            GenerationSample(query="q", reference_answer="a"),
        ]

        def failing_gen(query, context):
            raise RuntimeError("Model crashed")

        result = evaluator.evaluate(samples, failing_gen)
        assert result.total_samples == 1
        assert result.aggregate["f1_token"] == 0.0  # Failed → empty → 0


# ═══════════════════════════════════════════
# E2E Evaluator
# ═══════════════════════════════════════════

class TestE2EEvaluator:

    def test_full_pipeline(self):
        evaluator = E2EEvaluator(k=3)
        samples = [
            E2ESample(
                query="ما وظيفة القلب؟",
                reference_answer="القلب يضخ الدم",
                relevant_doc_ids=["doc1"],
            ),
        ]

        def retrieve(query):
            return (["doc1", "doc2"], "القلب عضو يضخ الدم في الجسم")

        def generate(query, context):
            return "القلب يضخ الدم في جسم الإنسان"

        result = evaluator.evaluate(samples, retrieve, generate)
        assert result.total_samples == 1
        assert result.aggregate.get("ret_hit_rate", 0) == 1.0
        assert result.aggregate.get("gen_f1_token", 0) > 0.5

    def test_no_relevant_docs(self):
        evaluator = E2EEvaluator(k=3)
        samples = [
            E2ESample(query="q", reference_answer="a", relevant_doc_ids=[]),
        ]

        result = evaluator.evaluate(
            samples,
            lambda q: (["x"], "context"),
            lambda q, c: "answer",
        )
        # No relevant_doc_ids → no retrieval metrics
        assert "ret_hit_rate" not in result.aggregate


# ═══════════════════════════════════════════
# Eval Dataset I/O
# ═══════════════════════════════════════════

class TestEvalDatasetIO:

    def test_create_sample_dataset(self, tmp_dir):
        path = str(tmp_dir / "eval.jsonl")
        create_sample_eval_dataset(path)
        samples = load_eval_dataset(path)
        assert len(samples) == 8
        assert samples[0].query
        assert samples[0].reference_answer

    def test_load_with_missing_fields(self, tmp_dir):
        path = tmp_dir / "bad.jsonl"
        with open(path, "w") as f:
            f.write('{"query": "q1"}\n')  # Missing reference_answer
            f.write('{"query": "q2", "reference_answer": "a2"}\n')  # Good
        samples = load_eval_dataset(str(path))
        assert len(samples) == 1  # Only the valid one

    def test_load_empty_file(self, tmp_dir):
        path = tmp_dir / "empty.jsonl"
        path.write_text("")
        samples = load_eval_dataset(str(path))
        assert len(samples) == 0


# ═══════════════════════════════════════════
# EvalResult
# ═══════════════════════════════════════════

class TestEvalResult:

    def test_summary_format(self):
        result = EvalResult(
            name="Test",
            aggregate={"metric_a": 0.85, "metric_b": 0.42},
            total_samples=10,
            runtime_seconds=1.5,
        )
        summary = result.summary()
        assert "Test" in summary
        assert "0.85" in summary
        assert "10" in summary

    def test_to_dict(self):
        result = EvalResult(name="Test", aggregate={"f1": 0.9}, total_samples=5)
        d = result.to_dict()
        assert d["name"] == "Test"
        assert d["aggregate"]["f1"] == 0.9
        assert d["total_samples"] == 5
