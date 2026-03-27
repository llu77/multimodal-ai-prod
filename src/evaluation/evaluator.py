"""
Evaluation Pipeline — RAG Retrieval, Generation Quality, End-to-End.
خط أنابيب التقييم — جودة الاسترجاع، جودة التوليد، شامل

Three evaluators:
1. RetrievalEvaluator  — Is RAG finding the right documents?
2. GenerationEvaluator — Is the model generating correct answers?
3. E2EEvaluator        — Full pipeline: query → retrieve → generate → score

Usage:
    python -m src.evaluation.evaluator --config config/config.yaml --eval-file data/eval_set.jsonl
"""
import json
import time
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field
from loguru import logger

from src.evaluation.metrics import (
    precision_at_k, recall_at_k, mrr, ndcg_at_k, hit_rate,
    rouge_l, bleu_simple, exact_match, f1_token,
    faithfulness, answer_relevance,
)


# ═══════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════

@dataclass
class RetrievalSample:
    """Single retrieval evaluation sample."""
    query: str
    relevant_doc_ids: list[str]  # Ground truth: which docs SHOULD be retrieved
    metadata: dict = field(default_factory=dict)


@dataclass
class GenerationSample:
    """Single generation evaluation sample."""
    query: str
    reference_answer: str        # Ground truth: expected answer
    context: str = ""            # RAG context provided to model
    metadata: dict = field(default_factory=dict)


@dataclass
class E2ESample:
    """End-to-end evaluation sample — combines retrieval + generation."""
    query: str
    reference_answer: str
    relevant_doc_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Evaluation result with per-sample and aggregate scores."""
    name: str
    aggregate: dict = field(default_factory=dict)
    per_sample: list[dict] = field(default_factory=list)
    total_samples: int = 0
    runtime_seconds: float = 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"\n{'=' * 60}", f"  {self.name}", f"{'=' * 60}"]
        lines.append(f"  Samples: {self.total_samples}")
        lines.append(f"  Runtime: {self.runtime_seconds:.1f}s")
        lines.append("")
        for metric, value in sorted(self.aggregate.items()):
            bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            lines.append(f"  {metric:.<30s} {value:.4f}  {bar}")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "aggregate": self.aggregate,
            "total_samples": self.total_samples,
            "runtime_seconds": self.runtime_seconds,
            "per_sample": self.per_sample,
        }


# ═══════════════════════════════════════════
# Retrieval Evaluator
# ═══════════════════════════════════════════

class RetrievalEvaluator:
    """
    Evaluate RAG retrieval quality.
    تقييم جودة استرجاع RAG

    Answers: "Is RAG finding the right documents?"

    Metrics:
    - Precision@K: Of top-K retrieved, how many are relevant?
    - Recall@K: Of all relevant docs, how many did we find?
    - MRR: How high is the first relevant result?
    - NDCG@K: How good is the ranking order?
    - Hit Rate: Did we find at least one relevant doc?
    """

    def __init__(self, k: int = 5):
        self.k = k

    def evaluate(
        self,
        samples: list[RetrievalSample],
        retrieve_fn: Callable[[str], list[str]],
    ) -> EvalResult:
        """
        Run retrieval evaluation.

        Args:
            samples: List of (query, relevant_doc_ids)
            retrieve_fn: Function(query) → list[doc_ids] (your RAG retrieve)

        Returns:
            EvalResult with aggregate and per-sample scores
        """
        start = time.time()
        per_sample = []
        agg = {
            f"precision@{self.k}": 0.0,
            f"recall@{self.k}": 0.0,
            "mrr": 0.0,
            f"ndcg@{self.k}": 0.0,
            "hit_rate": 0.0,
        }

        for sample in samples:
            retrieved = retrieve_fn(sample.query)
            relevant = sample.relevant_doc_ids

            scores = {
                f"precision@{self.k}": precision_at_k(retrieved, relevant, self.k),
                f"recall@{self.k}": recall_at_k(retrieved, relevant, self.k),
                "mrr": mrr(retrieved, relevant),
                f"ndcg@{self.k}": ndcg_at_k(retrieved, relevant, self.k),
                "hit_rate": hit_rate(retrieved, relevant),
            }

            per_sample.append({
                "query": sample.query,
                "retrieved": retrieved[:self.k],
                "relevant": relevant,
                **scores,
            })

            for key in agg:
                agg[key] += scores[key]

        n = max(len(samples), 1)
        for key in agg:
            agg[key] = round(agg[key] / n, 4)

        return EvalResult(
            name="Retrieval Quality / جودة الاسترجاع",
            aggregate=agg,
            per_sample=per_sample,
            total_samples=len(samples),
            runtime_seconds=round(time.time() - start, 2),
        )


# ═══════════════════════════════════════════
# Generation Evaluator
# ═══════════════════════════════════════════

class GenerationEvaluator:
    """
    Evaluate generation quality against reference answers.
    تقييم جودة التوليد مقابل الإجابات المرجعية

    Answers: "Is the model generating correct, faithful answers?"

    Metrics:
    - Exact Match: Perfect text match (strict)
    - F1 Token: Token overlap (lenient)
    - ROUGE-L: Subsequence similarity (structure)
    - BLEU: N-gram precision (fluency)
    - Faithfulness: Is the answer grounded in provided context?
    - Answer Relevance: Does the answer address the question?
    """

    def evaluate(
        self,
        samples: list[GenerationSample],
        generate_fn: Callable[[str, str], str],
    ) -> EvalResult:
        """
        Run generation evaluation.

        Args:
            samples: List of (query, reference_answer, context)
            generate_fn: Function(query, context) → generated_answer

        Returns:
            EvalResult with aggregate and per-sample scores
        """
        start = time.time()
        per_sample = []
        agg = {
            "exact_match": 0.0,
            "f1_token": 0.0,
            "rouge_l": 0.0,
            "bleu": 0.0,
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
        }

        for sample in samples:
            try:
                prediction = generate_fn(sample.query, sample.context)
            except Exception as e:
                logger.warning(f"Generation failed for '{sample.query[:50]}': {e}")
                prediction = ""

            scores = {
                "exact_match": exact_match(prediction, sample.reference_answer),
                "f1_token": f1_token(prediction, sample.reference_answer),
                "rouge_l": rouge_l(prediction, sample.reference_answer),
                "bleu": bleu_simple(prediction, sample.reference_answer),
                "faithfulness": faithfulness(prediction, sample.context) if sample.context else 0.0,
                "answer_relevance": answer_relevance(prediction, sample.query),
            }

            per_sample.append({
                "query": sample.query,
                "reference": sample.reference_answer[:200],
                "prediction": prediction[:200],
                **{k: round(v, 4) for k, v in scores.items()},
            })

            for key in agg:
                agg[key] += scores[key]

        n = max(len(samples), 1)
        for key in agg:
            agg[key] = round(agg[key] / n, 4)

        return EvalResult(
            name="Generation Quality / جودة التوليد",
            aggregate=agg,
            per_sample=per_sample,
            total_samples=len(samples),
            runtime_seconds=round(time.time() - start, 2),
        )


# ═══════════════════════════════════════════
# End-to-End Evaluator
# ═══════════════════════════════════════════

class E2EEvaluator:
    """
    End-to-end evaluation: query → retrieve → generate → score.
    تقييم شامل: استعلام → استرجاع → توليد → تقييم

    Combines retrieval and generation evaluation in one pass.
    Answers: "Does the full pipeline produce correct answers?"
    """

    def __init__(self, k: int = 5):
        self.retrieval_eval = RetrievalEvaluator(k=k)
        self.generation_eval = GenerationEvaluator()

    def evaluate(
        self,
        samples: list[E2ESample],
        retrieve_fn: Callable[[str], tuple[list[str], str]],
        generate_fn: Callable[[str, str], str],
    ) -> EvalResult:
        """
        Run end-to-end evaluation.

        Args:
            samples: List of E2ESamples
            retrieve_fn: Function(query) → (doc_ids, context_string)
            generate_fn: Function(query, context) → generated_answer

        Returns:
            Combined EvalResult
        """
        start = time.time()
        per_sample = []

        # Aggregates for both retrieval and generation
        agg = {}

        retrieval_samples = []
        generation_samples = []

        for sample in samples:
            # Step 1: Retrieve
            try:
                doc_ids, context = retrieve_fn(sample.query)
            except Exception as e:
                logger.warning(f"Retrieval failed for '{sample.query[:50]}': {e}")
                doc_ids, context = [], ""

            # Step 2: Generate
            try:
                prediction = generate_fn(sample.query, context)
            except Exception as e:
                logger.warning(f"Generation failed for '{sample.query[:50]}': {e}")
                prediction = ""

            # Step 3: Score retrieval
            ret_scores = {}
            if sample.relevant_doc_ids:
                k = self.retrieval_eval.k
                ret_scores = {
                    f"ret_precision@{k}": precision_at_k(doc_ids, sample.relevant_doc_ids, k),
                    f"ret_recall@{k}": recall_at_k(doc_ids, sample.relevant_doc_ids, k),
                    "ret_mrr": mrr(doc_ids, sample.relevant_doc_ids),
                    "ret_hit_rate": hit_rate(doc_ids, sample.relevant_doc_ids),
                }

            # Step 4: Score generation
            gen_scores = {
                "gen_f1_token": f1_token(prediction, sample.reference_answer),
                "gen_rouge_l": rouge_l(prediction, sample.reference_answer),
                "gen_faithfulness": faithfulness(prediction, context) if context else 0.0,
                "gen_answer_relevance": answer_relevance(prediction, sample.query),
            }

            combined = {**ret_scores, **gen_scores}
            per_sample.append({
                "query": sample.query,
                "reference": sample.reference_answer[:200],
                "prediction": prediction[:200],
                "num_docs_retrieved": len(doc_ids),
                **{k: round(v, 4) for k, v in combined.items()},
            })

            for k, v in combined.items():
                agg[k] = agg.get(k, 0.0) + v

        # Average
        n = max(len(samples), 1)
        for k in agg:
            agg[k] = round(agg[k] / n, 4)

        return EvalResult(
            name="End-to-End Pipeline / التقييم الشامل",
            aggregate=agg,
            per_sample=per_sample,
            total_samples=len(samples),
            runtime_seconds=round(time.time() - start, 2),
        )


# ═══════════════════════════════════════════
# Eval Dataset Loader
# ═══════════════════════════════════════════

def load_eval_dataset(path: str) -> list[E2ESample]:
    """
    Load evaluation dataset from JSONL.
    تحميل مجموعة بيانات التقييم من JSONL

    Expected format per line:
    {
        "query": "ما هو القلب؟",
        "reference_answer": "القلب هو عضو عضلي يضخ الدم",
        "relevant_doc_ids": ["doc_001", "doc_002"],  // optional
    }
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                samples.append(E2ESample(
                    query=data["query"],
                    reference_answer=data["reference_answer"],
                    relevant_doc_ids=data.get("relevant_doc_ids", []),
                    metadata=data.get("metadata", {}),
                ))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping line {line_num}: {e}")

    logger.info(f"Loaded {len(samples)} eval samples from {path}")
    return samples


def create_sample_eval_dataset(output_path: str) -> str:
    """
    Create a sample evaluation dataset.
    إنشاء مجموعة بيانات تقييم نموذجية
    """
    samples = [
        {
            "query": "ما هو العضو المسؤول عن ضخ الدم في جسم الإنسان؟",
            "reference_answer": "القلب هو العضو المسؤول عن ضخ الدم في جسم الإنسان. وهو عضو عضلي يقع في الصدر ويعمل كمضخة لتوزيع الدم المحمّل بالأكسجين إلى جميع أنحاء الجسم.",
            "relevant_doc_ids": ["cardiology_001", "anatomy_heart"],
        },
        {
            "query": "ما هي أعراض نقص فيتامين D؟",
            "reference_answer": "أعراض نقص فيتامين D تشمل التعب المزمن وآلام العظام والمفاصل وضعف العضلات وتساقط الشعر والاكتئاب وضعف المناعة.",
            "relevant_doc_ids": ["nutrition_vitd", "deficiency_symptoms"],
        },
        {
            "query": "كم عدد عظام الجسم البشري البالغ؟",
            "reference_answer": "يحتوي الجسم البشري البالغ على 206 عظمة.",
            "relevant_doc_ids": ["anatomy_skeletal"],
        },
        {
            "query": "ما الفرق بين الشريان والوريد؟",
            "reference_answer": "الشريان ينقل الدم المؤكسج من القلب إلى أنحاء الجسم ويتميز بجدران سميكة ومرنة. الوريد ينقل الدم غير المؤكسج من الجسم إلى القلب ويتميز بجدران أرق ووجود صمامات.",
            "relevant_doc_ids": ["cardiovascular_001", "anatomy_vessels"],
        },
        {
            "query": "ما هو الأنسولين وما وظيفته؟",
            "reference_answer": "الأنسولين هو هرمون يُفرَز من خلايا بيتا في البنكرياس. وظيفته الرئيسية هي تنظيم مستوى السكر (الجلوكوز) في الدم عن طريق تسهيل دخول الجلوكوز إلى الخلايا لاستخدامه كطاقة.",
            "relevant_doc_ids": ["endocrine_insulin", "diabetes_basics"],
        },
        {
            "query": "What is the function of the liver?",
            "reference_answer": "The liver performs over 500 vital functions including detoxification, protein synthesis, bile production, and metabolism of fats, proteins, and carbohydrates.",
            "relevant_doc_ids": ["anatomy_liver", "hepatology_001"],
        },
        {
            "query": "ما هي مكونات الدم الرئيسية؟",
            "reference_answer": "مكونات الدم الرئيسية هي: كريات الدم الحمراء التي تنقل الأكسجين، كريات الدم البيضاء التي تحارب العدوى، الصفائح الدموية التي تساعد في تخثر الدم، والبلازما وهي السائل الذي يحمل هذه المكونات.",
            "relevant_doc_ids": ["hematology_001", "blood_components"],
        },
        {
            "query": "كيف يعمل الجهاز التنفسي؟",
            "reference_answer": "الجهاز التنفسي يعمل عبر استنشاق الهواء من الأنف أو الفم، مروراً بالقصبة الهوائية إلى الرئتين حيث يتم تبادل الغازات في الحويصلات الهوائية. يدخل الأكسجين إلى الدم ويخرج ثاني أكسيد الكربون.",
            "relevant_doc_ids": ["pulmonology_001", "respiratory_system"],
        },
    ]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"Sample eval dataset created: {path} ({len(samples)} samples)")
    return str(path)


# ═══════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════

def save_report(results: list[EvalResult], output_path: str) -> None:
    """Save evaluation report as JSON + human-readable summary."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # JSON report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluations": [r.to_dict() for r in results],
    }
    json_path = path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON report saved: {json_path}")

    # Human-readable
    txt_path = path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Multimodal AI Evaluation Report\n")
        f.write(f"تقرير تقييم نظام الذكاء الاصطناعي\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for r in results:
            f.write(r.summary())
            f.write("\n")

        # Worst samples (for debugging)
        f.write(f"\n{'=' * 60}\n")
        f.write(f"  Worst Performing Samples / أسوأ العينات أداءً\n")
        f.write(f"{'=' * 60}\n")
        for r in results:
            if not r.per_sample:
                continue
            # Find samples with lowest average score
            for sample in r.per_sample:
                scores = [v for k, v in sample.items() if isinstance(v, float)]
                if scores:
                    sample["_avg_score"] = sum(scores) / len(scores)
            worst = sorted(r.per_sample, key=lambda x: x.get("_avg_score", 0))[:3]
            for w in worst:
                f.write(f"\n  Query: {w.get('query', '')[:80]}")
                f.write(f"\n  Avg Score: {w.get('_avg_score', 0):.4f}\n")

    logger.info(f"Text report saved: {txt_path}")


# ═══════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluation Pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--eval-file", type=str, help="Path to eval JSONL")
    parser.add_argument("--create-sample", action="store_true", help="Create sample eval dataset")
    parser.add_argument("--output", default="./reports/eval_report", help="Output report path")
    parser.add_argument("--mode", choices=["retrieval", "generation", "e2e", "all"], default="all")
    args = parser.parse_args()

    if args.create_sample:
        create_sample_eval_dataset("./data/eval/eval_set.jsonl")
        print("Sample eval dataset created. Edit it with your own Q&A pairs.")
    elif args.eval_file:
        samples = load_eval_dataset(args.eval_file)
        print(f"Loaded {len(samples)} samples. Connect to your RAG/model to run evaluation.")
        print("See README for integration examples.")
    else:
        parser.print_help()
