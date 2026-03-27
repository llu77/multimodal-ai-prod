"""
Evaluation Metrics — Retrieval and Generation quality measurement.
مقاييس التقييم — قياس جودة الاسترجاع والتوليد

All metrics are standalone functions with NO external ML dependencies.
They work with plain strings and lists — testable without GPU.

Retrieval Metrics:
    precision_at_k, recall_at_k, mrr, ndcg, hit_rate

Generation Metrics:
    rouge_l, bleu_simple, exact_match, f1_token, faithfulness
"""
import re
import math
from collections import Counter
from typing import Optional


# ═══════════════════════════════════════════
# Retrieval Metrics
# ═══════════════════════════════════════════

def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """
    Precision@K — fraction of top-K retrieved docs that are relevant.
    الدقة@ك — نسبة المستندات المسترجعة ذات الصلة من أعلى ك

    Perfect = 1.0 (all top-K are relevant)
    Worst = 0.0 (none are relevant)
    """
    if k <= 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """
    Recall@K — fraction of relevant docs found in top-K.
    الاستدعاء@ك — نسبة المستندات ذات الصلة الموجودة في أعلى ك

    Perfect = 1.0 (all relevant docs retrieved)
    Worst = 0.0 (none found)
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """
    Mean Reciprocal Rank — 1/rank of first relevant doc.
    متوسط الرتبة المتبادلة — 1/ترتيب أول مستند ذي صلة

    Perfect = 1.0 (first result is relevant)
    Worst = 0.0 (no relevant results)
    """
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """
    Normalized Discounted Cumulative Gain@K.
    الكسب التراكمي المخصوم والمُطبَّع

    Measures ranking quality — rewards relevant docs appearing earlier.
    """
    if not relevant_ids or k <= 0:
        return 0.0

    relevant_set = set(relevant_ids)

    # DCG: sum of 1/log2(rank+1) for relevant docs
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # +2 because rank starts at 1

    # Ideal DCG: all relevant docs at top positions
    ideal_relevant = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_relevant))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_rate(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """
    Hit Rate — 1 if ANY relevant doc is in results, else 0.
    معدل الإصابة — 1 إذا وُجد أي مستند ذي صلة، وإلا 0
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0
    return 1.0 if set(retrieved_ids) & set(relevant_ids) else 0.0


# ═══════════════════════════════════════════
# Generation Metrics
# ═══════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for metrics."""
    # Remove Arabic diacritics for fair comparison
    text = re.sub(r'[\u064B-\u065F\u0610-\u061A\u0670]', '', text)
    tokens = re.findall(r'\w+', text.lower())
    return tokens


def exact_match(prediction: str, reference: str) -> float:
    """
    Exact Match — 1.0 if normalized texts are identical.
    المطابقة التامة — 1.0 إذا تطابق النصان بعد التطبيع
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    return 1.0 if pred_tokens == ref_tokens else 0.0


def f1_token(prediction: str, reference: str) -> float:
    """
    Token-level F1 — harmonic mean of token precision and recall.
    F1 على مستوى التوكنات — المتوسط التوافقي للدقة والاستدعاء

    Better than exact_match for free-form answers.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """
    ROUGE-L — Longest Common Subsequence based metric.
    ROUGE-L — مقياس مبني على أطول تتابع فرعي مشترك

    Captures sentence-level structure similarity.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS via dynamic programming
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    if lcs_length == 0:
        return 0.0

    precision = lcs_length / m
    recall = lcs_length / n
    return 2 * precision * recall / (precision + recall)


def bleu_simple(prediction: str, reference: str, max_n: int = 4) -> float:
    """
    Simplified BLEU — n-gram precision with brevity penalty.
    BLEU مبسّط — دقة n-gram مع عقوبة الإيجاز

    Uses uniform weights across n-gram orders.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    bp = 1.0
    if len(pred_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1))

    # N-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(
            tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)
        )

        clipped = sum(min(count, ref_ngrams[ng]) for ng, count in pred_ngrams.items())
        total = max(sum(pred_ngrams.values()), 1)
        precisions.append(clipped / total if total > 0 else 0.0)

    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / len(precisions)
    return bp * math.exp(log_avg)


def faithfulness(
    answer: str,
    context: str,
    threshold: float = 0.3,
) -> float:
    """
    Faithfulness — how much of the answer is grounded in the context.
    الأمانة — مدى استناد الإجابة إلى السياق المقدم

    Measures token overlap between answer and RAG context.
    High score = answer sticks to provided context.
    Low score = answer may contain hallucinations.

    This is a simplified version. For production, use NLI-based checking.
    """
    if not answer or not context:
        return 0.0

    answer_tokens = set(_tokenize(answer))
    context_tokens = set(_tokenize(context))

    if not answer_tokens:
        return 0.0

    # What fraction of answer tokens appear in the context?
    grounded = len(answer_tokens & context_tokens)
    # Exclude common stop words from the count
    stop_words = _arabic_stop_words() | _english_stop_words()
    meaningful_answer = answer_tokens - stop_words
    meaningful_grounded = len((answer_tokens & context_tokens) - stop_words)

    if not meaningful_answer:
        return 1.0  # Only stop words — trivially faithful

    return meaningful_grounded / len(meaningful_answer)


def answer_relevance(answer: str, query: str) -> float:
    """
    Answer Relevance — token overlap between answer and query.
    صلة الإجابة — تداخل التوكنات بين الإجابة والسؤال

    High = answer addresses the question terms.
    Low = answer is off-topic.
    """
    if not answer or not query:
        return 0.0

    answer_tokens = set(_tokenize(answer))
    query_tokens = set(_tokenize(query)) - _arabic_stop_words() - _english_stop_words()

    if not query_tokens:
        return 1.0

    overlap = len(answer_tokens & query_tokens)
    return overlap / len(query_tokens)


# ═══════════════════════════════════════════
# Stop Words (minimal sets for metric calculation)
# ═══════════════════════════════════════════

def _arabic_stop_words() -> set:
    return {
        "في", "من", "على", "إلى", "عن", "مع", "هو", "هي", "هذا", "هذه",
        "ذلك", "تلك", "التي", "الذي", "التي", "ما", "لا", "لم", "لن",
        "قد", "كان", "كانت", "يكون", "أن", "إن", "أو", "و", "ثم", "بل",
        "هل", "كل", "بعض", "غير", "بين", "حتى", "عند", "منذ", "خلال",
        "أي", "أيضا", "فقط", "مثل", "بعد", "قبل", "فوق", "تحت",
    }


def _english_stop_words() -> set:
    return {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "but", "and", "or",
        "not", "no", "nor", "so", "if", "then", "than", "too", "very",
        "just", "about", "up", "out", "that", "this", "it", "its", "i",
    }
