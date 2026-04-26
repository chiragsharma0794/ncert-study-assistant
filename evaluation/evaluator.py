"""
evaluation/evaluator.py -- Evaluation framework for the NCERT RAG pipeline.

Scores each question on three criteria:
  - correctness   : does the answer contain the expected key terms?
  - grounding     : does the answer stay within textbook vocabulary?
  - refusal_correct: for out-of-scope questions, did the model refuse?

Usage:
    from evaluator import evaluate_pipeline, print_report
    results = evaluate_pipeline(questions, retriever, generator)
    print_report(results)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


# =========================================================================
# Scoring functions
# =========================================================================

def score_correctness(answer: str, key_terms: list[str], threshold: int = 2) -> int:
    """Score whether the answer contains enough expected key terms.

    Parameters
    ----------
    answer : str
        The generated answer text.
    key_terms : list[str]
        Expected key terms from the ground-truth answer.
    threshold : int
        Minimum number of key terms that must appear (default: 2).

    Returns
    -------
    int
        1 if at least `threshold` key terms appear, 0 otherwise.
    """
    if not key_terms:
        return 0  # out-of-scope questions have no key terms

    answer_lower = answer.lower()
    matched = sum(1 for term in key_terms if term.lower() in answer_lower)
    return 1 if matched >= threshold else 0


def score_grounding(answer: str, corpus_text: str) -> int:
    """Score whether the answer stays within textbook vocabulary.

    Heuristic: extract capitalized multi-word phrases (potential named
    entities) from the answer and check if they appear in the corpus.
    If more than 2 unknown entities appear, the answer is ungrounded.

    Parameters
    ----------
    answer : str
        The generated answer text.
    corpus_text : str
        All corpus text concatenated (for vocabulary checking).

    Returns
    -------
    int
        1 if grounded (few unknown entities), 0 otherwise.
    """
    # Extract potential named entities (capitalized phrases not at sentence start)
    # Skip very common words and the refusal phrase
    skip_words = {
        "the", "this", "that", "these", "those", "a", "an", "it", "is",
        "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "shall", "should", "may",
        "might", "can", "could", "i", "you", "he", "she", "we", "they",
        "according", "however", "therefore", "also", "not", "no", "yes",
        "based", "class",
    }

    # Find capitalized words in the answer (potential entities)
    entity_pattern = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b")
    entities = entity_pattern.findall(answer)

    # Filter out common words and check against corpus
    corpus_lower = corpus_text.lower()
    unknown_count = 0
    for entity in entities:
        if entity.lower() in skip_words:
            continue
        if entity.lower() not in corpus_lower:
            unknown_count += 1

    return 1 if unknown_count <= 2 else 0


def score_refusal(answer: str, in_scope: bool) -> int:
    """Score whether the refusal behavior is correct.

    Parameters
    ----------
    answer : str
        The generated answer text.
    in_scope : bool
        Whether the question is within the textbook's scope.

    Returns
    -------
    int
        1 if behavior is correct:
          - in_scope=True  and model answered (did not refuse) -> 1
          - in_scope=False and model refused -> 1
          - otherwise -> 0
    """
    refused = "could not find" in answer.lower()

    if in_scope:
        return 1 if not refused else 0   # should NOT refuse
    else:
        return 1 if refused else 0        # SHOULD refuse


# =========================================================================
# Main evaluation runner
# =========================================================================

def evaluate_pipeline(
    questions: list[dict],
    retriever,
    generator,
    corpus_text: str,
    top_k: int = 5,
    dry_run: bool = False,
) -> list[dict]:
    """Run the full evaluation pipeline on all questions.

    Parameters
    ----------
    questions : list[dict]
        Evaluation questions with {id, question, key_terms, type, in_scope}.
    retriever : BM25Retriever
        The retriever to use.
    generator : GroundedGenerator or None
        The generator.  If None, dry_run must be True.
    corpus_text : str
        Full corpus text for grounding checks.
    top_k : int
        Number of chunks to retrieve per question.
    dry_run : bool
        If True, skip generation and score as 0/0/0.

    Returns
    -------
    list[dict]
        Per-question results with scores and metadata.
    """
    results = []

    for q in questions:
        # Retrieve
        retrieved = retriever.retrieve(q["question"], top_k=top_k)
        top_scores = [r["score"] for r in retrieved]

        if dry_run or generator is None:
            answer_text = "[DRY-RUN]"
            sources = []
            refused_flag = False
        else:
            gen_result = generator.generate(q["question"], retrieved)
            answer_text = gen_result["answer"]
            sources = gen_result["sources"]
            refused_flag = gen_result["refused"]

        # Score
        correctness = score_correctness(
            answer_text,
            q.get("key_terms", []),
        )
        grounding = score_grounding(answer_text, corpus_text)
        refusal = score_refusal(answer_text, q["in_scope"])

        results.append({
            "id":               q["id"],
            "question":         q["question"],
            "type":             q["type"],
            "in_scope":         q["in_scope"],
            "answer":           answer_text,
            "sources":          sources,
            "refused":          refused_flag,
            "retrieval_scores": top_scores[:3],
            "correctness":      correctness,
            "grounding":        grounding,
            "refusal_correct":  refusal,
        })

    return results


# =========================================================================
# Aggregate metrics
# =========================================================================

def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate evaluation metrics.

    Returns
    -------
    dict
        Metrics broken down by overall and by question type.
    """
    def _avg(vals):
        return sum(vals) / len(vals) if vals else 0.0

    # Overall
    all_correct = [r["correctness"] for r in results if r["in_scope"]]
    all_ground  = [r["grounding"]   for r in results if r["in_scope"]]
    all_refusal = [r["refusal_correct"] for r in results]

    # By type
    by_type = {}
    for qtype in ["factual", "paraphrased", "out_of_scope"]:
        typed = [r for r in results if r["type"] == qtype]
        if not typed:
            continue
        by_type[qtype] = {
            "count":     len(typed),
            "correctness": _avg([r["correctness"] for r in typed if r["in_scope"]]),
            "grounding":   _avg([r["grounding"]   for r in typed if r["in_scope"]]),
            "refusal":     _avg([r["refusal_correct"] for r in typed]),
        }

    return {
        "total_questions":    len(results),
        "correctness_pct":    _avg(all_correct) * 100,
        "grounding_pct":      _avg(all_ground) * 100,
        "refusal_accuracy_pct": _avg(all_refusal) * 100,
        "by_type":            by_type,
    }


# =========================================================================
# Pretty printing
# =========================================================================

def print_report(results: list[dict], metrics: Optional[dict] = None) -> None:
    """Print a formatted evaluation report."""
    if metrics is None:
        metrics = compute_metrics(results)

    print("=" * 80)
    print("  EVALUATION RESULTS")
    print("=" * 80)

    print(f"\n  {'ID':<6} {'Type':<14} {'InScope':>7} {'Correct':>8} "
          f"{'Grounded':>9} {'Refused':>8} {'RefOK':>6}  Question")
    print(f"  {'-'*6} {'-'*14} {'-'*7} {'-'*8} {'-'*9} {'-'*8} {'-'*6}  {'-'*30}")

    for r in results:
        q_preview = r["question"][:30]
        print(f"  {r['id']:<6} {r['type']:<14} {str(r['in_scope']):>7} "
              f"{r['correctness']:>8} {r['grounding']:>9} "
              f"{str(r['refused']):>8} {r['refusal_correct']:>6}  {q_preview}...")

    print(f"\n{'='*80}")
    print(f"  AGGREGATE METRICS")
    print(f"{'='*80}")
    print(f"\n  Correctness (in-scope)  : {metrics['correctness_pct']:.1f}%")
    print(f"  Grounding (in-scope)    : {metrics['grounding_pct']:.1f}%")
    print(f"  Refusal accuracy (all)  : {metrics['refusal_accuracy_pct']:.1f}%")

    print(f"\n  {'Type':<14} {'N':>4} {'Correct':>9} {'Grounded':>9} {'Refusal':>9}")
    print(f"  {'-'*14} {'-'*4} {'-'*9} {'-'*9} {'-'*9}")
    for qtype, stats in metrics["by_type"].items():
        print(f"  {qtype:<14} {stats['count']:>4} "
              f"{stats['correctness']*100:>8.1f}% "
              f"{stats['grounding']*100:>8.1f}% "
              f"{stats['refusal']*100:>8.1f}%")


# =========================================================================
# Save results
# =========================================================================

def save_results(results: list[dict], metrics: dict, path: Path) -> None:
    """Save evaluation results and metrics to JSON."""
    output = {
        "metrics": metrics,
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved evaluation results -> {path}")
