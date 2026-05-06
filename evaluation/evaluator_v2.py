"""
evaluation/evaluator_v2.py -- V2 evaluation framework with RAGAS-style metrics.

Adds four semantic evaluation metrics alongside V1's keyword-overlap scoring:
  - faithfulness     : is the answer faithful to the retrieved context?
  - answer_relevancy : is the answer relevant to the question?
  - context_precision: are the retrieved chunks actually relevant?
  - context_recall   : do retrieved chunks cover the expected answer?

DESIGN DECISION: RAGAS LLM BRIDGE
==================================
RAGAS natively requires OpenAI or a LangChain-compatible LLM for its
internal evaluation calls.  Three options were evaluated:

  (a) LiteLLM wrapper:
      Pros: Gemini support, minimal code.
      Cons: Adds a dependency, may not work with free-tier quota limits,
            requires RAGAS to call the LLM internally (no caching control).

  (b) Custom RAGAS LLM adapter:
      Pros: Full control, can use ResponseCache.
      Cons: RAGAS internals change frequently, adapter may break.

  (c) IMPLEMENT METRICS MANUALLY (CHOSEN):
      Pros: Zero dependency on RAGAS internals, full caching control,
            works in offline mode with MockLLM, deterministic scoring,
            no extra API calls beyond generation.
      Cons: Metrics are approximations (not RAGAS's exact formulas).

  Decision: Option (c).  We implement the four metrics using lightweight
  NLP heuristics + optional LLM-as-judge.  This avoids the RAGAS/OpenAI
  dependency entirely while providing semantically meaningful scores.

  The metrics follow RAGAS definitions but use practical approximations:
    - faithfulness    -> sentence-level claim verification against context
    - answer_relevancy -> question-answer semantic overlap
    - context_precision -> relevant chunks / total chunks (scored by overlap)
    - context_recall   -> expected_answer coverage by retrieved chunks

DESIGN DECISION: MOCK/CACHED LLM WITH RAGAS-STYLE METRICS
==========================================================
  Since we implement metrics manually (not via RAGAS library), the
  LLM is ONLY called for generation (already cached via CachedGenerator).
  The evaluation metrics themselves use NLP heuristics -- no LLM calls.
  This means offline mode (MockLLM + cache) works perfectly.

DESIGN DECISION: results_v2.json FORMAT
========================================
  results_v2.json includes BOTH V1 and V2 metrics side-by-side:

  {
    "version": "v2",
    "v1_metrics": { correctness_pct, grounding_pct, refusal_accuracy_pct },
    "v2_metrics": { faithfulness, answer_relevancy, context_precision, context_recall },
    "combined_score": weighted average,
    "results": [ per-question results with all V1 + V2 fields ]
  }

Usage:
    from evaluator_v2 import evaluate_pipeline_v2, print_report_v2
    results = evaluate_pipeline_v2(questions, retriever, generator, corpus_text)
    print_report_v2(results)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Import V1 scoring functions (unchanged)
from evaluator import score_correctness, score_grounding, score_refusal


# =========================================================================
# V2 RAGAS-style scoring functions
# =========================================================================

def score_faithfulness(answer: str, context_chunks: list[dict]) -> float:
    """Score whether the answer is faithful to the retrieved context.

    RAGAS Definition: proportion of answer claims that can be inferred
    from the context.

    Our approximation: split the answer into sentences, then check what
    fraction of sentences have significant word overlap with at least
    one context chunk.  This catches hallucinated claims that mention
    terms absent from any retrieved chunk.

    Parameters
    ----------
    answer : str
        The generated answer.
    context_chunks : list[dict]
        Retrieved chunks, each with a "text" field.

    Returns
    -------
    float
        Score between 0.0 (unfaithful) and 1.0 (fully faithful).
    """
    if not answer or not context_chunks:
        return 0.0

    # Skip refusal answers (they are always faithful by definition)
    if "could not find" in answer.lower():
        return 1.0

    context_text = " ".join((c.get("content") or c.get("text", "")).lower() for c in context_chunks)
    context_words = set(context_text.split())

    # Split answer into sentences
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    if not sentences:
        return 0.0

    # Stopwords to ignore in overlap check
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "shall", "should", "may", "might", "can", "could",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "it", "its", "this", "that",
        "these", "those", "i", "you", "he", "she", "we", "they",
        "and", "but", "or", "nor", "not", "so", "if", "than",
        "also", "very", "just", "about", "all", "each", "every",
        "both", "few", "more", "most", "other", "some", "such",
        "no", "only", "own", "same", "too", "which", "who", "whom",
    }

    faithful_count = 0
    for sent in sentences:
        sent_words = set(sent.lower().split()) - stopwords
        if not sent_words:
            faithful_count += 1  # empty/trivial sentence
            continue
        overlap = sent_words & context_words
        # A sentence is faithful if >40% of its content words appear in context
        ratio = len(overlap) / len(sent_words) if sent_words else 0
        if ratio >= 0.4:
            faithful_count += 1

    return faithful_count / len(sentences)


def score_answer_relevancy(answer: str, question: str) -> float:
    """Score whether the answer is relevant to the question.

    RAGAS Definition: measures if the answer addresses what was asked.

    Our approximation: bidirectional keyword overlap between question
    and answer, weighted toward question terms appearing in the answer.
    A relevant answer echoes the subject of the question.

    Parameters
    ----------
    answer : str
        The generated answer.
    question : str
        The original question.

    Returns
    -------
    float
        Score between 0.0 (irrelevant) and 1.0 (highly relevant).
    """
    if not answer or not question:
        return 0.0

    # Refusal is relevant for OOS questions (but we score it at 0.5
    # since it doesn't actually answer anything)
    if "could not find" in answer.lower():
        return 0.5

    stopwords = {
        "what", "is", "the", "a", "an", "how", "does", "do", "which",
        "who", "when", "where", "why", "are", "was", "were", "in",
        "of", "and", "to", "for", "on", "with", "at", "by", "from",
        "also", "its", "it", "this", "that", "can", "could", "would",
    }

    q_words = set(question.lower().split()) - stopwords
    a_words = set(answer.lower().split()) - stopwords

    if not q_words:
        return 1.0  # trivial question

    # What fraction of question keywords appear in the answer?
    q_in_a = len(q_words & a_words) / len(q_words)

    # Bonus: if the answer is reasonably long (not one-word), add confidence
    length_bonus = min(1.0, len(a_words) / 10)  # caps at 10 unique words

    return min(1.0, q_in_a * 0.7 + length_bonus * 0.3)


def score_context_precision(question: str, context_chunks: list[dict],
                            key_terms: list[str]) -> float:
    """Score whether retrieved chunks are actually relevant to the question.

    RAGAS Definition: proportion of retrieved chunks that are relevant.

    Our approximation: for each chunk, check if it contains any of the
    question's key terms or question words.  Chunks with zero overlap
    are irrelevant noise.

    Parameters
    ----------
    question : str
        The original question.
    context_chunks : list[dict]
        Retrieved chunks.
    key_terms : list[str]
        Expected key terms from the ground truth.

    Returns
    -------
    float
        Score between 0.0 (no relevant chunks) and 1.0 (all relevant).
    """
    if not context_chunks:
        return 0.0

    # Build relevance vocabulary from question + key_terms
    stopwords = {"what", "is", "the", "a", "how", "does", "do", "which",
                 "and", "to", "in", "of", "for", "on", "with", "are"}
    q_words = set(question.lower().split()) - stopwords
    kw_words = set(t.lower() for t in key_terms)
    relevance_vocab = q_words | kw_words

    if not relevance_vocab:
        return 0.5  # can't judge without terms

    relevant_count = 0
    for chunk in context_chunks:
        chunk_text = (chunk.get("content") or chunk.get("text", "")).lower()
        hits = sum(1 for w in relevance_vocab if w in chunk_text)
        if hits >= 2:  # at least 2 relevance words
            relevant_count += 1

    return relevant_count / len(context_chunks)


def score_context_recall(expected_answer: str,
                         context_chunks: list[dict]) -> float:
    """Score whether retrieved chunks cover the expected answer.

    RAGAS Definition: proportion of the ground-truth answer that can
    be attributed to the retrieved context.

    Our approximation: split expected answer into key phrases and check
    what fraction appear in the concatenated context text.

    Parameters
    ----------
    expected_answer : str or None
        The ground-truth expected answer.  None for OOS questions.
    context_chunks : list[dict]
        Retrieved chunks.

    Returns
    -------
    float
        Score between 0.0 (no coverage) and 1.0 (full coverage).
    """
    if not expected_answer:
        return 1.0  # OOS questions: no expected answer = perfect recall

    if not context_chunks:
        return 0.0

    context_text = " ".join((c.get("content") or c.get("text", "")).lower() for c in context_chunks)

    # Extract meaningful words from expected answer
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "and", "or", "but", "not", "it", "its", "this", "that",
        "they", "them", "their", "which", "who", "also", "each",
    }
    expected_words = set(expected_answer.lower().split()) - stopwords
    if not expected_words:
        return 1.0

    found = sum(1 for w in expected_words if w in context_text)
    return found / len(expected_words)


# =========================================================================
# V2 evaluation runner
# =========================================================================

def evaluate_pipeline_v2(
    questions: list[dict],
    retriever,
    generator,
    corpus_text: str,
    top_k: int = 5,
    dry_run: bool = False,
) -> list[dict]:
    """Run V1 + V2 evaluation on all questions.

    Returns per-question results with both V1 keyword metrics and
    V2 RAGAS-style semantic metrics.

    Parameters
    ----------
    questions : list[dict]
        Evaluation questions with {id, question, expected_answer, key_terms, type, in_scope}.
    retriever : object
        Retriever with retrieve(query, top_k) method.
    generator : object
        Generator with generate(query, chunks) method.
        Can be CachedGenerator or MockLLMGenerator.
    corpus_text : str
        Full corpus text for grounding checks.
    top_k : int
        Number of chunks to retrieve.
    dry_run : bool
        If True, skip generation.

    Returns
    -------
    list[dict]
        Per-question results with all V1 + V2 metrics.
    """
    results = []

    for q in questions:
        # ── Retrieve ─────────────────────────────────────────────────
        retrieved = retriever.retrieve(q["question"], top_k=top_k)
        top_scores = [r["score"] for r in retrieved]

        # ── Generate ─────────────────────────────────────────────────
        if dry_run or generator is None:
            answer_text = "[DRY-RUN]"
            sources = []
            refused_flag = False
        else:
            gen_result = generator.generate(q["question"], retrieved)
            answer_text = gen_result["answer"]
            sources = gen_result["sources"]
            refused_flag = gen_result["refused"]

        # ── V1 scores ────────────────────────────────────────────────
        correctness = score_correctness(answer_text, q.get("key_terms", []))
        grounding = score_grounding(answer_text, corpus_text)
        refusal = score_refusal(answer_text, q["in_scope"])

        # ── V2 RAGAS-style scores ────────────────────────────────────
        faithfulness = score_faithfulness(answer_text, retrieved)
        answer_rel = score_answer_relevancy(answer_text, q["question"])
        ctx_precision = score_context_precision(
            q["question"], retrieved, q.get("key_terms", [])
        )
        ctx_recall = score_context_recall(
            q.get("expected_answer"), retrieved
        )

        results.append({
            # Identity
            "id":               q["id"],
            "question":         q["question"],
            "type":             q["type"],
            "in_scope":         q["in_scope"],
            # Generation
            "answer":           answer_text,
            "sources":          sources,
            "refused":          refused_flag,
            "retrieval_scores": top_scores[:3],
            # V1 metrics (preserved)
            "correctness":      correctness,
            "grounding":        grounding,
            "refusal_correct":  refusal,
            # V2 metrics (new)
            "faithfulness":     round(faithfulness, 3),
            "answer_relevancy": round(answer_rel, 3),
            "context_precision": round(ctx_precision, 3),
            "context_recall":   round(ctx_recall, 3),
        })

    return results


# =========================================================================
# Aggregate metrics
# =========================================================================

def compute_metrics_v2(results: list[dict]) -> dict:
    """Compute V1 + V2 aggregate metrics.

    Returns results_v2.json schema with both metric sets side-by-side.
    """
    def _avg(vals):
        return sum(vals) / len(vals) if vals else 0.0

    # ── V1 aggregates ────────────────────────────────────────────────
    in_scope = [r for r in results if r["in_scope"]]
    v1 = {
        "correctness_pct":    round(_avg([r["correctness"] for r in in_scope]) * 100, 1),
        "grounding_pct":      round(_avg([r["grounding"] for r in in_scope]) * 100, 1),
        "refusal_accuracy_pct": round(_avg([r["refusal_correct"] for r in results]) * 100, 1),
    }

    # ── V2 aggregates ────────────────────────────────────────────────
    v2 = {
        "faithfulness":       round(_avg([r["faithfulness"] for r in in_scope]), 3),
        "answer_relevancy":   round(_avg([r["answer_relevancy"] for r in in_scope]), 3),
        "context_precision":  round(_avg([r["context_precision"] for r in in_scope]), 3),
        "context_recall":     round(_avg([r["context_recall"] for r in in_scope]), 3),
    }

    # ── Combined score (weighted average of V2 metrics) ──────────────
    combined = round(
        v2["faithfulness"] * 0.30
        + v2["answer_relevancy"] * 0.20
        + v2["context_precision"] * 0.25
        + v2["context_recall"] * 0.25,
        3,
    )

    # ── By type breakdown ────────────────────────────────────────────
    by_type = {}
    for qtype in ["factual", "paraphrased", "out_of_scope"]:
        typed = [r for r in results if r["type"] == qtype]
        if not typed:
            continue
        typed_in_scope = [r for r in typed if r["in_scope"]]
        by_type[qtype] = {
            "count": len(typed),
            # V1
            "correctness": round(_avg([r["correctness"] for r in typed_in_scope]), 3),
            "grounding":   round(_avg([r["grounding"] for r in typed_in_scope]), 3),
            "refusal":     round(_avg([r["refusal_correct"] for r in typed]), 3),
            # V2
            "faithfulness":      round(_avg([r["faithfulness"] for r in typed_in_scope]), 3),
            "answer_relevancy":  round(_avg([r["answer_relevancy"] for r in typed_in_scope]), 3),
            "context_precision": round(_avg([r["context_precision"] for r in typed_in_scope]), 3),
            "context_recall":    round(_avg([r["context_recall"] for r in typed_in_scope]), 3),
        }

    return {
        "version": "v2",
        "total_questions": len(results),
        "v1_metrics": v1,
        "v2_metrics": v2,
        "combined_score": combined,
        "by_type": by_type,
    }


# =========================================================================
# Pretty printing
# =========================================================================

def print_report_v2(results: list[dict], metrics: Optional[dict] = None) -> None:
    """Print a formatted V2 evaluation report."""
    if metrics is None:
        metrics = compute_metrics_v2(results)

    print("=" * 90)
    print("  EVALUATION RESULTS V2 (V1 + RAGAS-style metrics)")
    print("=" * 90)

    # Per-question table
    print(f"\n  {'ID':<6} {'Type':<14} {'C':>3} {'G':>3} {'R':>3} "
          f"{'Faith':>6} {'AnsRel':>7} {'CtxP':>5} {'CtxR':>5}  Question")
    print(f"  {'-'*6} {'-'*14} {'-'*3} {'-'*3} {'-'*3} "
          f"{'-'*6} {'-'*7} {'-'*5} {'-'*5}  {'-'*25}")

    for r in results:
        q_preview = r["question"][:25]
        print(f"  {r['id']:<6} {r['type']:<14} "
              f"{r['correctness']:>3} {r['grounding']:>3} {r['refusal_correct']:>3} "
              f"{r['faithfulness']:>6.2f} {r['answer_relevancy']:>7.2f} "
              f"{r['context_precision']:>5.2f} {r['context_recall']:>5.2f}  "
              f"{q_preview}...")

    # V1 metrics
    v1 = metrics["v1_metrics"]
    print(f"\n{'='*90}")
    print(f"  V1 METRICS (keyword-overlap)")
    print(f"{'='*90}")
    print(f"  Correctness (in-scope)  : {v1['correctness_pct']:.1f}%")
    print(f"  Grounding (in-scope)    : {v1['grounding_pct']:.1f}%")
    print(f"  Refusal accuracy (all)  : {v1['refusal_accuracy_pct']:.1f}%")

    # V2 metrics
    v2 = metrics["v2_metrics"]
    print(f"\n{'='*90}")
    print(f"  V2 METRICS (RAGAS-style semantic)")
    print(f"{'='*90}")
    print(f"  Faithfulness            : {v2['faithfulness']:.3f}")
    print(f"  Answer Relevancy        : {v2['answer_relevancy']:.3f}")
    print(f"  Context Precision       : {v2['context_precision']:.3f}")
    print(f"  Context Recall          : {v2['context_recall']:.3f}")
    print(f"\n  Combined Score          : {metrics['combined_score']:.3f}")

    # By type
    print(f"\n  {'Type':<14} {'N':>4} {'C%':>6} {'G%':>6} {'R%':>6} "
          f"{'Faith':>6} {'AnsR':>5} {'CtxP':>5} {'CtxR':>5}")
    print(f"  {'-'*14} {'-'*4} {'-'*6} {'-'*6} {'-'*6} "
          f"{'-'*6} {'-'*5} {'-'*5} {'-'*5}")
    for qtype, s in metrics["by_type"].items():
        print(f"  {qtype:<14} {s['count']:>4} "
              f"{s['correctness']*100:>5.1f}% {s['grounding']*100:>5.1f}% "
              f"{s['refusal']*100:>5.1f}% "
              f"{s['faithfulness']:>6.3f} {s['answer_relevancy']:>5.3f} "
              f"{s['context_precision']:>5.3f} {s['context_recall']:>5.3f}")


# =========================================================================
# Save results
# =========================================================================

def save_results_v2(results: list[dict], metrics: dict, path: Path) -> None:
    """Save V2 evaluation results to JSON.

    Schema:
    {
      "version": "v2",
      "v1_metrics": { correctness_pct, grounding_pct, refusal_accuracy_pct },
      "v2_metrics": { faithfulness, answer_relevancy, context_precision, context_recall },
      "combined_score": float,
      "by_type": { ... },
      "results": [ per-question with all V1 + V2 fields ]
    }
    """
    output = {
        "version": metrics["version"],
        "total_questions": metrics["total_questions"],
        "v1_metrics": metrics["v1_metrics"],
        "v2_metrics": metrics["v2_metrics"],
        "combined_score": metrics["combined_score"],
        "by_type": metrics["by_type"],
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved V2 evaluation results -> {path}")
