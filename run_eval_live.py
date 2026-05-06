"""
run_eval_live.py -- Run V2 evaluation with live Gemini API.

Populates the response cache so subsequent runs are instant.
Rate-limited: 3-second delay between API calls.

Usage:
    python run_eval_live.py
"""

import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from retriever import BM25Retriever, load_chunks
from generator import GroundedGenerator
from cache import ResponseCache
from evaluator_v2 import (
    evaluate_pipeline_v2,
    compute_metrics_v2,
    print_report_v2,
    save_results_v2,
)

# ── Paths ────────────────────────────────────────────────────────────────
CHUNKS_PATH    = str(PROJECT_ROOT / "outputs" / "chunks_semantic.json")
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_PATH   = PROJECT_ROOT / "evaluation" / "results_v2.json"
CACHE_PATH     = str(PROJECT_ROOT / "outputs" / "response_cache.json")

# ── Rate limiting ────────────────────────────────────────────────────────
DELAY_BETWEEN_CALLS = 3  # seconds -- conservative for free tier


def main():
    print("=" * 80)
    print("  NCERT Study Assistant V2 -- LIVE EVALUATION")
    print("=" * 80)

    # ── Load chunks (for corpus_text used by grounding scorer) ────────
    print("\n[1/5] Loading chunks...")
    chunks = load_chunks(CHUNKS_PATH)
    corpus_text = " ".join(c["text"] for c in chunks)
    print(f"       {len(chunks)} chunks loaded from chunks_semantic.json")

    # ── Load questions ───────────────────────────────────────────────
    print("[2/5] Loading questions...")
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"       {len(questions)} questions loaded")

    # ── Init retriever (V2 takes path string) ────────────────────────
    print("[3/5] Building BM25 retriever...")
    retriever = BM25Retriever(chunks_path=CHUNKS_PATH)
    print(f"       BM25Retriever ready ({retriever.n_docs} docs)")

    # ── Init generator with cache ────────────────────────────────────
    print("[4/5] Initializing GroundedGenerator + ResponseCache...")
    cache = ResponseCache(cache_path=CACHE_PATH)
    generator = GroundedGenerator(cache=cache)
    print(f"       {generator}")

    # ── Run evaluation with rate limiting ────────────────────────────
    print(f"\n[5/5] Running evaluation ({len(questions)} questions, "
          f"{DELAY_BETWEEN_CALLS}s delay between API calls)...")
    print("-" * 80)

    results = []
    api_calls_made = 0
    cache_hits = 0

    for i, q in enumerate(questions):
        prefix = f"  [{i+1:2d}/{len(questions)}]"

        # Retrieve
        retrieved = retriever.retrieve(q["question"], top_k=5)
        chunk_ids = [c["chunk_id"] for c in retrieved]

        # Check cache before calling API
        cached = cache.get(q["question"], chunk_ids)
        if cached is not None:
            print(f"{prefix} CACHE HIT  | {q['id']} | {q['question'][:50]}...")
            gen_result = cached
            cache_hits += 1
        else:
            print(f"{prefix} API CALL   | {q['id']} | {q['question'][:50]}...")
            try:
                gen_result = generator.generate(q["question"], retrieved)
                api_calls_made += 1
            except Exception as e:
                err_str = str(e)
                print(f"         ERROR: {err_str[:100]}")
                if "429" in err_str or "quota" in err_str.lower() or "resource" in err_str.lower():
                    print(f"\n  ⚠ QUOTA EXHAUSTED after {api_calls_made} API calls.")
                    print(f"  Saving partial results ({len(results)} questions scored)...")
                    break
                raise

            # Rate limit (only after actual API calls, not on last question)
            if i < len(questions) - 1:
                time.sleep(DELAY_BETWEEN_CALLS)

        # Score this question using V1+V2 metrics
        from evaluator_v2 import (
            score_correctness, score_grounding, score_refusal,
            score_faithfulness, score_answer_relevancy,
            score_context_precision, score_context_recall,
        )

        answer_text = gen_result["answer"]
        sources = gen_result.get("sources", [])
        refused_flag = gen_result.get("refused", False)

        results.append({
            "id":               q["id"],
            "question":         q["question"],
            "type":             q["type"],
            "in_scope":         q["in_scope"],
            "answer":           answer_text,
            "sources":          sources,
            "refused":          refused_flag,
            "retrieval_scores": [r["score"] for r in retrieved[:3]],
            "correctness":      score_correctness(answer_text, q.get("key_terms", [])),
            "grounding":        score_grounding(answer_text, corpus_text),
            "refusal_correct":  score_refusal(answer_text, q["in_scope"]),
            "faithfulness":     round(score_faithfulness(answer_text, retrieved), 3),
            "answer_relevancy": round(score_answer_relevancy(answer_text, q["question"]), 3),
            "context_precision": round(score_context_precision(
                q["question"], retrieved, q.get("key_terms", [])
            ), 3),
            "context_recall":   round(score_context_recall(
                q.get("expected_answer"), retrieved
            ), 3),
        })

    # ── Compute and display metrics ──────────────────────────────────
    print("\n" + "-" * 80)
    print(f"  Completed: {len(results)}/{len(questions)} questions")
    print(f"  API calls: {api_calls_made} | Cache hits: {cache_hits}")
    print("-" * 80)

    metrics = compute_metrics_v2(results)
    print_report_v2(results, metrics)

    # ── Save results ─────────────────────────────────────────────────
    save_results_v2(results, metrics, RESULTS_PATH)

    # ── Validation summary ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  VALIDATION CHECKLIST")
    print("=" * 80)

    # Check answers are real (not mock default)
    mock_default = "Robert Hooke first observed cells in 1665"
    n_unique_answers = len(set(r["answer"][:80] for r in results if r["in_scope"]))
    n_mock = sum(1 for r in results if r["in_scope"] and mock_default in r["answer"])
    n_real = sum(1 for r in results if r["in_scope"]) - n_mock

    n_refused_oos = sum(1 for r in results
                        if not r["in_scope"] and r["refused"])

    v1 = metrics["v1_metrics"]
    v2 = metrics["v2_metrics"]

    checks = [
        ("Real answers (not mock/dry-run)",
         n_real >= 10 and n_unique_answers > 5,
         f"{n_real} real, {n_unique_answers} unique, {n_mock} mock"),
        ("Correctness > 50%",
         v1["correctness_pct"] > 50,
         f"{v1['correctness_pct']:.1f}%"),
        ("Grounding > 80%",
         v1["grounding_pct"] > 80,
         f"{v1['grounding_pct']:.1f}%"),
        ("OOS refusal > 60%",
         v1["refusal_accuracy_pct"] > 60,
         f"{v1['refusal_accuracy_pct']:.1f}% ({n_refused_oos}/{sum(1 for r in results if not r['in_scope'])} OOS refused)"),
        ("Faithfulness > 0.3",
         v2["faithfulness"] > 0.3,
         f"{v2['faithfulness']:.3f}"),
        ("Answer relevancy > 0.3",
         v2["answer_relevancy"] > 0.3,
         f"{v2['answer_relevancy']:.3f}"),
        ("Context precision > 0.5",
         v2["context_precision"] > 0.5,
         f"{v2['context_precision']:.3f}"),
        ("Context recall > 0.5",
         v2["context_recall"] > 0.5,
         f"{v2['context_recall']:.3f}"),
    ]

    all_pass = True
    for label, passed, value in checks:
        icon = "  [PASS]" if passed else "  [FAIL]"
        if not passed:
            all_pass = False
        print(f"{icon} {label}: {value}")

    status = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED -- review above"
    print(f"\n  {status}")
    print("=" * 80)


if __name__ == "__main__":
    main()
