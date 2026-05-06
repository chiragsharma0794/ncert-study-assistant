"""Verification test for evaluator_v2 with full 20-question run."""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from evaluator_v2 import (
    score_faithfulness, score_answer_relevancy,
    score_context_precision, score_context_recall,
    evaluate_pipeline_v2, compute_metrics_v2, print_report_v2, save_results_v2,
)
from retriever import BM25Retriever, load_chunks
from cache import CachedGenerator, MockLLMGenerator, ResponseCache

CHUNKS_PATH = str(PROJECT_ROOT / "outputs" / "chunks_semantic.json")
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_V2_PATH = PROJECT_ROOT / "evaluation" / "results_v2.json"

chunks = load_chunks(CHUNKS_PATH)
corpus_text = " ".join(c.get("text", c.get("content", "")) for c in chunks)

with open(QUESTIONS_PATH) as f:
    questions = json.load(f)

# =========================================================================
# TEST 1: Faithfulness scoring
# =========================================================================
print("=" * 72)
print("  TEST 1: Faithfulness Scoring")
print("=" * 72)

ctx = [{"text": "The cell is the basic unit of life. Robert Hooke discovered cells."}]

# Faithful answer
score = score_faithfulness("The cell is the basic unit of life.", ctx)
assert score >= 0.8, f"Should be faithful: {score}"
print(f"  [OK] Faithful answer: {score:.3f}")

# Hallucinated answer
score = score_faithfulness("Quantum physics explains cellular dynamics.", ctx)
assert score < 0.5, f"Should be unfaithful: {score}"
print(f"  [OK] Hallucinated answer: {score:.3f}")

# Refusal always faithful
score = score_faithfulness("I could not find this in the textbook.", ctx)
assert score == 1.0
print(f"  [OK] Refusal: {score:.3f}")

# =========================================================================
# TEST 2: Answer relevancy scoring
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 2: Answer Relevancy Scoring")
print("=" * 72)

score = score_answer_relevancy(
    "The cell membrane is a thin barrier that controls entry and exit.",
    "What is a cell membrane?"
)
assert score > 0.5
print(f"  [OK] Relevant answer: {score:.3f}")

score = score_answer_relevancy(
    "Photosynthesis occurs in chloroplasts.",
    "What is a cell membrane?"
)
assert score < 0.5
print(f"  [OK] Irrelevant answer: {score:.3f}")

# =========================================================================
# TEST 3: Context precision scoring
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 3: Context Precision Scoring")
print("=" * 72)

relevant_chunks = [
    {"text": "The cell membrane is a thin barrier around the cell."},
    {"text": "The plasma membrane controls what enters and exits."},
    {"text": "Photosynthesis produces glucose in plants."},  # irrelevant
]
score = score_context_precision(
    "What is a cell membrane?", relevant_chunks, ["cell", "membrane", "plasma"]
)
assert 0.5 <= score <= 1.0
print(f"  [OK] Mixed relevance: {score:.3f} (2/3 relevant)")

# =========================================================================
# TEST 4: Context recall scoring
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 4: Context Recall Scoring")
print("=" * 72)

score = score_context_recall(
    "Mitochondria are the powerhouses of the cell. They release energy as ATP.",
    [{"text": "Mitochondria release energy stored in food as ATP, the powerhouse of the cell."}]
)
assert score > 0.5
print(f"  [OK] Good recall: {score:.3f}")

# OOS: no expected answer = perfect recall
score = score_context_recall(None, [{"text": "anything"}])
assert score == 1.0
print(f"  [OK] OOS recall: {score:.3f}")

# =========================================================================
# TEST 5: Full 20-question evaluation with MockLLM
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 5: Full 20-Question Evaluation (MockLLM)")
print("=" * 72)

retriever = BM25Retriever(CHUNKS_PATH)
cache = ResponseCache(PROJECT_ROOT / "outputs" / "test_eval_cache.json")
mock_gen = CachedGenerator(MockLLMGenerator(), cache, offline=False)

results = evaluate_pipeline_v2(
    questions, retriever, mock_gen, corpus_text, top_k=5
)

assert len(results) == 20, f"Expected 20, got {len(results)}"
print(f"  [OK] {len(results)} questions evaluated")

# Check V1 fields present
for r in results:
    assert "correctness" in r
    assert "grounding" in r
    assert "refusal_correct" in r
print("  [OK] V1 fields present on all results")

# Check V2 fields present
for r in results:
    assert "faithfulness" in r
    assert "answer_relevancy" in r
    assert "context_precision" in r
    assert "context_recall" in r
    assert 0.0 <= r["faithfulness"] <= 1.0
    assert 0.0 <= r["answer_relevancy"] <= 1.0
    assert 0.0 <= r["context_precision"] <= 1.0
    assert 0.0 <= r["context_recall"] <= 1.0
print("  [OK] V2 fields present and in [0, 1] range")

# =========================================================================
# TEST 6: Metrics computation
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 6: Metrics Aggregation")
print("=" * 72)

metrics = compute_metrics_v2(results)
assert metrics["version"] == "v2"
assert "v1_metrics" in metrics
assert "v2_metrics" in metrics
assert "combined_score" in metrics
assert "by_type" in metrics
assert 0.0 <= metrics["combined_score"] <= 1.0
print(f"  [OK] V1: {metrics['v1_metrics']}")
print(f"  [OK] V2: {metrics['v2_metrics']}")
print(f"  [OK] Combined: {metrics['combined_score']}")

# =========================================================================
# TEST 7: Report printing + save
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 7: Report & Save")
print("=" * 72)

print_report_v2(results, metrics)

save_results_v2(results, metrics, RESULTS_V2_PATH)
assert RESULTS_V2_PATH.exists()

# Verify saved schema
with open(RESULTS_V2_PATH) as f:
    saved = json.load(f)
assert saved["version"] == "v2"
assert "v1_metrics" in saved
assert "v2_metrics" in saved
assert len(saved["results"]) == 20
print(f"  [OK] results_v2.json saved ({RESULTS_V2_PATH.stat().st_size / 1024:.1f} KB)")

# Verify backward compat: V1 results.json still exists
v1_path = PROJECT_ROOT / "evaluation" / "results.json"
assert v1_path.exists()
print(f"  [OK] V1 results.json preserved (backward compatible)")

# Cleanup test cache
cache.clear_all()
test_cache_path = PROJECT_ROOT / "outputs" / "test_eval_cache.json"
if test_cache_path.exists():
    test_cache_path.unlink()

# =========================================================================
print("\n" + "=" * 72)
print("  ALL TESTS PASSED [OK]")
print("=" * 72)
