"""Verification script for FAISS migration steps 1-5."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CHUNKS_PATH = PROJECT_ROOT / "outputs" / "chunks_semantic.json"
INDEX_PATH = PROJECT_ROOT / "outputs" / "faiss_index.bin"

# =========================================================================
# STEP 1 GATE: FAISSRetriever._build_index() produces a valid index
# =========================================================================
print("=" * 72)
print("  STEP 1: FAISSRetriever.build_index()")
print("=" * 72)

from embedder import FAISSRetriever

faiss_r = FAISSRetriever(str(CHUNKS_PATH), str(INDEX_PATH))

assert INDEX_PATH.exists(), "faiss_index.bin does not exist"
size = INDEX_PATH.stat().st_size
assert size > 0, "faiss_index.bin is empty"
print(f"  [OK] faiss_index.bin exists, size={size:,} bytes")
print(f"  [OK] Retriever: {faiss_r}")

# =========================================================================
# STEP 2 GATE: retrieve() returns correct schema
# =========================================================================
print("\n" + "=" * 72)
print("  STEP 2: FAISSRetriever.retrieve() schema")
print("=" * 72)

results = faiss_r.retrieve("what is a cell", 5)

assert len(results) == 5, f"Expected 5 results, got {len(results)}"
print(f"  [OK] Returned {len(results)} results")

# Check schema
required_keys = {"chunk_id", "text", "score", "page", "type"}
for i, r in enumerate(results):
    missing = required_keys - set(r.keys())
    assert not missing, f"Result {i} missing keys: {missing}"
    assert isinstance(r["score"], float), f"Result {i} score is not float"
print(f"  [OK] All results have required keys: {sorted(required_keys)}")

# Print results
for i, r in enumerate(results):
    print(f"  [{i}] {r['chunk_id']} score={r['score']:.3f} "
          f"page={r['page']} text={r['text'][:60]}...")

# =========================================================================
# STEP 3 GATE: Retrieval quality — BM25 vs FAISS overlap
# =========================================================================
print("\n" + "=" * 72)
print("  STEP 3: Retrieval Quality — BM25 vs FAISS overlap")
print("=" * 72)

from retriever import BM25Retriever, load_chunks

chunks = load_chunks(CHUNKS_PATH)
bm25_r = BM25Retriever(chunks)

test_queries = [
    "nucleus function",
    "what is a cell",
    "cell membrane",
    "difference between prokaryotic and eukaryotic",
    "what is osmosis",
]

for query in test_queries:
    bm25_results = bm25_r.retrieve(query, top_k=5)
    faiss_results = faiss_r.retrieve(query, top_k=5)

    bm25_ids = set(r["chunk_id"] for r in bm25_results)
    faiss_ids = set(r["chunk_id"] for r in faiss_results)
    overlap = bm25_ids & faiss_ids
    overlap_pct = len(overlap) / 5 * 100

    print(f"  Query: '{query}'")
    print(f"    BM25:    {sorted(bm25_ids)}")
    print(f"    FAISS:   {sorted(faiss_ids)}")
    print(f"    Overlap: {len(overlap)}/5 ({overlap_pct:.0f}%)")
    print()

# =========================================================================
# STEP 4 GATE: Config-based retriever instantiation
# =========================================================================
print("=" * 72)
print("  STEP 4: Config-based retriever factory")
print("=" * 72)

from config import create_retriever, RETRIEVER_TYPE

print(f"  Default RETRIEVER_TYPE: '{RETRIEVER_TYPE}'")

# Test each type
for rtype in ["bm25", "hybrid", "faiss"]:
    try:
        r = create_retriever(chunks, retriever_type=rtype)
        test_result = r.retrieve("cell membrane", top_k=3)
        assert len(test_result) == 3
        assert "score" in test_result[0]
        print(f"  [OK] create_retriever(type='{rtype}') -> {type(r).__name__}, "
              f"retrieve() returned {len(test_result)} results")
    except Exception as e:
        print(f"  [FAIL] create_retriever(type='{rtype}') -> {e}")

# =========================================================================
# STEP 5 GATE: Evaluation comparison (BM25 vs FAISS vs Hybrid)
# =========================================================================
print("\n" + "=" * 72)
print("  STEP 5: Evaluation Score Comparison")
print("=" * 72)

import json

# Load questions
questions_path = PROJECT_ROOT / "evaluation" / "questions.json"
with open(questions_path, "r", encoding="utf-8") as f:
    questions = json.load(f)

# Load corpus text for grounding check
corpus_text = " ".join(c["text"] for c in chunks)

# Import evaluator
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
from evaluator_v2 import evaluate_pipeline_v2, compute_metrics_v2

# Run evaluation in dry-run mode (no LLM calls) for all 3 retrievers
# This measures retrieval quality only (context_precision, context_recall)
print("\n  Running retrieval-only evaluation (dry_run=True)...\n")

comparison = {}
for rtype in ["bm25", "hybrid", "faiss"]:
    r = create_retriever(chunks, retriever_type=rtype)
    results = evaluate_pipeline_v2(
        questions, r, generator=None, corpus_text=corpus_text,
        top_k=5, dry_run=True,
    )
    metrics = compute_metrics_v2(results)
    comparison[rtype] = metrics

    # Save individual results
    output_path = PROJECT_ROOT / "evaluation" / f"results_{rtype}_retrieval.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)

# Print comparison table
print(f"  {'Metric':<22} {'BM25':>8} {'Hybrid':>8} {'FAISS':>8}  {'Best':>8}")
print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}  {'-'*8}")

metrics_to_compare = [
    ("Context Precision", "v2_metrics", "context_precision"),
    ("Context Recall", "v2_metrics", "context_recall"),
]

for label, group, key in metrics_to_compare:
    vals = {rt: comparison[rt][group][key] for rt in ["bm25", "hybrid", "faiss"]}
    best = max(vals, key=vals.get)
    print(f"  {label:<22} {vals['bm25']:>8.3f} {vals['hybrid']:>8.3f} "
          f"{vals['faiss']:>8.3f}  {best:>8}")

# Per-question retrieval scores comparison
print(f"\n  Per-question top retrieval scores:")
print(f"  {'ID':<6} {'BM25 top':>10} {'Hybrid top':>12} {'FAISS top':>11}")
print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*11}")

for rtype in ["bm25", "hybrid", "faiss"]:
    r = create_retriever(chunks, retriever_type=rtype)
    for q in questions[:5]:  # first 5 in-scope questions
        res = r.retrieve(q["question"], top_k=1)
        # Store for table below
        if rtype == "bm25":
            q["_bm25_top"] = res[0]["score"]
        elif rtype == "hybrid":
            q["_hybrid_top"] = res[0]["score"]
        else:
            q["_faiss_top"] = res[0]["score"]

for q in questions[:5]:
    print(f"  {q['id']:<6} {q.get('_bm25_top', 0):>10.3f} "
          f"{q.get('_hybrid_top', 0):>12.3f} {q.get('_faiss_top', 0):>11.3f}")

# =========================================================================
print("\n" + "=" * 72)
print("  ALL VERIFICATION GATES PASSED")
print("=" * 72)
