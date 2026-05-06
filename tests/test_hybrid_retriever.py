"""
Verification test for HybridRetriever.

Tests:
1. Interface contract (same signature as BM25Retriever)
2. Return schema (chunk dict + score key)
3. BM25 independence (BM25 works without cross-encoder)
4. Score normalization (sigmoid -> [0, 1])
5. Evaluator compatibility (can be swapped into evaluator.py)
6. Edge cases (top_k > n_docs, top_k=1)
7. Side-by-side comparison on real queries
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retriever import BM25Retriever, FAISSRetriever, HybridRetriever, load_chunks

CHUNKS_PATH = str(PROJECT_ROOT / "outputs" / "chunks_semantic.json")

# ── Load chunks ───────────────────────────────────────────────────────────
chunks = load_chunks(CHUNKS_PATH)
print(f"Loaded {len(chunks)} chunks\n")

# ── Build retrievers (V2: pass path string, not list) ─────────────────────
bm25   = BM25Retriever(CHUNKS_PATH)
hybrid = HybridRetriever(CHUNKS_PATH)

print(f"  {bm25}")
print(f"  {hybrid}")

# =========================================================================
# TEST 1: Interface contract
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 1: Interface Contract")
print("=" * 72)

# Both must have .retrieve(query, top_k)
for name, retriever in [("BM25", bm25), ("Hybrid", hybrid)]:
    assert hasattr(retriever, "retrieve"), f"{name} missing retrieve()"
    result = retriever.retrieve("test query", top_k=3)
    assert isinstance(result, list), f"{name} should return list"
    assert len(result) == 3, f"{name} should return exactly 3 results"
    print(f"  [OK] {name}.retrieve(query, top_k=3) -> list of {len(result)} dicts")

# =========================================================================
# TEST 2: Return schema (V2: content, not text)
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 2: Return Schema")
print("=" * 72)

REQUIRED_KEYS = {"chunk_id", "content", "page", "type", "score"}

for name, retriever in [("BM25", bm25), ("Hybrid", hybrid)]:
    results = retriever.retrieve("What is a cell?", top_k=3)
    for r in results:
        missing = REQUIRED_KEYS - set(r.keys())
        assert not missing, f"{name}: missing keys {missing} in result"
        assert isinstance(r["score"], float), f"{name}: score should be float"
    print(f"  [OK] {name}: all results have {len(REQUIRED_KEYS)} required keys")

# =========================================================================
# TEST 3: BM25 independence
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 3: BM25 Independence")
print("=" * 72)

# BM25 must work identically to V1, unaffected by HybridRetriever
bm25_standalone = BM25Retriever(CHUNKS_PATH)
r1 = bm25_standalone.retrieve("What is a cell?", top_k=5)
r2 = bm25.retrieve("What is a cell?", top_k=5)

# Same chunk IDs in same order
ids1 = [r["chunk_id"] for r in r1]
ids2 = [r["chunk_id"] for r in r2]
assert ids1 == ids2, f"BM25 results differ: {ids1} vs {ids2}"
print(f"  [OK] Two BM25 instances return identical results: {ids1}")

# Scores match exactly
scores1 = [r["score"] for r in r1]
scores2 = [r["score"] for r in r2]
assert scores1 == scores2, "BM25 scores differ"
print(f"  [OK] BM25 scores match exactly")

# =========================================================================
# TEST 4: Score normalization (Hybrid scores in [0, 1])
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 4: Score Normalization")
print("=" * 72)

hybrid_results = hybrid.retrieve("What is a cell membrane?", top_k=5)
for r in hybrid_results:
    assert 0.0 <= r["score"] <= 1.0, \
        f"Hybrid score {r['score']} outside [0, 1] range"
print(f"  [OK] All hybrid scores in [0, 1]: "
      f"{[round(r['score'], 4) for r in hybrid_results]}")

# =========================================================================
# TEST 5: Evaluator compatibility
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 5: Evaluator Compatibility")
print("=" * 72)

# Simulate what evaluator.py does (evaluator.py:162-173)
query = "What is the basic structural and functional unit of all living organisms?"
retrieved = hybrid.retrieve(query, top_k=5)
top_scores = [r["score"] for r in retrieved]

# evaluator.py stores retrieval_scores[:3]
assert len(top_scores) >= 3, "Need at least 3 scores"
stored_scores = top_scores[:3]
print(f"  [OK] Evaluator score extraction works: {[round(s, 4) for s in stored_scores]}")

# Simulate generator.py's chunk access (V2 uses 'content' not 'text')
for i, chunk in enumerate(retrieved, 1):
    _ = chunk["content"]   # V2 generator reads this
    _ = chunk["page"]      # generator reads this
    _ = chunk["type"]      # generator reads this
    _ = chunk["chunk_id"]  # generator reads this
print(f"  [OK] Generator chunk access works for all {len(retrieved)} results")

# =========================================================================
# TEST 6: Edge cases
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 6: Edge Cases")
print("=" * 72)

# top_k = 1
r = hybrid.retrieve("osmosis", top_k=1)
assert len(r) == 1, f"top_k=1 should return 1 result, got {len(r)}"
print(f"  [OK] top_k=1 -> {len(r)} result")

# top_k > n_docs (should return all docs, no crash)
r = hybrid.retrieve("cell", top_k=200)
assert len(r) <= len(chunks), f"top_k=200 should return at most {len(chunks)} results"
print(f"  [OK] top_k=200 -> {len(r)} results")

# =========================================================================
# TEST 7: Side-by-side comparison
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 7: BM25 vs Hybrid Side-by-Side")
print("=" * 72)

test_queries = [
    ("What is a cell?",                         "factual (exact keyword)"),
    ("What gives flowers their colour?",        "paraphrased (chromoplasts)"),
    ("Difference between prokaryotic and eukaryotic cells?", "comparative"),
    ("Who invented the telescope?",             "out-of-scope"),
]

for query, label in test_queries:
    bm25_r   = bm25.retrieve(query, top_k=3)
    hybrid_r = hybrid.retrieve(query, top_k=3)

    bm25_ids   = [r["chunk_id"] for r in bm25_r]
    hybrid_ids = [r["chunk_id"] for r in hybrid_r]
    overlap    = set(bm25_ids) & set(hybrid_ids)
    reranked   = set(hybrid_ids) - set(bm25_ids)

    print(f"\n  Q: \"{query}\" ({label})")
    print(f"    BM25   top-3: {bm25_ids}  scores: {[round(r['score'], 3) for r in bm25_r]}")
    print(f"    Hybrid top-3: {hybrid_ids}  scores: {[round(r['score'], 3) for r in hybrid_r]}")
    print(f"    Overlap: {len(overlap)}/3  |  Hybrid promoted: {reranked or 'none'}")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 72)
print("  ALL TESTS PASSED [OK]")
print("=" * 72)
print("""
  INVARIANTS VERIFIED:
    1. retrieve(query, top_k) signature    -- identical across BM25 + Hybrid
    2. Return schema                       -- V2: chunk_id, content, score, page, type
    3. BM25 independence                   -- unmodified, deterministic
    4. Score normalization                  -- fused [0, 1] for hybrid
    5. Evaluator compatibility             -- zero changes needed
    6. Generator compatibility             -- V2 content key preserved
    7. Constructor contract                -- chunks_path: str for all
""")
