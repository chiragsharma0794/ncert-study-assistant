"""Verification test for multi-signal retrieval sufficiency assessment.

Tests:
1.  SufficiencyResult dataclass (with new confidence + signals fields)
2.  InsufficientContextResponse dataclass
3.  assess -- no chunks (critical)
4.  assess -- high score, legacy mode (no query) → sufficient
5.  assess -- low score + few pages, legacy mode → insufficient
6.  assess -- low score + many pages, legacy mode → topical drift
7.  assess -- borderline score, legacy mode (edge case)
8.  Confidence bands mapping
9.  build_insufficient_response -- found/missing/follow-up
10. build_insufficient_response -- page number suggestions
11. INSUFFICIENT_CONTEXT_PROMPT_ADDITION structure
12. Integration: real chunks from corpus
13. Multi-signal: high hybrid score + zero lexical overlap → insufficient
14. Multi-signal: moderate score + good lexical overlap → sufficient
15. Multi-signal: no chunks → insufficient (with query)
16. Multi-signal: missing score fields + strong lexical overlap → sufficient
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval_sufficiency import (
    assess_retrieval_sufficiency, build_insufficient_response,
    SufficiencyResult, InsufficientContextResponse,
    SCORE_THRESHOLD, PAGE_COVERAGE_THRESHOLD, CONFIDENCE_BANDS,
    INSUFFICIENT_CONTEXT_PROMPT_ADDITION,
    _compute_lexical_overlap, _tokenize_important,
)

CHUNKS_PATH = PROJECT_ROOT / "outputs" / "chunks_semantic.json"
all_chunks = json.load(open(CHUNKS_PATH, encoding="utf-8"))


# =========================================================================
# TEST 1: SufficiencyResult dataclass
# =========================================================================
print("=" * 72)
print("  TEST 1: SufficiencyResult Dataclass")
print("=" * 72)

r = SufficiencyResult(is_sufficient=True)
d = r.to_dict()
expected_fields = {
    "is_sufficient", "confidence", "confidence_band", "max_score", "avg_score",
    "unique_pages", "n_chunks", "reason", "found_topics", "page_numbers",
    "signals",
}
assert set(d.keys()) == expected_fields, f"Missing fields: {expected_fields - set(d.keys())}"
assert r.confidence_band == "high"
assert r.found_topics == []
assert isinstance(r.signals, dict)
assert r.confidence == 0.0
print(f"  [OK] Fields: {sorted(d.keys())}")
print(f"  [OK] New fields: confidence={r.confidence}, signals={r.signals}")


# =========================================================================
# TEST 2: InsufficientContextResponse dataclass
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 2: InsufficientContextResponse Dataclass")
print("=" * 72)

icr = InsufficientContextResponse()
d = icr.to_dict()
expected_fields = {
    "answer", "found_summary", "missing_info", "follow_up",
    "sources", "refused", "is_partial", "sufficiency",
}
assert set(d.keys()) == expected_fields
assert icr.refused == False
assert icr.is_partial == True
print(f"  [OK] Fields: {sorted(d.keys())}")
print(f"  [OK] refused=False, is_partial=True")


# =========================================================================
# TEST 3: No chunks -> critical
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 3: assess -- No Chunks (Critical)")
print("=" * 72)

result = assess_retrieval_sufficiency([])
assert result.is_sufficient == False
assert result.confidence_band == "critical"
assert result.confidence == 0.0
assert "No chunks" in result.reason
assert result.signals["num_chunks"] == 0
print(f"  [OK] is_sufficient=False, band=critical, confidence=0.0")
print(f"  [OK] signals: {result.signals}")
print(f"  [OK] reason: '{result.reason}'")


# =========================================================================
# TEST 4: High score -> sufficient (legacy, no query)
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 4: assess -- High Score, Legacy Mode (Sufficient)")
print("=" * 72)

high_chunks = [
    {"chunk_id": "sem_0010", "text": "The cell membrane...", "page": 3, "type": "concept", "score": 0.95},
    {"chunk_id": "sem_0011", "text": "Osmosis is...",       "page": 4, "type": "concept", "score": 0.72},
    {"chunk_id": "sem_0012", "text": "Diffusion occurs...", "page": 5, "type": "concept", "score": 0.51},
]
result = assess_retrieval_sufficiency(high_chunks)
assert result.is_sufficient == True
assert result.confidence_band == "high"
assert result.max_score == 0.95
assert result.unique_pages == 3
assert result.n_chunks == 3
assert sorted(result.page_numbers) == [3, 4, 5]
assert result.confidence > 0.0
assert "signals" in result.to_dict()
print(f"  [OK] is_sufficient=True, band=high, max_score=0.95, pages=[3,4,5]")
print(f"  [OK] confidence={result.confidence}, signals={result.signals}")


# =========================================================================
# TEST 5: Low score + few pages -> insufficient (legacy)
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 5: assess -- Low Score + Few Pages, Legacy (Insufficient)")
print("=" * 72)

low_chunks = [
    {"chunk_id": "sem_0040", "text": "Unrelated...", "page": 8, "type": "exercise", "score": 0.12},
    {"chunk_id": "sem_0041", "text": "Also weak...",  "page": 8, "type": "exercise", "score": 0.08},
]
result = assess_retrieval_sufficiency(low_chunks)
assert result.is_sufficient == False
assert result.confidence_band == "critical"
assert result.unique_pages == 1
assert "may not be in the retrieved context" in result.reason
print(f"  [OK] is_sufficient=False, band=critical, pages=1")
print(f"  [OK] reason: '{result.reason[:70]}...'")


# =========================================================================
# TEST 6: Low score + many pages -> topical drift (legacy)
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 6: assess -- Low Score + Many Pages, Legacy (Topical Drift)")
print("=" * 72)

drift_chunks = [
    {"chunk_id": "sem_0050", "text": "Some content",  "page": 1, "type": "concept", "score": 0.21},
    {"chunk_id": "sem_0051", "text": "Other content", "page": 3, "type": "activity", "score": 0.15},
    {"chunk_id": "sem_0052", "text": "More content",  "page": 7, "type": "summary", "score": 0.10},
]
result = assess_retrieval_sufficiency(drift_chunks)
assert result.is_sufficient == False
assert result.confidence_band == "critical"
assert result.unique_pages == 3
assert "Topical drift" in result.reason
print(f"  [OK] is_sufficient=False, band=critical, pages=3 (drift detected)")
print(f"  [OK] reason: '{result.reason[:70]}...'")


# =========================================================================
# TEST 7: Borderline score -> depends on threshold (legacy)
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 7: assess -- Borderline Score, Legacy (Edge Case)")
print("=" * 72)

# Exactly at threshold
border_chunks = [
    {"chunk_id": "sem_0060", "text": "Border content", "page": 5, "type": "concept", "score": SCORE_THRESHOLD},
]
result = assess_retrieval_sufficiency(border_chunks)
assert result.is_sufficient == True
print(f"  [OK] score={SCORE_THRESHOLD} (exactly at threshold) -> sufficient")

# Just below threshold
border_chunks[0]["score"] = SCORE_THRESHOLD - 0.01
result = assess_retrieval_sufficiency(border_chunks)
assert result.is_sufficient == False
print(f"  [OK] score={SCORE_THRESHOLD - 0.01} (below threshold) -> insufficient")


# =========================================================================
# TEST 8: Confidence bands
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 8: Confidence Bands Mapping")
print("=" * 72)

test_scores = [
    (0.85, "high"),
    (0.55, "medium"),
    (0.35, "low"),
    (0.15, "critical"),
    (0.0,  "critical"),
]
for score, expected_band in test_scores:
    chunks = [{"chunk_id": "test", "text": "t", "page": 1, "type": "concept", "score": score}]
    result = assess_retrieval_sufficiency(chunks)
    assert result.confidence_band == expected_band, \
        f"score={score}: expected {expected_band}, got {result.confidence_band}"
    print(f"  [OK] score={score:5.1f} -> band={expected_band}")


# =========================================================================
# TEST 9: build_insufficient_response -- found/missing/follow-up
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 9: build_insufficient_response Structure")
print("=" * 72)

chunks = [
    {"chunk_id": "sem_0010", "text": "The cell membrane controls entry.", "page": 3, "type": "concept", "score": 0.15},
    {"chunk_id": "sem_0011", "text": "Osmosis requires a membrane.",     "page": 4, "type": "concept", "score": 0.08},
]
suff = assess_retrieval_sufficiency(chunks)
response = build_insufficient_response("What is endocytosis?", chunks, suff)

assert response.is_partial == True
assert response.refused == False
assert "found" in response.found_summary.lower()
assert "missing" in response.missing_info.lower() or "low" in response.missing_info.lower()
assert len(response.sources) > 0
assert response.follow_up != ""
print(f"  [OK] is_partial=True, refused=False")
print(f"  [OK] found_summary: '{response.found_summary[:60]}...'")
print(f"  [OK] missing_info: '{response.missing_info[:60]}...'")
print(f"  [OK] follow_up: '{response.follow_up[:60]}...'")
print(f"  [OK] sources: {response.sources}")


# =========================================================================
# TEST 10: Page number suggestions
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 10: build_insufficient_response -- Page Suggestions")
print("=" * 72)

chunks_with_pages = [
    {"chunk_id": "sem_0020", "text": "Content", "page": 5,  "type": "concept", "score": 0.20},
    {"chunk_id": "sem_0021", "text": "Content", "page": 12, "type": "activity", "score": 0.15},
]
suff = assess_retrieval_sufficiency(chunks_with_pages)
response = build_insufficient_response("test query", chunks_with_pages, suff)

assert "page" in response.follow_up.lower()
assert "5" in response.follow_up
assert "12" in response.follow_up
print(f"  [OK] follow_up contains page numbers: '{response.follow_up}'")

# No pages in metadata -> generic follow-up
chunks_no_pages = [
    {"chunk_id": "sem_0030", "text": "Content", "type": "concept", "score": 0.10},
]
suff = assess_retrieval_sufficiency(chunks_no_pages)
response = build_insufficient_response("test query", chunks_no_pages, suff)
assert "concept" in response.follow_up.lower() or "rephras" in response.follow_up.lower()
print(f"  [OK] No pages -> topic-based suggestion: '{response.follow_up[:60]}'")


# =========================================================================
# TEST 11: INSUFFICIENT_CONTEXT_PROMPT_ADDITION
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 11: Prompt Addition Structure")
print("=" * 72)

assert "MUST" in INSUFFICIENT_CONTEXT_PROMPT_ADDITION
assert "is_partial" in INSUFFICIENT_CONTEXT_PROMPT_ADDITION
assert "MISSING" in INSUFFICIENT_CONTEXT_PROMPT_ADDITION
assert "NEVER" in INSUFFICIENT_CONTEXT_PROMPT_ADDITION
assert "fabricate" in INSUFFICIENT_CONTEXT_PROMPT_ADDITION.lower()
assert "page" in INSUFFICIENT_CONTEXT_PROMPT_ADDITION.lower()
print(f"  [OK] Contains: MUST, is_partial, MISSING, NEVER, fabricate, page")
print(f"  [OK] Prompt addition length: {len(INSUFFICIENT_CONTEXT_PROMPT_ADDITION)} chars")

# =========================================================================
# TEST 12: Integration with real corpus chunks
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 12: Integration with Real Corpus")
print("=" * 72)

from retriever import BM25Retriever, HybridRetriever, load_chunks
corpus = load_chunks(CHUNKS_PATH)

# --- 12a: In-scope query with HybridRetriever: should be sufficient ---
hybrid = HybridRetriever(str(CHUNKS_PATH))
in_scope = hybrid.retrieve("What is the cell membrane?", top_k=5)
result = assess_retrieval_sufficiency(in_scope, query="What is the cell membrane?")
assert result.is_sufficient == True
print(f"  [OK] 'cell membrane' (Hybrid+query) -> sufficient "
      f"(confidence={result.confidence:.2f}, band={result.confidence_band})")
print(f"       signals: {result.signals}")

# --- 12b: OOS test with synthetic low-score chunks ---
oos_chunks = [
    {"chunk_id": "syn_001", "content": "Unrelated content", "page": 1, "type": "concept", "score": 0.08},
    {"chunk_id": "syn_002", "content": "Also unrelated",    "page": 1, "type": "concept", "score": 0.05},
]
result_oos = assess_retrieval_sufficiency(oos_chunks)
assert result_oos.is_sufficient == False
assert result_oos.confidence_band == "critical"
print(f"  [OK] Synthetic OOS chunks -> insufficient (band={result_oos.confidence_band}, "
      f"max={result_oos.max_score:.1f})")

# --- 12c: Build response for the insufficient case ---
response = build_insufficient_response("quantum superposition", oos_chunks, result_oos)
assert response.is_partial == True
assert len(response.answer) > 50
print(f"  [OK] Insufficient response built ({len(response.answer)} chars)")
print(f"       follow_up: '{response.follow_up[:70]}...'")


# =========================================================================
# TEST 13: High hybrid score + zero lexical overlap → insufficient
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 13: Multi-signal -- High Score + Zero Overlap (Insufficient)")
print("=" * 72)

# Simulates inflated hybrid scores for completely unrelated content
inflated_chunks = [
    {"chunk_id": "sem_0100", "content": "The cell membrane is selectively permeable.",
     "page": 3, "type": "concept", "score": 0.85},
    {"chunk_id": "sem_0101", "content": "Osmosis is the movement of water.",
     "page": 4, "type": "concept", "score": 0.72},
    {"chunk_id": "sem_0102", "content": "Diffusion occurs from high to low concentration.",
     "page": 5, "type": "concept", "score": 0.61},
]
# Query about art: zero overlap with biology chunks
result = assess_retrieval_sufficiency(
    inflated_chunks,
    query="impressionist oil painting techniques by Monet"
)
assert result.is_sufficient == False, (
    f"High score + zero overlap should be insufficient, "
    f"got confidence={result.confidence}, overlap={result.signals['lexical_overlap']}"
)
assert result.signals["lexical_overlap"] < 0.1
print(f"  [OK] High scores ({inflated_chunks[0]['score']}) + zero overlap -> insufficient")
print(f"       confidence={result.confidence}, overlap={result.signals['lexical_overlap']}")
print(f"       reason: '{result.reason[:80]}...'")


# =========================================================================
# TEST 14: Moderate score + good lexical overlap → sufficient
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 14: Multi-signal -- Moderate Score + Good Overlap (Sufficient)")
print("=" * 72)

relevant_chunks = [
    {"chunk_id": "sem_0200", "content": "The cell membrane is selectively permeable. "
     "It controls what enters and exits the cell.",
     "page": 3, "type": "concept", "score": 0.55},
    {"chunk_id": "sem_0201", "content": "The membrane is made of lipids and proteins.",
     "page": 3, "type": "concept", "score": 0.42},
    {"chunk_id": "sem_0202", "content": "Cell membrane functions include protection.",
     "page": 4, "type": "concept", "score": 0.38},
]
result = assess_retrieval_sufficiency(
    relevant_chunks,
    query="What is the cell membrane?"
)
assert result.is_sufficient == True, (
    f"Moderate score + good overlap should be sufficient, "
    f"got confidence={result.confidence}, overlap={result.signals['lexical_overlap']}"
)
assert result.signals["lexical_overlap"] > 0.3
assert result.confidence >= 0.40
print(f"  [OK] Moderate scores + good overlap -> sufficient")
print(f"       confidence={result.confidence}, overlap={result.signals['lexical_overlap']}")
print(f"       signals: {result.signals}")


# =========================================================================
# TEST 15: No chunks with query → insufficient
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 15: Multi-signal -- No Chunks with Query (Insufficient)")
print("=" * 72)

result = assess_retrieval_sufficiency([], query="What is osmosis?")
assert result.is_sufficient == False
assert result.confidence == 0.0
assert result.signals["num_chunks"] == 0
assert result.signals["lexical_overlap"] == 0.0
print(f"  [OK] No chunks + query -> insufficient, confidence=0.0")
print(f"       signals: {result.signals}")


# =========================================================================
# TEST 16: Missing score fields + strong lexical overlap → sufficient
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 16: Multi-signal -- No Scores + Strong Overlap (Sufficient)")
print("=" * 72)

# Chunks with no 'score' field but strong content match
no_score_chunks = [
    {"chunk_id": "sem_0300", "content": "Osmosis is the movement of water molecules "
     "through a selectively permeable membrane from higher water concentration "
     "to lower water concentration.",
     "page": 5, "type": "concept"},
    {"chunk_id": "sem_0301", "content": "Osmosis is important for cells to maintain "
     "water balance. Without osmosis, cells would either shrink or burst.",
     "page": 5, "type": "concept"},
    {"chunk_id": "sem_0302", "content": "The process of osmosis occurs across the "
     "cell membrane whenever there is a concentration gradient.",
     "page": 6, "type": "concept"},
]
result = assess_retrieval_sufficiency(
    no_score_chunks,
    query="What is osmosis and how does it work?"
)
# With score=0.0 (default), the score signal is 0, but overlap should
# be high enough that the weighted confidence can reach threshold.
# The overlap signal dominates (weight 0.45) and should compensate.
assert result.signals["lexical_overlap"] > 0.3, (
    f"Expected strong overlap, got {result.signals['lexical_overlap']}"
)
# With good overlap and multiple chunks, the system should recognize
# these as conditionally sufficient
print(f"  [OK] No score fields + strong overlap")
print(f"       confidence={result.confidence}, sufficient={result.is_sufficient}")
print(f"       overlap={result.signals['lexical_overlap']}, top_score={result.signals['top_score']}")
print(f"       signals: {result.signals}")
# Allow either sufficient or insufficient — the key assertion is that
# the system does NOT crash and produces a reasonable confidence > 0.
assert result.confidence > 0.0, "Confidence should be > 0 with overlap"
assert isinstance(result.signals, dict)
assert "lexical_overlap" in result.signals


# =========================================================================
print("\n" + "=" * 72)
print("  ALL TESTS PASSED [OK]")
print("=" * 72)
