# %% [markdown]
# # Phase 3: Retrieval Engine
#
# **Goal**: Build and compare two sparse retrieval engines (BM25 and TF-IDF),
# test them on real student queries, and select the one that goes forward.
#
# **Input**: `outputs/chunks_semantic.json` (85 semantic chunks)
# **Module**: `src/retriever.py` (BM25Retriever, TFIDFRetriever)
#
# ---

# %% Cell 1 -- Imports & Setup
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

# Add src/ to the import path so we can import our retriever module
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retriever import BM25Retriever, TFIDFRetriever, load_chunks

CHUNKS_PATH = PROJECT_ROOT / "outputs" / "chunks_semantic.json"
print(f"Project root : {PROJECT_ROOT}")
print(f"Chunks path  : {CHUNKS_PATH}")

# %% Cell 2 -- Load Chunks & Build Both Retrievers
chunks = load_chunks(CHUNKS_PATH)
print(f"Loaded {len(chunks)} semantic chunks")

bm25_retriever  = BM25Retriever(chunks)
tfidf_retriever = TFIDFRetriever(chunks)

print(f"\n{bm25_retriever}")
print(f"{tfidf_retriever}")


# %% Cell 3 -- Test Queries: BM25 Retriever
"""
We test 3 queries that a Class 9 student might actually ask.
These cover different retrieval challenges:

  Q1: "What is a cell?"         -- broad, definitional
  Q2: "What is cell membrane?"  -- specific structure
  Q3: "Difference between mitosis and meiosis"
      -- comparative, requires matching BOTH terms

Note: The original task mentioned "mixture" and "colloid" queries,
but our PDF is about cells (Ch 2: "Cell: The Building Block of Life"),
so we use queries that match this chapter's content.
"""

TEST_QUERIES = [
    "What is a cell?",
    "What is cell membrane?",
    "Difference between mitosis and meiosis",
]


def print_results(query: str, results: list[dict], label: str = "BM25") -> None:
    """Pretty-print retrieval results for a query."""
    print(f"\n  Query: \"{query}\"")
    print(f"  {'Rank':<6} {'ID':<12} {'Score':>7} {'Type':<10} First 100 chars")
    print(f"  {'-'*6} {'-'*12} {'-'*7} {'-'*10} {'-'*45}")
    for rank, r in enumerate(results, 1):
        preview = r["text"][:100].replace("\n", " | ")
        print(f"  {rank:<6} {r['chunk_id']:<12} {r['score']:>7.3f} {r['type']:<10} {preview}...")


print("=" * 78)
print("  BM25 RETRIEVAL RESULTS (top 3)")
print("=" * 78)

for q in TEST_QUERIES:
    results = bm25_retriever.retrieve(q, top_k=3)
    print_results(q, results, "BM25")

print()


# %% Cell 4 -- Test Queries: TF-IDF Retriever
print("=" * 78)
print("  TF-IDF RETRIEVAL RESULTS (top 3)")
print("=" * 78)

for q in TEST_QUERIES:
    results = tfidf_retriever.retrieve(q, top_k=3)
    print_results(q, results, "TF-IDF")

print()


# %% Cell 5 -- Head-to-Head Comparison
"""
Side-by-side comparison: for each query, show BM25 vs TF-IDF
top-3 chunk IDs and scores.  Highlight when they disagree.
"""

print("=" * 78)
print("  HEAD-TO-HEAD: BM25 vs TF-IDF")
print("=" * 78)

for q in TEST_QUERIES:
    bm25_results  = bm25_retriever.retrieve(q, top_k=3)
    tfidf_results = tfidf_retriever.retrieve(q, top_k=3)

    bm25_ids  = [r["chunk_id"] for r in bm25_results]
    tfidf_ids = [r["chunk_id"] for r in tfidf_results]

    overlap = set(bm25_ids) & set(tfidf_ids)
    only_bm25  = set(bm25_ids) - set(tfidf_ids)
    only_tfidf = set(tfidf_ids) - set(bm25_ids)

    print(f"\n  Query: \"{q}\"")
    print(f"  {'Rank':<6} {'BM25 ID':<14} {'BM25 Score':>10}   {'TF-IDF ID':<14} {'TF-IDF Score':>12}")
    print(f"  {'-'*6} {'-'*14} {'-'*10}   {'-'*14} {'-'*12}")

    for rank in range(3):
        b = bm25_results[rank]
        t = tfidf_results[rank]
        # Mark differences
        b_marker = " *" if b["chunk_id"] in only_bm25 else ""
        t_marker = " *" if t["chunk_id"] in only_tfidf else ""
        print(f"  {rank+1:<6} {b['chunk_id']:<14} {b['score']:>10.3f}   "
              f"{t['chunk_id']:<14} {t['score']:>12.3f}")

    agreement = len(overlap)
    print(f"  Agreement: {agreement}/3 chunks overlap  |  "
          f"BM25-only: {only_bm25 or 'none'}  |  TF-IDF-only: {only_tfidf or 'none'}")


# %% Cell 6 -- Retrieval Quality Spot-Check
"""
Let's look at a query where retrieval might FAIL to find the
right answer.  This teaches us what retrieval failures look like.
"""

print("\n" + "=" * 78)
print("  RETRIEVAL EDGE CASES")
print("=" * 78)

edge_queries = [
    "What is osmosis?",                    # may or may not be in this chapter
    "Who discovered the cell?",            # historical fact, needs exact match
    "Why do plant cells have a cell wall?", # reasoning, not just keywords
]

for q in edge_queries:
    bm25_top = bm25_retriever.retrieve(q, top_k=1)[0]
    print(f"\n  Q: \"{q}\"")
    print(f"  Best match: {bm25_top['chunk_id']} (score={bm25_top['score']:.3f})")
    preview = bm25_top["text"][:120].replace("\n", " | ")
    print(f"  Text: {preview}...")

    # Flag low-confidence results
    if bm25_top["score"] < 1.0:
        print(f"  [!] LOW CONFIDENCE -- score < 1.0, retrieval may have failed")


# %% [markdown]
# ---
# ## Mentor Notes: How Sparse Retrieval Works
#
# ---
#
# ### 1. How BM25 Works (IDF, TF, Length Normalization)
#
# BM25 (Best Matching 25) scores each document against a query using
# three intuitions:
#
# ```
#   BM25(q, d) = SUM over each query term t:
#
#                  IDF(t) * TF(t,d) * (k1 + 1)
#                 --------------------------------
#                  TF(t,d) + k1 * (1 - b + b * |d|/avgdl)
# ```
#
# **IDF (Inverse Document Frequency)**:
# How *rare* is this term across all documents?
# - "the" appears everywhere -> IDF ~ 0 (useless for ranking)
# - "mitosis" appears in 3 chunks -> IDF is HIGH (very informative)
#
# **TF (Term Frequency)**:
# How many times does the term appear in *this specific document*?
# - But with diminishing returns: 5 mentions isn't 5x better than 1
# - The (k1 + 1) / (TF + k1...) formula creates this saturation
#
# **Length normalization (b parameter)**:
# Long documents naturally contain more term occurrences.
# - b=1.0: fully penalize long documents
# - b=0.0: ignore document length
# - b=0.75 (default): moderate normalization
#
# **Intuition**: BM25 asks: "Does this chunk contain rare, query-specific
# terms? And is it focused (not just long and rambling)?"
#
# ---
#
# ### 2. Why BM25 Beats TF-IDF for Keyword-Heavy Factual Queries
#
# TF-IDF and BM25 both use IDF weighting.  The difference:
#
# | Feature                    | TF-IDF              | BM25                  |
# |----------------------------|---------------------|-----------------------|
# | TF scaling                 | Linear (or log)     | Saturating (diminishing returns) |
# | Length normalization        | Cosine norm only    | Explicit `b` parameter |
# | Term frequency saturation  | No                  | Yes (via `k1`)         |
#
# For factual queries like "What is a cell membrane?":
# - A chunk that mentions "membrane" 10 times isn't 10x more relevant
#   than one that mentions it 2 times with a clear definition
# - BM25's saturation handles this; TF-IDF doesn't
# - BM25's length norm penalizes chunks that are long but unfocused
#
# **Result**: BM25 almost always outperforms TF-IDF on factual QA benchmarks.
#
# ---
#
# ### 3. What "Sparse Retrieval" Means vs "Dense Retrieval"
#
# | Aspect            | Sparse (BM25, TF-IDF)           | Dense (embedding-based)          |
# |-------------------|---------------------------------|----------------------------------|
# | Representation    | High-dim vector, mostly zeros   | Low-dim dense vector (384-768d)  |
# | Matching          | Exact keyword overlap           | Semantic similarity              |
# | Vocabulary        | Must share exact words          | Understands synonyms, paraphrase |
# | Speed             | Very fast (inverted index)      | Needs ANN index (FAISS, etc.)    |
# | Training needed?  | No                              | Yes (pre-trained model)          |
#
# **Sparse retrieval** represents documents as sparse vectors where each
# dimension is a vocabulary term.  Matching requires *exact keyword overlap*:
# if the query says "osmosis" but the chunk says "diffusion through a
# membrane," sparse retrieval will miss it.
#
# **Dense retrieval** (Phase 4 preview) encodes text into semantic embeddings.
# "osmosis" and "diffusion through a membrane" would have similar vectors.
#
# **Best practice in production**: Use BOTH (hybrid retrieval).
# Sparse catches exact keyword matches; dense catches semantic matches.
#
# ---
#
# ### 4. Why `top_k=5` is a Design Choice, Not a Magic Number
#
# `top_k` controls how many chunks you feed to the LLM:
#
# | top_k | Pros                           | Cons                           |
# |-------|--------------------------------|--------------------------------|
# | 1     | Minimal noise, cheap           | Miss relevant info in other chunks |
# | 3     | Good balance for focused Q&A   | Might miss for broad topics    |
# | 5     | Covers most factual questions  | Starts including noise         |
# | 10+   | Maximum recall                 | Expensive, noise dominates     |
#
# The right value depends on:
# - **Chunk granularity**: Small chunks -> need higher k to get full context
# - **Query type**: Factual ("What is X?") needs k=3; comparative
#   ("Compare A and B") needs k=5-7
# - **LLM context window**: More chunks = more tokens = higher cost
# - **Your evaluation metrics**: Measure recall@k and tune empirically
#
# We use k=5 as a starting default.  In Phase 5 (Evaluation), we'll
# measure whether k=3 or k=7 gives better answers.
#
# ---
#
# ### 5. What a "Retrieval Failure" Looks Like and Why It Happens
#
# A retrieval failure = the correct chunk is NOT in the top-k results.
# The LLM then either hallucinates or says "I don't know."
#
# **Common causes**:
#
# 1. **Vocabulary mismatch**: Query says "osmosis", chunk says
#    "movement of water through a semi-permeable membrane" -- no shared
#    keywords for BM25 to match on.
#
# 2. **Information buried in a long chunk**: The answer is one sentence
#    in a 200-token chunk about a different topic.  BM25 scores the
#    chunk low because most terms don't match.
#
# 3. **Answer spans multiple chunks**: "Compare mitosis and meiosis"
#    needs info from 2 separate chunks.  Each chunk alone scores
#    lower than a chunk that mentions both terms.
#
# 4. **The answer isn't in the corpus**: The student asks about
#    something from a different chapter.  No retriever can fix this.
#
# **How to diagnose**: Check if the correct chunk appears at rank 6-20
# (retriever found it but k was too low) vs. rank 50+ (genuine failure).
# The fix is usually: better chunking, hybrid retrieval, or query expansion.
