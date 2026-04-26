# %% [markdown]
# # Phase 4: Grounded Answer Generation
#
# **Goal**: Wire the BM25 retriever to Google Gemini so the LLM generates
# answers grounded ONLY in retrieved textbook chunks.
#
# **Pipeline**: Query -> BM25 Retrieve (top 5) -> Build Prompt -> Gemini -> Answer + Sources
#
# **Core rule**: If the context doesn't contain the answer, REFUSE politely.
#
# ---

# %% Cell 1 -- Imports & Setup
import sys
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

# Load .env from project root
load_dotenv(PROJECT_ROOT / ".env")

# Add src/ to import path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retriever import BM25Retriever, load_chunks
from generator import GroundedGenerator

CHUNKS_PATH = PROJECT_ROOT / "outputs" / "chunks_semantic.json"
print(f"Project root : {PROJECT_ROOT}")

# Check API key
api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    print("\n[!] GEMINI_API_KEY not found in .env")
    print("    Create a .env file in the project root with:")
    print("    GEMINI_API_KEY=your-key-here")
    print("\n    Get a free key at: https://aistudio.google.com/apikey")
    print("\n    Running in DRY-RUN mode (prompt preview only)\n")
    DRY_RUN = True
else:
    print(f"[OK] API key loaded ({len(api_key)} chars)")
    DRY_RUN = False


# %% Cell 2 -- Build Retriever & Generator
chunks = load_chunks(CHUNKS_PATH)
print(f"Loaded {len(chunks)} chunks")

retriever = BM25Retriever(chunks)
print(f"{retriever}")

if not DRY_RUN:
    generator = GroundedGenerator(api_key=api_key, model_name="gemini-2.0-flash")
    print(f"{generator}")
else:
    generator = None
    print("Generator: DRY-RUN (no API key)")


# %% Cell 3 -- Full RAG Pipeline Function
def run_rag_pipeline(query: str,
                     retriever: BM25Retriever,
                     generator: GroundedGenerator | None,
                     top_k: int = 5,
                     dry_run: bool = False) -> dict:
    """Run the full RAG pipeline: retrieve -> generate -> display.

    Parameters
    ----------
    query : str
        The student's question.
    retriever : BM25Retriever
        The retriever to use.
    generator : GroundedGenerator or None
        The generator. If None, runs in dry-run mode (prompt only).
    top_k : int
        Number of chunks to retrieve.
    dry_run : bool
        If True, print the prompt but don't call the API.

    Returns
    -------
    dict
        The generation result, or a dry-run placeholder.
    """
    # Step 1: Retrieve
    retrieved = retriever.retrieve(query, top_k=top_k)

    print(f"\n{'='*70}")
    print(f"  QUERY: \"{query}\"")
    print(f"{'='*70}")

    print(f"\n  Retrieved {len(retrieved)} chunks:")
    for i, r in enumerate(retrieved, 1):
        preview = r["text"][:80].replace("\n", " | ")
        print(f"    [{i}] {r['chunk_id']} (score={r['score']:.2f}, "
              f"type={r['type']}) {preview}...")

    # Step 2: Generate (or dry-run)
    if dry_run or generator is None:
        # Show the prompt that WOULD be sent
        from generator import GroundedGenerator as _GG
        temp_gen = object.__new__(_GG)
        temp_gen.model_name = "dry-run"
        prompt = temp_gen.build_prompt(query, retrieved)
        print(f"\n  --- DRY-RUN: Prompt ({len(prompt)} chars) ---")
        # Show first 500 chars of prompt
        print(f"  {prompt[:500]}...")
        return {
            "answer": "[DRY-RUN] No API call made.",
            "sources": [],
            "refused": False,
            "model": "dry-run",
        }

    result = generator.generate(query, retrieved)

    # Step 3: Display
    print(f"\n  ANSWER ({result['model']}):")
    print(f"  {'~'*60}")

    # Indent the answer for readability
    for line in result["answer"].split("\n"):
        print(f"    {line}")

    print(f"  {'~'*60}")
    print(f"  Sources cited : {result['sources'] or 'none'}")
    print(f"  Refused       : {result['refused']}")

    return result


# %% Cell 4 -- Test: In-Scope Query
"""
Test 1: A question directly answerable from the textbook.
The chapter is about cells, so we ask about cell structure.
"""
result_1 = run_rag_pipeline(
    "What is a cell and what are its main components?",
    retriever, generator, top_k=5, dry_run=DRY_RUN,
)


# %% Cell 5 -- Test: Paraphrased Query
"""
Test 2: A question phrased differently from the textbook wording.
Tests whether BM25 can handle paraphrasing (spoiler: barely).
"""
result_2 = run_rag_pipeline(
    "How is a prokaryotic cell different from a eukaryotic cell?",
    retriever, generator, top_k=5, dry_run=DRY_RUN,
)


# %% Cell 6 -- Test: Out-of-Scope Query
"""
Test 3: A question NOT covered in this chapter.
The model MUST refuse to answer -- this tests grounding.
"""
result_3 = run_rag_pipeline(
    "Who invented the telescope?",
    retriever, generator, top_k=5, dry_run=DRY_RUN,
)


# %% Cell 7 -- Pipeline Summary
print("\n" + "=" * 70)
print("  RAG PIPELINE SUMMARY")
print("=" * 70)

test_cases = [
    ("In-scope",     "What is a cell and what are its main components?", result_1),
    ("Paraphrased",  "How is a prokaryotic cell different from a eukaryotic cell?", result_2),
    ("Out-of-scope", "Who invented the telescope?", result_3),
]

print(f"\n  {'Test Case':<14} {'Refused':>8} {'Sources':>8} {'Answer Preview'}")
print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*40}")
for label, query, result in test_cases:
    preview = result["answer"][:50].replace("\n", " ")
    n_sources = len(result["sources"])
    print(f"  {label:<14} {str(result['refused']):>8} {n_sources:>8} {preview}...")

if not DRY_RUN:
    print("\n  [OK] All 3 test cases executed successfully.")
    if result_3["refused"]:
        print("  [OK] Out-of-scope query was correctly REFUSED.")
    else:
        print("  [!] WARNING: Out-of-scope query was NOT refused -- check grounding prompt.")
else:
    print("\n  [!] Running in DRY-RUN mode. Set GEMINI_API_KEY in .env to test generation.")


# %% [markdown]
# ---
# ## Mentor Notes: Grounded Generation Deep-Dive
#
# ---
#
# ### 1. Why "Grounding Prompts" Prevent Hallucination
#
# A **grounding prompt** tells the LLM: "Only use the information I give you.
# If it's not there, say so."
#
# Without grounding:
# ```
# Q: "What is a cell wall?"
# A: "A cell wall is a rigid layer found outside the cell membrane
#     in plant cells, fungi, and bacteria. It is made of cellulose
#     in plants..." (may be correct, but unverifiable)
# ```
#
# With grounding:
# ```
# Q: "What is a cell wall?"
# Context: [1] "The cell wall is present in plant cells...provides
#           structural support..."
# A: "According to the textbook [1], the cell wall is present in
#     plant cells and provides structural support."
# ```
#
# The grounded answer is:
# - **Verifiable**: you can check chunk [1] to confirm
# - **Bounded**: can't introduce facts not in the textbook
# - **Traceable**: student sees exactly where the info came from
#
# Grounding doesn't make hallucination *impossible*, but it reduces it
# dramatically because the model has explicit source text to copy from
# rather than relying on parametric memory.
#
# ---
#
# ### 2. What `temperature=0` Does Mathematically
#
# LLMs generate text by predicting the next token from a probability
# distribution over the vocabulary.  **Temperature** scales these
# probabilities before sampling:
#
# ```
#   P(token_i) = exp(logit_i / T) / SUM(exp(logit_j / T))
# ```
#
# | Temperature | Effect                                        |
# |-------------|-----------------------------------------------|
# | T = 0       | Always pick the highest-probability token (greedy/deterministic) |
# | T = 0.3     | Mostly deterministic, slight variation         |
# | T = 1.0     | Standard sampling (default)                    |
# | T > 1.0     | More random, creative, unpredictable           |
#
# **For RAG, always use T=0** because:
# - We want the *most likely* factual answer, not creative prose
# - Reproducibility: same query + same context = same answer
# - Lower hallucination risk (less sampling randomness)
#
# ---
#
# ### 3. Why You MUST Label Retrieved Chunks [1], [2] in the Prompt
#
# Labeling chunks serves three purposes:
#
# **a) Citation**: The model can say "According to [2]..." which lets
# the student verify the answer against a specific source.
#
# **b) Disambiguation**: When chunks contain contradictory or overlapping
# info, labels let the model distinguish between them:
# "[1] says X, while [3] adds Y."
#
# **c) Attention guidance**: Transformer attention benefits from
# structural markers.  Labels act as anchors that help the model
# *locate* relevant parts of the context more precisely.
#
# Without labels, the model sees one big wall of text and can't
# reference specific parts in its answer.
#
# ---
#
# ### 4. The Difference Between "Retrieval Failure" and "Generation Failure"
#
# | Failure Type         | What Happened                                      | Symptom                        |
# |----------------------|----------------------------------------------------|--------------------------------|
# | **Retrieval failure** | Correct chunk was NOT in top-k results             | Model refuses (correctly!) or  |
# |                      | BM25 couldn't match the query to the right chunk   | answers from wrong chunk       |
# | **Generation failure**| Correct chunk WAS retrieved, but model misread it  | Wrong answer despite good context |
# |                      | or hallucinated beyond the context                 | or hallucination with citation |
#
# **How to diagnose**:
# 1. Look at the retrieved chunks -- is the answer actually there?
# 2. If YES -> generation failure (prompt or model issue)
# 3. If NO  -> retrieval failure (chunking or retriever issue)
#
# **Most failures in RAG are retrieval failures** (~80%).
# Fix retrieval first, then tune generation.
#
# ---
#
# ### 5. Why RAG Beats Plain ChatGPT for This Use Case
#
# | Aspect            | Plain ChatGPT / Gemini              | RAG (our pipeline)              |
# |-------------------|--------------------------------------|---------------------------------|
# | Source of truth    | Training data (frozen, opaque)      | Your textbook (specific, known) |
# | Hallucination      | Common for niche topics             | Bounded by retrieved context    |
# | Verifiability      | No citations, can't check           | Chunk IDs, page numbers         |
# | Freshness          | Stale (training cutoff)             | Always uses current corpus      |
# | Domain accuracy    | General knowledge, may be wrong     | Exact NCERT content             |
# | Cost               | Full model inference per query      | Smaller context, cheaper        |
#
# For a Class 9 student asking about cells:
# - ChatGPT might say "the cell membrane is semipermeable" (correct but
#   unverifiable, and might use college-level language)
# - Our RAG system says "According to [1] (page 6), the cell membrane
#   is called the plasma membrane and is made of lipids and proteins"
#   -- exact textbook wording, verifiable, age-appropriate.
#
# **RAG wins when the answer must come from a specific source.**
