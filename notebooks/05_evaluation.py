# %% [markdown]
# # Phase 5: Evaluation Framework
#
# **Goal**: Run a reproducible, deterministic evaluation of the full
# RAG pipeline on 20 test questions and report honest metrics.
#
# **Test set**: `evaluation/questions.json`
# - 8 factual, 6 paraphrased, 6 out-of-scope
#
# **Scoring**:
# - Correctness  : key-term overlap (>= 2 terms match)
# - Grounding    : no unknown named entities in the answer
# - Refusal      : out-of-scope questions should be refused
#
# ---

# %% Cell 1 -- Imports & Setup
import sys
import os
import json
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

from retriever import BM25Retriever, load_chunks
from generator import GroundedGenerator
from evaluator import (
    evaluate_pipeline, compute_metrics, print_report, save_results,
)

CHUNKS_PATH    = PROJECT_ROOT / "outputs" / "chunks_semantic.json"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_PATH   = PROJECT_ROOT / "evaluation" / "results.json"

print(f"Project root  : {PROJECT_ROOT}")

# Check API key
api_key = os.getenv("GEMINI_API_KEY", "")
DRY_RUN = not bool(api_key)
if DRY_RUN:
    print("\n[!] No GEMINI_API_KEY -- running in DRY-RUN mode")
    print("    Retrieval will run, but generation will be skipped.")
    print("    Set GEMINI_API_KEY in .env to run the full evaluation.\n")
else:
    print(f"[OK] API key loaded ({len(api_key)} chars)")


# %% Cell 2 -- Load Everything
# Chunks
chunks = load_chunks(CHUNKS_PATH)
print(f"Loaded {len(chunks)} chunks")

# Build full corpus text for grounding checks
corpus_text = " ".join(c["text"] for c in chunks)
print(f"Corpus vocabulary: {len(set(corpus_text.lower().split())):,} unique words")

# Questions
with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)
print(f"Loaded {len(questions)} evaluation questions")

# Count by type
from collections import Counter
type_counts = Counter(q["type"] for q in questions)
for t, n in type_counts.most_common():
    print(f"  {t:<14} {n:>3}")

# Retriever
retriever = BM25Retriever(chunks)
print(f"\n{retriever}")

# Generator
if not DRY_RUN:
    generator = GroundedGenerator(api_key=api_key, model_name="gemini-2.5-flash-lite")
    print(f"{generator}")
else:
    generator = None
    print("Generator: DRY-RUN")


# %% Cell 3 -- Run Evaluation
"""
This runs the full pipeline for each of the 20 questions:
  query -> BM25 retrieve (top 5) -> Gemini generate -> score

With temperature=0, results are DETERMINISTIC: same questions,
same corpus, same retriever -> same scores every time.
"""
print("\nRunning evaluation...")
print(f"  Mode     : {'LIVE (API calls)' if not DRY_RUN else 'DRY-RUN (retrieval only)'}")
print(f"  Questions: {len(questions)}")
print(f"  top_k    : 5")
print()

results = evaluate_pipeline(
    questions=questions,
    retriever=retriever,
    generator=generator,
    corpus_text=corpus_text,
    top_k=5,
    dry_run=DRY_RUN,
)

metrics = compute_metrics(results)
print(f"[OK] Evaluation complete\n")


# %% Cell 4 -- Results Table
print_report(results, metrics)


# %% Cell 5 -- Retrieval Quality Analysis
"""
Even in dry-run mode, we can analyze RETRIEVAL quality:
how well does BM25 find relevant chunks for each question?

High retrieval scores (>5) suggest the right chunks were found.
Low scores (<2) suggest retrieval failure.
"""
print("\n" + "=" * 80)
print("  RETRIEVAL QUALITY ANALYSIS")
print("=" * 80)

print(f"\n  {'ID':<6} {'Type':<14} {'InScope':>7} {'Top Score':>10} "
      f"{'Quality':>10}  Question")
print(f"  {'-'*6} {'-'*14} {'-'*7} {'-'*10} {'-'*10}  {'-'*30}")

for r in results:
    top_score = r["retrieval_scores"][0] if r["retrieval_scores"] else 0
    if r["in_scope"]:
        quality = "GOOD" if top_score > 5 else ("WEAK" if top_score > 2 else "FAIL")
    else:
        quality = "n/a (OOS)"
    q_preview = r["question"][:30]
    print(f"  {r['id']:<6} {r['type']:<14} {str(r['in_scope']):>7} "
          f"{top_score:>10.2f} {quality:>10}  {q_preview}...")

# Summary
in_scope_results = [r for r in results if r["in_scope"]]
top_scores = [r["retrieval_scores"][0] for r in in_scope_results]
good = sum(1 for s in top_scores if s > 5)
weak = sum(1 for s in top_scores if 2 < s <= 5)
fail = sum(1 for s in top_scores if s <= 2)
print(f"\n  In-scope retrieval quality: "
      f"GOOD={good}/{len(in_scope_results)}, "
      f"WEAK={weak}/{len(in_scope_results)}, "
      f"FAIL={fail}/{len(in_scope_results)}")


# %% Cell 6 -- Save Results
save_results(results, metrics, RESULTS_PATH)


# %% Cell 7 -- Detailed View: Best & Worst Cases
"""
Show the actual retrieved chunks for the best and worst
retrieval scores -- helps diagnose where the pipeline
succeeds and fails.
"""
print("\n" + "=" * 80)
print("  DETAILED VIEW: BEST vs WORST RETRIEVAL")
print("=" * 80)

# Best in-scope retrieval
best = max(in_scope_results, key=lambda r: r["retrieval_scores"][0])
print(f"\n  BEST RETRIEVAL:")
print(f"    Q: \"{best['question']}\"")
print(f"    Top BM25 score: {best['retrieval_scores'][0]:.2f}")
print(f"    Answer preview: {best['answer'][:120]}...")

# Worst in-scope retrieval
worst = min(in_scope_results, key=lambda r: r["retrieval_scores"][0])
print(f"\n  WORST RETRIEVAL:")
print(f"    Q: \"{worst['question']}\"")
print(f"    Top BM25 score: {worst['retrieval_scores'][0]:.2f}")
print(f"    Answer preview: {worst['answer'][:120]}...")

if not DRY_RUN:
    # Show an out-of-scope example
    oos = [r for r in results if not r["in_scope"]]
    if oos:
        ex = oos[0]
        print(f"\n  OUT-OF-SCOPE EXAMPLE:")
        print(f"    Q: \"{ex['question']}\"")
        print(f"    Refused: {ex['refused']}")
        print(f"    Answer: {ex['answer'][:120]}...")


# %% [markdown]
# ---
# ## Mentor Notes: Evaluation Deep-Dive
#
# ---
#
# ### 1. Why Evaluation Must Be DETERMINISTIC (temp=0, fixed questions)
#
# If your evaluation gives different results each time you run it,
# you cannot:
# - Compare two versions of your pipeline
# - Debug a specific failure
# - Report reliable metrics in a paper or review
#
# **Three things make evaluation deterministic:**
#
# | Control         | How                                       |
# |-----------------|-------------------------------------------|
# | Fixed questions | `questions.json` -- same 20 questions every time |
# | Fixed retrieval | BM25 is deterministic (no randomness)      |
# | Fixed generation| `temperature=0` -- greedy decoding, same output |
#
# If you change the corpus, the chunks, or the retriever, scores WILL
# change -- but that's signal, not noise.  You can diff the results
# JSON to see exactly what changed.
#
# ---
#
# ### 2. The 3 Failure Modes: Retrieval Miss, Generation Drift, False Refusal
#
# | Failure Mode        | What Happens                                    | How to Detect                   |
# |---------------------|-------------------------------------------------|---------------------------------|
# | **Retrieval miss**  | Right chunk is NOT in top-k                     | Low BM25 score, answer is wrong |
# | **Generation drift**| Right chunk IS retrieved, but model misreads it  | High BM25 score, answer is wrong|
# | **False refusal**   | In-scope question, but model says "not found"   | `refusal_correct = 0` for in-scope |
#
# **Retrieval miss** (~60% of failures): Fix chunking or add synonyms.
# **Generation drift** (~30%): Fix the grounding prompt or use a better model.
# **False refusal** (~10%): Usually caused by the retriever returning
# low-quality chunks -- the generator "sees" irrelevant context and
# correctly concludes the answer isn't there.
#
# ---
#
# ### 3. Why "Keyword Overlap" Is a Weak but Acceptable Baseline Metric
#
# Our correctness metric counts how many **key terms** from the expected
# answer appear in the generated answer.  If >= 2 match, we score 1.
#
# **Strengths:**
# - Dead simple, no dependencies, fully deterministic
# - Works well for factual NCERT questions (answers have distinctive terms)
# - Fast: no model inference needed for scoring
#
# **Weaknesses:**
# - Misses paraphrases: "energy currency" vs "ATP" won't match
# - No semantic understanding: word order, negation, context ignored
# - Threshold (2 terms) is arbitrary
#
# **Why it's acceptable for this project:**
# - We're evaluating a prototype, not publishing a paper
# - The questions have distinctive key terms (cellulose, mitochondria, etc.)
# - It flags obvious failures reliably
# - A reviewer sees you THOUGHT about evaluation, which is 90% of the battle
#
# ---
#
# ### 4. What BLEU/ROUGE Are and Why They're Overkill Here
#
# | Metric   | What It Measures                                  | When to Use              |
# |----------|---------------------------------------------------|--------------------------|
# | **BLEU** | N-gram precision between generated and reference   | Machine translation      |
# | **ROUGE**| N-gram recall between generated and reference      | Summarization            |
# | **BERTScore** | Semantic similarity via BERT embeddings       | Any text generation      |
#
# **Why overkill here:**
# - Our answers are free-form, not fixed translations
# - The reference answers in questions.json are just one valid phrasing
# - BLEU/ROUGE penalize valid paraphrases (false negatives)
# - For 20 questions, manual inspection is more informative than any metric
#
# **When to upgrade:** If you scale to 200+ questions and need automated
# comparison between pipeline versions, use BERTScore or LLM-as-judge.
#
# ---
#
# ### 5. How a Reviewer Will Judge Your Evaluation Section
#
# A reviewer (teacher, interviewer, or peer) looks for:
#
# 1. **Honest reporting**: Did you show failures, not just successes?
#    Our report shows retrieval quality AND failure cases.
#
# 2. **Reproducibility**: Can someone re-run and get the same numbers?
#    Yes -- deterministic pipeline, fixed questions, saved results.
#
# 3. **Appropriate scope**: Did you over-claim?
#    We use simple metrics and acknowledge their limitations.
#
# 4. **Failure analysis**: Do you understand WHY things fail?
#    We diagnose retrieval miss vs. generation drift vs. false refusal.
#
# 5. **Actionable insights**: What would you improve next?
#    The retrieval quality analysis shows exactly which queries need
#    better chunking or a dense retriever.
#
# **The bar is not "perfect scores."  The bar is "thoughtful evaluation
# with honest analysis."**
