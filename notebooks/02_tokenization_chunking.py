# %% [markdown]
# # Phase 2: Tokenization & Chunking
#
# **Goal**: Compare tokenizers, implement two chunking strategies, and produce
# retrieval-ready chunk files for the RAG pipeline.
#
# **Input**: `outputs/corpus.json` (53 typed chunks from Phase 1)
# **Outputs**:
# - `outputs/chunks_fixed.json`    (fixed-window strategy)
# - `outputs/chunks_semantic.json` (sentence-boundary strategy)
#
# ---

# %% Cell 1 -- Imports & Configuration
import json
import re
from pathlib import Path
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

CORPUS_PATH      = PROJECT_ROOT / "outputs" / "corpus.json"
CHUNKS_FIXED     = PROJECT_ROOT / "outputs" / "chunks_fixed.json"
CHUNKS_SEMANTIC  = PROJECT_ROOT / "outputs" / "chunks_semantic.json"

print(f"Project root : {PROJECT_ROOT}")
print(f"Corpus path  : {CORPUS_PATH}  (exists: {CORPUS_PATH.exists()})")

# Load corpus
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)
print(f"Loaded {len(corpus)} corpus entries")


# %% Cell 2 -- Tokenizer Comparison
"""
We compare THREE tokenizers on the same passage to show how
tokenization choices affect token counts and vocabulary:

  1. Python str.split()   -- whitespace baseline (naive)
  2. NLTK word_tokenize   -- rule-based, linguistically aware
  3. BERT WordPiece       -- subword, model-native

This matters because your RETRIEVER and your LLM each have a
specific tokenizer.  If you count tokens with .split() but
your model uses WordPiece, your chunk sizes will be WRONG.
"""

# Pick a rich sample passage (first concept chunk with decent length)
sample_entry = next(c for c in corpus if c["type"] == "concept" and len(c["content"]) > 400)
sample_text  = sample_entry["content"]

print(f"Sample passage ({len(sample_text)} chars, from {sample_entry['id']}):")
print(f"  \"{sample_text[:120]}...\"")
print()

# --- Tokenizer 1: Python .split() (whitespace) ---
tokens_split = sample_text.split()

# --- Tokenizer 2: NLTK word_tokenize ---
tokens_nltk = word_tokenize(sample_text)

# --- Tokenizer 3: BERT WordPiece ---
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens_bert = bert_tokenizer.tokenize(sample_text)

# --- Comparison Table ---
print("=" * 72)
print("  TOKENIZER COMPARISON")
print("=" * 72)
print(f"  {'Tokenizer':<20} {'Tokens':>8} {'Vocab Size':>12}   First 20 Tokens")
print(f"  {'-'*20} {'-'*8} {'-'*12}   {'-'*40}")

# Whitespace
ws_vocab = set(tokens_split)
print(f"  {'str.split()':<20} {len(tokens_split):>8} {len(ws_vocab):>12}   {tokens_split[:20]}")

# NLTK
nltk_vocab = set(tokens_nltk)
print(f"  {'NLTK word_tokenize':<20} {len(tokens_nltk):>8} {len(nltk_vocab):>12}   {tokens_nltk[:20]}")

# BERT
bert_vocab = bert_tokenizer.vocab_size
print(f"  {'BERT WordPiece':<20} {len(tokens_bert):>8} {bert_vocab:>12}   {tokens_bert[:20]}")

print()
print("  Key observations:")
print(f"  - .split() undercounts: misses punctuation as separate tokens")
print(f"  - NLTK overcounts slightly: splits contractions & punctuation")
print(f"  - BERT produces MORE tokens due to subword splits (## prefixes)")
print(f"  - BERT's vocab is fixed at {bert_vocab:,} -- unknown words get split")


# %% Cell 3 -- Helper: token counting with BERT tokenizer
"""
For chunking, we need a CONSISTENT token counter.  Since our
downstream model will use a transformer tokenizer, we count
tokens with BERT's WordPiece tokenizer -- not .split().

This ensures chunk sizes match the model's actual consumption.
"""


def count_tokens(text: str) -> int:
    """Count tokens using the BERT tokenizer (no special tokens)."""
    return len(bert_tokenizer.tokenize(text))


def count_words(text: str) -> int:
    """Count whitespace-separated words (for the semantic strategy)."""
    return len(text.split())


# Quick sanity check
print(f"Sample token count (BERT) : {count_tokens(sample_text)}")
print(f"Sample word count         : {count_words(sample_text)}")


# %% Cell 4 -- Strategy A: Fixed-Window Chunking (200 tokens, 50-overlap)
"""
FIXED-WINDOW CHUNKING
---------------------
  - Window size : 200 BERT tokens
  - Overlap     : 50 tokens (25% overlap)
  - Stride      : 150 tokens

This is the simplest strategy: slide a fixed window across the
tokenized text.  Overlap ensures that information at chunk
boundaries is not lost.

When to use: quick baseline, uniform chunk sizes, predictable
memory usage.
"""

WINDOW_SIZE = 200    # tokens
OVERLAP     = 50     # tokens
STRIDE      = WINDOW_SIZE - OVERLAP  # 150 tokens


def chunk_fixed_window(corpus: list[dict],
                       window: int = WINDOW_SIZE,
                       overlap: int = OVERLAP) -> list[dict]:
    """Split corpus entries into fixed-size token windows with overlap.

    Parameters
    ----------
    corpus : list[dict]
        Corpus entries with {id, page, type, content}.
    window : int
        Window size in BERT tokens.
    overlap : int
        Number of overlapping tokens between consecutive chunks.

    Returns
    -------
    list[dict]
        Chunks with {chunk_id, source_id, page, type, text, token_count}.
    """
    stride = window - overlap
    chunks = []
    chunk_id = 0

    for entry in corpus:
        # Tokenize into token IDs, then decode windows back to text
        token_ids = bert_tokenizer.encode(entry["content"], add_special_tokens=False)

        if len(token_ids) <= window:
            # Entire entry fits in one chunk
            chunk_id += 1
            chunks.append({
                "chunk_id":    f"fixed_{chunk_id:04d}",
                "source_id":   entry["id"],
                "page":        entry["page"],
                "type":        entry["type"],
                "text":        entry["content"],
                "token_count": len(token_ids),
            })
        else:
            # Slide the window
            for start in range(0, len(token_ids), stride):
                end = min(start + window, len(token_ids))
                window_ids = token_ids[start:end]

                # Skip tiny tail fragments (< 30 tokens)
                if len(window_ids) < 30 and start > 0:
                    continue

                text = bert_tokenizer.decode(window_ids, clean_up_tokenization_spaces=True)
                chunk_id += 1
                chunks.append({
                    "chunk_id":    f"fixed_{chunk_id:04d}",
                    "source_id":   entry["id"],
                    "page":        entry["page"],
                    "type":        entry["type"],
                    "text":        text,
                    "token_count": len(window_ids),
                })

                # If we've reached the end, stop
                if end >= len(token_ids):
                    break

    return chunks


chunks_fixed = chunk_fixed_window(corpus)
print(f"\n[OK] Fixed-window chunking: {len(chunks_fixed)} chunks")
print(f"     Window={WINDOW_SIZE}, Overlap={OVERLAP}, Stride={STRIDE}")

# Preview
for ch in chunks_fixed[:3]:
    preview = ch["text"][:80].replace("\n", " | ")
    print(f"  {ch['chunk_id']} | {ch['token_count']:>3} tok | {preview}...")


# %% Cell 5 -- Strategy B: Semantic Chunking (sentence boundaries, ~150 words)
"""
SEMANTIC CHUNKING
-----------------
  - Split on sentence boundaries (NLTK sent_tokenize)
  - Group sentences until ~150 words
  - NO overlap (sentences are atomic units -- no mid-sentence cuts)
  - Respects the natural structure of the text

When to use: when retrieval quality matters more than uniformity.
Semantic chunks preserve complete thoughts, which gives the LLM
better context for answering questions.
"""

TARGET_WORDS = 150    # soft target per chunk


def chunk_semantic(corpus: list[dict],
                   target_words: int = TARGET_WORDS) -> list[dict]:
    """Split corpus entries at sentence boundaries, grouping to ~target_words.

    Parameters
    ----------
    corpus : list[dict]
        Corpus entries with {id, page, type, content}.
    target_words : int
        Soft word-count target per chunk.

    Returns
    -------
    list[dict]
        Chunks with {chunk_id, source_id, page, type, text, token_count}.
    """
    chunks = []
    chunk_id = 0

    for entry in corpus:
        sentences = sent_tokenize(entry["content"])

        if not sentences:
            continue

        current_sentences = []
        current_word_count = 0

        for sent in sentences:
            sent_words = count_words(sent)

            # If adding this sentence exceeds target AND we already have content,
            # finalize the current chunk
            if current_word_count + sent_words > target_words and current_sentences:
                text = " ".join(current_sentences)
                chunk_id += 1
                chunks.append({
                    "chunk_id":    f"sem_{chunk_id:04d}",
                    "source_id":   entry["id"],
                    "page":        entry["page"],
                    "type":        entry["type"],
                    "text":        text,
                    "token_count": count_tokens(text),
                })
                current_sentences = []
                current_word_count = 0

            current_sentences.append(sent)
            current_word_count += sent_words

        # Flush remaining sentences
        if current_sentences:
            text = " ".join(current_sentences)
            chunk_id += 1
            chunks.append({
                "chunk_id":    f"sem_{chunk_id:04d}",
                "source_id":   entry["id"],
                "page":        entry["page"],
                "type":        entry["type"],
                "text":        text,
                "token_count": count_tokens(text),
            })

    return chunks



MIN_SEMANTIC_TOKENS = 30  # merge chunks below this into neighbors


def merge_tiny_semantic(chunks: list[dict], min_tokens: int = MIN_SEMANTIC_TOKENS) -> list[dict]:
    """Merge semantic chunks that are too small into their neighbor.

    A tiny chunk usually means a short sentence was left over after
    the previous chunk crossed the word target.  We fold it into the
    previous chunk from the same source, or the next one if types match.
    """
    if not chunks:
        return chunks

    merged = [chunks[0]]
    for ch in chunks[1:]:
        prev = merged[-1]
        if ch["token_count"] < min_tokens and ch["source_id"] == prev["source_id"]:
            # Fold into previous chunk
            prev["text"] = prev["text"] + " " + ch["text"]
            prev["token_count"] = count_tokens(prev["text"])
        elif prev["token_count"] < min_tokens and prev["source_id"] == ch["source_id"]:
            # Previous was tiny -- fold forward
            ch["text"] = prev["text"] + " " + ch["text"]
            ch["token_count"] = count_tokens(ch["text"])
            merged[-1] = ch
        else:
            merged.append(ch)

    # Re-number chunk IDs
    for i, ch in enumerate(merged):
        ch["chunk_id"] = f"sem_{i+1:04d}"

    return merged


chunks_semantic = chunk_semantic(corpus)
chunks_semantic = merge_tiny_semantic(chunks_semantic)
print(f"\n[OK] Semantic chunking: {len(chunks_semantic)} chunks")
print(f"     Target ~{TARGET_WORDS} words per chunk, sentence boundaries")
print(f"     (merged fragments under {MIN_SEMANTIC_TOKENS} tokens)")

# Preview
for ch in chunks_semantic[:3]:
    preview = ch["text"][:80].replace("\n", " | ")
    print(f"  {ch['chunk_id']} | {ch['token_count']:>3} tok | {preview}...")


# %% Cell 6 -- Save Both Chunk Sets
for path, data, label in [
    (CHUNKS_FIXED,    chunks_fixed,    "Fixed-window"),
    (CHUNKS_SEMANTIC, chunks_semantic, "Semantic"),
]:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    size_kb = path.stat().st_size / 1024
    print(f"[OK] {label:14s} -> {path.name}  ({size_kb:.1f} KB, {len(data)} chunks)")


# %% Cell 7 -- Side-by-Side Comparison
"""
Compare the two strategies across key metrics.
The right strategy depends on your use case:
  - Fixed:    uniform sizes, predictable, but cuts mid-sentence
  - Semantic: variable sizes, preserves meaning, but uneven
"""


def compute_stats(chunks: list[dict]) -> dict:
    """Compute summary statistics for a chunk set."""
    token_counts = [c["token_count"] for c in chunks]
    return {
        "total":  len(chunks),
        "avg":    sum(token_counts) / len(token_counts),
        "min":    min(token_counts),
        "max":    max(token_counts),
        "median": sorted(token_counts)[len(token_counts) // 2],
    }


stats_fixed    = compute_stats(chunks_fixed)
stats_semantic = compute_stats(chunks_semantic)

print("\n" + "=" * 68)
print("  CHUNKING STRATEGY COMPARISON")
print("=" * 68)
print(f"  {'Strategy':<18} {'Chunks':>8} {'Avg Tok':>9} {'Min':>6} {'Max':>6} {'Median':>8}")
print(f"  {'-'*18} {'-'*8} {'-'*9} {'-'*6} {'-'*6} {'-'*8}")

for label, s in [("Fixed (200/50)", stats_fixed), ("Semantic (~150w)", stats_semantic)]:
    print(f"  {label:<18} {s['total']:>8} {s['avg']:>9.1f} {s['min']:>6} {s['max']:>6} {s['median']:>8}")

print()

# Type distribution comparison
print(f"  {'Type':<12} {'Fixed':>8} {'Semantic':>10}")
print(f"  {'-'*12} {'-'*8} {'-'*10}")
all_types = sorted(set(c["type"] for c in corpus))
for t in all_types:
    f_count = sum(1 for c in chunks_fixed if c["type"] == t)
    s_count = sum(1 for c in chunks_semantic if c["type"] == t)
    print(f"  {t:<12} {f_count:>8} {s_count:>10}")

print()
print("  Recommendation: Use SEMANTIC chunks for this RAG pipeline.")
print("  They preserve complete thoughts and match how students ask questions.")


# %% Cell 8 -- load_chunks() utility function
"""
This function will be imported by downstream notebooks (retrieval,
generation, evaluation).  It's the single entry point for loading
any chunk file.
"""


def load_chunks(path: str | Path = CHUNKS_SEMANTIC) -> list[dict]:
    """Load a chunk JSON file and return the list of chunk dicts.

    Parameters
    ----------
    path : str or Path
        Path to a chunks JSON file.  Defaults to semantic chunks.

    Returns
    -------
    list[dict]
        Each dict has: chunk_id, source_id, page, type, text, token_count.

    Raises
    ------
    FileNotFoundError
        If the chunk file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[OK] Loaded {len(chunks)} chunks from {path.name}")
    return chunks


# Demo
demo_chunks = load_chunks()
print(f"  First chunk: {demo_chunks[0]['chunk_id']} "
      f"({demo_chunks[0]['token_count']} tokens, type={demo_chunks[0]['type']})")
print(f"  Last chunk:  {demo_chunks[-1]['chunk_id']} "
      f"({demo_chunks[-1]['token_count']} tokens, type={demo_chunks[-1]['type']})")


# %% [markdown]
# ---
# ## Mentor Notes: Tokenization & Chunking Deep-Dive
#
# ---
#
# ### 1. What tokenization IS (BPE vs WordPiece)
#
# **Tokenization** = converting a string of characters into a sequence of
# discrete units ("tokens") that a model can process.
#
# There are three major families:
#
# | Method              | How it works                                          | Used by         |
# |---------------------|-------------------------------------------------------|-----------------|
# | **Whitespace**      | Split on spaces/tabs.  Ignores punctuation.           | Baselines only  |
# | **Rule-based**      | Linguistic rules for contractions, punctuation, etc.  | NLTK, spaCy     |
# | **Subword (BPE)**   | Learn merges from data. Frequent words stay whole,    | GPT, LLaMA      |
# |                     | rare words get split: "playing" -> "play" + "ing"     |                 |
# | **Subword (WordPiece)** | Similar to BPE but uses likelihood-based merging. | BERT, DistilBERT |
# |                     | Subwords marked with "##": "embedding" -> "em" + "##bed" + "##ding" | |
#
# **Why it matters for RAG**: Your retriever's tokenizer determines how text
# is sliced.  If you count chunks with `.split()` (whitespace) but your model
# uses WordPiece, you'll consistently *undercount* tokens -- and your chunks
# may silently exceed the model's context window.
#
# **Rule of thumb**: Always count tokens with the *same tokenizer* your
# model uses.
#
# ---
#
# ### 2. Why chunk size is the single most important RAG parameter
#
# Chunk size controls the **precision vs. context** tradeoff:
#
# ```
#   Small chunks (50-100 tokens)       Large chunks (500+ tokens)
#   +---------------------------+      +---------------------------+
#   | + High retrieval precision|      | + More context per chunk  |
#   | + Less noise per chunk    |      | + Fewer total chunks      |
#   | - May lose context        |      | - Lower precision         |
#   | - More chunks to search   |      | - Retriever may miss      |
#   +---------------------------+      +---------------------------+
# ```
#
# Research (e.g., LlamaIndex benchmarks) consistently shows that chunks of
# **128-256 tokens** hit the sweet spot for most QA tasks.  Our semantic
# chunks target ~150 words (~180-200 BERT tokens), which falls right in
# this range.
#
# **Chunk size affects EVERYTHING downstream**:
# - Embedding quality (too long = diluted; too short = no context)
# - Retrieval accuracy (the chunk must contain the answer)
# - LLM generation (the model reads the retrieved chunks)
# - Cost (more tokens = more API spend)
#
# ---
#
# ### 3. What "overlap" does and when it helps vs hurts
#
# **Overlap** = repeating N tokens from the end of chunk K at the start
# of chunk K+1.
#
# ```
#   Chunk 1:  [----------- 200 tokens -----------]
#   Chunk 2:            [--overlap--][--- 150 new tokens ---]
#                       <-- 50 tok -->
# ```
#
# **When it helps**:
# - Fixed-window chunking, where a sentence might straddle a boundary
# - Dense retrieval, where a key phrase split across chunks would be missed
# - Long documents with no clear structural boundaries
#
# **When it hurts**:
# - Increases total chunk count (more storage, slower search)
# - Duplicates information (the same sentence appears in 2 chunks)
# - Confuses re-ranking if both chunks score high for the same query
# - Wastes embedding computation
#
# **Our semantic strategy uses NO overlap** because it splits on sentence
# boundaries -- there are no mid-sentence cuts to "heal."
#
# ---
#
# ### 4. Why BERT has a 512-token limit and why that matters here
#
# BERT uses **positional embeddings** -- a learned vector for each position
# in the sequence (position 0, 1, 2, ..., 511).  During pre-training, BERT
# only saw sequences up to 512 tokens.  If you feed it 600 tokens:
#
# - Tokens 513-600 have **no learned position embedding**
# - The model either truncates them (silently!) or crashes
#
# This means:
# - Every chunk you embed MUST be under 512 tokens
# - Our chunks (avg ~180 tokens) are safely within this limit
# - If you switch to a model with a larger window (e.g., 8192 for
#   some newer models), you could use larger chunks -- but bigger
#   is not always better (see point 2 above)
#
# **Practical rule**: Keep chunks at **50-75% of the model's max**
# to leave room for special tokens ([CLS], [SEP]) and query tokens
# when doing cross-encoding.
#
# ---
#
# ### 5. The tradeoff: more chunks = better precision, slower retrieval
#
# | More chunks (smaller)              | Fewer chunks (larger)              |
# |------------------------------------|------------------------------------|
# | Better precision per chunk         | Faster vector search               |
# | Easier to pinpoint the exact answer| Lower storage / embedding cost     |
# | More embeddings to compute & store | Each chunk has richer context       |
# | Slower nearest-neighbor search     | May miss specific details           |
#
# For our 53-entry corpus, the semantic strategy produces a manageable
# number of chunks.  For a 10,000-page textbook, you'd need approximate
# nearest-neighbor indices (FAISS, Annoy) to keep search fast.
#
# **Bottom line**: Start with semantic chunking at ~150 words.  Measure
# retrieval recall.  Only tune chunk size if recall is low.
