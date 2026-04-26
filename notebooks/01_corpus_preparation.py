# %% [markdown]
# #  Phase 1: Corpus Preparation
#
# **Goal**: Extract clean, structured text from the NCERT PDF and save it as a
# machine-readable JSON corpus for downstream retrieval.
#
# **PDF**: `data/iesc102.pdf` — NCERT Class 9 Science, Chapter 2
# *"Cell: The Building Block of Life"*
#
# **Pipeline**: Raw Extraction → Cleaning → Segmentation → JSON Export
#
# ---

# %% Cell 1 — Imports & Configuration
"""
We use pdfplumber as our primary extractor because it gives
layout-aware text positioning.  PyMuPDF (fitz) is available
as a fallback but is not needed for this chapter.
"""
import pdfplumber
import re
import json
import os
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Paths — resolve correctly whether run as .py or inside Jupyter / VS Code
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

PDF_PATH    = PROJECT_ROOT / "data" / "iesc102.pdf"
OUTPUT_DIR  = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
CORPUS_PATH = OUTPUT_DIR / "corpus.json"

print(f"Project root : {PROJECT_ROOT}")
print(f"PDF path     : {PDF_PATH}  (exists: {PDF_PATH.exists()})")
print(f"Output dir   : {OUTPUT_DIR}")

# %% Cell 2 — Raw Text Extraction (page by page)
"""
┌─────────────────────────────────────────────────────────┐
│  WHY pdfplumber?                                        │
│                                                         │
│  • Layout-aware: respects column order & glyph spacing  │
│  • Returns one string per page — easy to iterate        │
│  • Handles font-encoded "y" bullets (common in NCERT)   │
│  • Pure-Python, no C dependencies → Windows friendly    │
└─────────────────────────────────────────────────────────┘
"""


def extract_raw_pages(pdf_path: Path) -> list[dict]:
    """Extract raw text from every page of the PDF.

    Returns
    -------
    list[dict]
        Each dict has: page_num (1-indexed), raw_text, line_count, char_count.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({
                "page_num":   i + 1,            # 1-indexed for humans
                "raw_text":   text,
                "line_count": len(text.split("\n")),
                "char_count": len(text),
            })
    return pages


raw_pages = extract_raw_pages(PDF_PATH)
print(f"\n[OK] Extracted {len(raw_pages)} pages")
print(f"  Total raw characters: {sum(p['char_count'] for p in raw_pages):,}")

# Quick preview — first page
print("\n--- PAGE 1 PREVIEW (first 500 chars) ---")
print(raw_pages[0]["raw_text"][:500])
print("...")

# %% Cell 3 — Text Cleaning Functions
"""
NCERT PDFs are typeset in Adobe InDesign and have several artefacts
that MUST be removed before any NLP or retrieval work:

  1. InDesign footer — doubled-character filenames:
     "CChhaapptteerr--0022..iinndddd 88 0033--AApprr--2266 77::2288::0011 PPMM"
  2. Running header — "Cell: The Building Block of Life" on every even page
  3. Standalone page numbers — a line that is just "9", "10", etc.
  4. "Exploration|Grade 9" footers
  5. Broken hyphenation — "environ-\nment" → "environment"
  6. Font-encoded bullets — "y " at line start is actually "•"
  7. Excessive blank lines
"""

# ─── Pattern Library ─────────────────────────────────────────────────────────
RE_INDESIGN_FOOTER = re.compile(
    r"^CChh.*$",  re.MULTILINE                     # doubled-char artifact
)
RE_RUNNING_HEADER = re.compile(
    r"^Cell:\s*The\s+Building\s+Block\s+of\s+Life\s*$", re.MULTILINE
)
RE_PAGE_NUMBER = re.compile(
    r"^\s*\d{1,3}\s*$",  re.MULTILINE               # lone "9", "27", etc.
)
RE_EXPLORATION_FOOTER = re.compile(
    r"^\s*\d+\s+Exploration\s*\|?\s*Grade\s+\d+\s*$", re.MULTILINE
)
RE_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")        # word-\nword
RE_MULTI_BLANK  = re.compile(r"\n{3,}")              # 3+ newlines → 2
RE_BULLET_Y = re.compile(r"^y\s+", re.MULTILINE)    # "y " → "• "


def clean_page_text(text: str, page_num: int) -> str:
    """Apply all cleaning steps to one page's raw text.

    Parameters
    ----------
    text : str
        Raw extracted text from one PDF page.
    page_num : int
        1-indexed page number (used to decide header stripping).

    Returns
    -------
    str
        Cleaned text, possibly empty.
    """
    # 1  InDesign footer (present on all pages)
    text = RE_INDESIGN_FOOTER.sub("", text)

    # 2  Running header (pages 2+ carry the chapter title at top)
    if page_num > 1:
        text = RE_RUNNING_HEADER.sub("", text)

    # 3  Standalone page numbers
    text = RE_PAGE_NUMBER.sub("", text)

    # 4  Exploration|Grade footers
    text = RE_EXPLORATION_FOOTER.sub("", text)

    # 5  Fix hyphenation across lines
    text = RE_HYPHEN_BREAK.sub(r"\1\2", text)

    # 6  Convert font-encoded "y" bullets to "•"
    text = RE_BULLET_Y.sub("- ", text)

    # 7  Normalise whitespace
    text = RE_MULTI_BLANK.sub("\n\n", text)
    text = text.strip()

    return text


# ── Demo: show before/after on page 2 ────────────────────────────────────────
demo_raw   = raw_pages[1]["raw_text"]
demo_clean = clean_page_text(demo_raw, page_num=2)

print("--- RAW (page 2, last 200 chars) ---")
print(demo_raw[-200:])
print("\n--- CLEANED (page 2, last 200 chars) ---")
print(demo_clean[-200:])

# %% Cell 4 — Apply Cleaning to All Pages
cleaned_pages = []
for p in raw_pages:
    cleaned = clean_page_text(p["raw_text"], p["page_num"])
    if cleaned:                                      # skip truly empty pages
        cleaned_pages.append({
            "page_num": p["page_num"],
            "text":     cleaned,
        })

total_clean_chars = sum(len(p["text"]) for p in cleaned_pages)
total_raw_chars   = sum(p["char_count"] for p in raw_pages)
reduction_pct     = (1 - total_clean_chars / total_raw_chars) * 100

print(f"\n[OK] Cleaned {len(cleaned_pages)} non-empty pages")
print(f"  Total characters : {total_clean_chars:,}")
print(f"  Noise removed    : {reduction_pct:.1f}% of raw text")

# %% Cell 5 — Segmentation Logic
"""
┌─────────────────────────────────────────────────────────┐
│  SEGMENTATION STRATEGY                                  │
│                                                         │
│  1. Split each page into paragraphs (double-newline)    │
│  2. Further split at structural markers (Activity X.X)  │
│  3. Split oversized chunks (>1200 chars) at sentence    │
│     boundaries so embeddings stay focused                │
│  4. Classify each chunk using rule-based heuristics      │
│  5. Track document zones via a simple state machine:     │
│                                                         │
│     [normal] ──"At a Glance"──▸ [summary]               │
│     [summary] ──numbered Q──▸ [exercises]               │
│     [exercises] ──"The Journey Beyond"──▸ [beyond]       │
└─────────────────────────────────────────────────────────┘

Chunk types:
  concept     — explanatory paragraphs, definitions, descriptions
  activity    — hands-on activities (Activity 2.X blocks)
  question    — in-text questions (Think It Over, What if..., ?)
  summary     — bullet points from "At a Glance" section
  exercise    — end-of-chapter numbered problems
"""

# ─── Classification patterns ─────────────────────────────────────────────────
RE_ACTIVITY_START = re.compile(r"Activity\s+\d+\.\d+", re.IGNORECASE)
RE_EXERCISE_NUM   = re.compile(r"^\d{1,2}\.\s+\w")
RE_SECTION_HEADER = re.compile(r"^\d+\.\d+(?:\.\d+)?\s+\w")
RE_QUESTION_MARKERS = re.compile(
    r"Think\s+It\s+Over"
    r"|What\s+if\s*\.{0,3}"
    r"|Threads\s+of\s+Curiosity"
    r"|Ready\s+to\s+Go\s+Beyond"
    r"|What\s+do\s+you\s+observe"
    r"|Do\s+you\s+observe"
    r"|Pause\s+and\s+Ponder"
    r"|What\s+do\s+you\s+infer",
    re.IGNORECASE,
)

# Zone transition markers
SUMMARY_MARKER  = "At a Glance"
BEYOND_MARKER   = "The Journey Beyond"
QUEST_MARKER    = "The Quest Continues"

# Chunk size limits
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 80


# ─── Paragraph splitting ─────────────────────────────────────────────────────
def split_into_paragraphs(text: str) -> list[str]:
    """Split cleaned page text into paragraph-level chunks.

    Tiny fragments (<MIN_CHUNK_CHARS) are merged into the previous
    chunk to avoid noisy micro-chunks.
    """
    raw_paras = re.split(r"\n\s*\n", text)

    result = []
    for para in raw_paras:
        para = para.strip()
        if not para or len(para) < MIN_CHUNK_CHARS:
            # Merge tiny fragments with the previous chunk
            if result and len(para) > 10:
                result[-1] = result[-1] + "\n" + para
            continue

        # Split on Activity headers appearing mid-paragraph
        parts = re.split(r"(?=Activity\s+\d+\.\d+:)", para)
        for p in parts:
            p = p.strip()
            if p:
                result.append(p)

    return result


# ─── Oversized chunk splitting ────────────────────────────────────────────────
def split_oversized(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split a chunk that exceeds max_chars at sentence boundaries.

    This keeps each chunk focused enough for embedding models
    (most have a sweet spot around 256–512 tokens ≈ 800–1500 chars).
    """
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=[.?!])\s+', text)

    chunks, current = [], ""
    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ─── Chunk classification ────────────────────────────────────────────────────
def classify_chunk(text: str, zone: str) -> str:
    """Classify a text chunk into a corpus type.

    Parameters
    ----------
    text : str
        The chunk content.
    zone : str
        Current document zone: "normal", "summary", "exercises", or "beyond".

    Returns
    -------
    str
        One of: concept, activity, question, summary, exercise.
    """
    # ── Zone-based overrides ──
    if zone == "summary":
        return "summary"
    if zone in ("exercises", "beyond"):
        return "exercise"

    first_line = text.split("\n")[0].strip()

    # ── Activity blocks ──
    if RE_ACTIVITY_START.search(first_line):
        return "activity"

    # ── Compute question-mark statistics (used by multiple rules) ──
    q_marks     = text.count("?")
    all_endings = len(re.findall(r"[.?!]", text)) or 1
    q_ratio     = q_marks / all_endings

    # ── Explicit question markers (search whole chunk) ──
    #    NCERT PDFs interleave sidebar text ("Think It Over") into body
    #    paragraphs via pdfplumber.  In LONG chunks (>600 chars), require
    #    a meaningful question-mark ratio too — otherwise the sidebar
    #    mention is incidental and the chunk is really a concept.
    if RE_QUESTION_MARKERS.search(text):
        if len(text) < 600 or q_ratio >= 0.25:
            return "question"

    # ── Question-heavy paragraphs ──
    #    Only classify as question if questions *dominate* the chunk
    if q_marks >= 2 and q_ratio >= 0.5:
        return "question"

    # ── Short paragraphs that ARE a question ──
    #    Must be short AND dominated by ? marks (not just one incidental ?)
    if q_marks >= 1 and len(text) < 300 and q_ratio >= 0.4:
        return "question"

    return "concept"


# ─── Main segmentation engine ────────────────────────────────────────────────
def segment_all_pages(cleaned_pages: list[dict]) -> list[dict]:
    """Segment every page into typed chunks with unique IDs.

    Uses a zone state machine to track document sections:
        normal → summary → exercises → beyond
    """
    corpus   = []
    chunk_id = 0
    zone     = "normal"           # "normal" | "summary" | "exercises" | "beyond"

    def add_chunks(text: str, page_num: int):
        """Helper: split, classify, and append chunks."""
        nonlocal chunk_id, zone

        for sub in split_oversized(text):
            # Check for zone transitions WITHIN chunks
            if BEYOND_MARKER in sub or QUEST_MARKER in sub:
                zone = "beyond"

            chunk_id += 1
            corpus.append({
                "id":      f"chunk_{chunk_id:03d}",
                "page":    page_num,
                "type":    classify_chunk(sub, zone),
                "content": sub,
            })

    for page in cleaned_pages:
        text     = page["text"]
        page_num = page["page_num"]

        # ── Check for "At a Glance" boundary ──
        if SUMMARY_MARKER in text and zone == "normal":
            before, _, after = text.partition(SUMMARY_MARKER)

            # Process pre-summary content
            if before.strip():
                for para in split_into_paragraphs(before):
                    add_chunks(para, page_num)

            zone = "summary"

            # Process summary content (may transition to exercises)
            if after.strip():
                for para in split_into_paragraphs(after):
                    first_line = para.split("\n")[0].strip()
                    if RE_EXERCISE_NUM.match(first_line) or first_line.startswith("(i)"):
                        zone = "exercises"
                    add_chunks(para, page_num)
            continue

        # ── Pages after "At a Glance" ──
        if zone in ("summary", "exercises", "beyond"):
            for para in split_into_paragraphs(text):
                first_line = para.split("\n")[0].strip()
                if RE_EXERCISE_NUM.match(first_line):
                    zone = "exercises"
                add_chunks(para, page_num)
            continue

        # ── Normal pages (before summary) ──
        for para in split_into_paragraphs(text):
            add_chunks(para, page_num)

    return corpus


def merge_undersized(corpus: list[dict], min_chars: int = MIN_CHUNK_CHARS) -> list[dict]:
    """Merge chunks smaller than min_chars into their nearest same-type neighbor.

    split_oversized can produce tail fragments below the minimum.
    This pass folds them back into the previous or next chunk.
    """
    if not corpus:
        return corpus

    merged = [corpus[0]]
    for entry in corpus[1:]:
        prev = merged[-1]
        if len(entry["content"]) < min_chars and entry["type"] == prev["type"]:
            # Merge into previous chunk
            prev["content"] = prev["content"] + "\n" + entry["content"]
        elif len(prev["content"]) < min_chars and entry["type"] == prev["type"]:
            # Previous was tiny — merge forward
            entry["content"] = prev["content"] + "\n" + entry["content"]
            merged[-1] = entry
        else:
            merged.append(entry)

    # Re-number IDs after merging
    for i, entry in enumerate(merged):
        entry["id"] = f"chunk_{i+1:03d}"

    return merged


corpus = segment_all_pages(cleaned_pages)
corpus = merge_undersized(corpus, min_chars=120)  # catch borderline fragments
print(f"\n[OK] Segmented into {len(corpus)} chunks")

# ── Preview a few chunks of each type ─────────────────────────────────────────
for ctype in ["concept", "activity", "question", "summary", "exercise"]:
    examples = [c for c in corpus if c["type"] == ctype]
    print(f"\n{'='*60}")
    print(f"  TYPE: {ctype}  ({len(examples)} chunks)")
    print(f"{'='*60}")
    if examples:
        preview = examples[0]["content"][:180].replace("\n", " | ")
        print(f"  [{examples[0]['id']}] {preview}...")

# %% Cell 6 — Save Structured Corpus to JSON
"""
Output schema:
[
  {
    "id":      "chunk_001",        // unique, zero-padded
    "page":    1,                  // 1-indexed PDF page
    "type":    "concept",          // concept | activity | question | summary | exercise
    "content": "..."              // cleaned text
  },
  ...
]
"""
with open(CORPUS_PATH, "w", encoding="utf-8") as f:
    json.dump(corpus, f, indent=2, ensure_ascii=False)

file_size_kb = CORPUS_PATH.stat().st_size / 1024
print(f"\n[OK] Saved corpus -> {CORPUS_PATH}")
print(f"  File size    : {file_size_kb:.1f} KB")
print(f"  Total chunks : {len(corpus)}")

# %% Cell 7 — Corpus Statistics Report
"""
A quick quality-assurance report: are the chunk sizes reasonable?
Is the type distribution balanced?  Any red flags?
"""
print("=" * 64)
print("     CORPUS STATISTICS REPORT")
print("=" * 64)

total_chunks = len(corpus)
total_chars  = sum(len(c["content"]) for c in corpus)
avg_length   = total_chars / total_chunks if total_chunks else 0
lengths      = [len(c["content"]) for c in corpus]

print(f"\n  Total chunks     : {total_chunks}")
print(f"  Total characters : {total_chars:,}")
print(f"  Avg chunk length : {avg_length:.0f} chars")
print(f"  Min chunk length : {min(lengths)} chars")
print(f"  Max chunk length : {max(lengths)} chars")
print(f"  Median length    : {sorted(lengths)[len(lengths)//2]} chars")

# ── Type distribution ─────────────────────────────────────────────────────────
type_counts = Counter(c["type"] for c in corpus)

print(f"\n  {'Type':<15} {'Count':>6} {'Pct':>7} {'Avg Len':>8} {'Pages':>12}")
print(f"  {'-'*15} {'-'*6} {'-'*7} {'-'*8} {'-'*12}")
for ctype, count in type_counts.most_common():
    pct         = count / total_chunks * 100
    type_lens   = [len(c["content"]) for c in corpus if c["type"] == ctype]
    type_avg    = sum(type_lens) / len(type_lens)
    type_pages  = sorted(set(c["page"] for c in corpus if c["type"] == ctype))
    page_range  = f"{type_pages[0]}-{type_pages[-1]}" if len(type_pages) > 1 else str(type_pages[0])
    print(f"  {ctype:<15} {count:>6} {pct:>6.1f}% {type_avg:>7.0f} {page_range:>12}")

# ── Page coverage ─────────────────────────────────────────────────────────────
pages_covered = sorted(set(c["page"] for c in corpus))
print(f"\n  Pages covered    : {pages_covered[0]}-{pages_covered[-1]} "
      f"({len(pages_covered)} pages)")

# ── Red-flag checks ───────────────────────────────────────────────────────────
tiny_chunks = [c for c in corpus if len(c["content"]) < 100]
huge_chunks = [c for c in corpus if len(c["content"]) > MAX_CHUNK_CHARS]
if tiny_chunks:
    print(f"\n  [!] {len(tiny_chunks)} chunks under 100 chars (may be noise)")
if huge_chunks:
    print(f"  [!] {len(huge_chunks)} chunks over {MAX_CHUNK_CHARS} chars (may need further splitting)")
if not tiny_chunks and not huge_chunks:
    print(f"\n  [OK] No red flags -- all chunks are between 100-{MAX_CHUNK_CHARS} chars")

print("\n" + "=" * 64)
print("   [OK]  Corpus is ready for Phase 2 (Retrieval)")
print("=" * 64)


# %% [markdown]
# ---
# ##  Mentor Notes: Why This Phase Matters
#
# ---
#
# ### 1. Why clean text matters MORE than model choice in RAG
#
# **Garbage in, garbage out** is the single most important lesson in RAG.
#
# If you feed a retriever (BM25 or dense) text full of `CChhaapptteerr--0022..iinndddd`
# footer junk, broken hyphens like `environ-\nment`, or stray page numbers,
# your retrieval scores **collapse**.  The retriever matches on surface tokens —
# noise tokens dilute the signal, and the model cannot fix what it cannot find.
#
# Think of it this way:
#
# | What you change           | Impact on answer quality |
# |---------------------------|------------------------|
# | GPT-4 → GPT-3.5          | –5 to –10%             |
# | Clean text → dirty text   | **–30 to –50%**        |
#
# The retriever is the *bottleneck* in RAG — if the right chunk never surfaces,
# even the best LLM will hallucinate.  Clean text is the highest-ROI
# investment in any RAG pipeline, period.
#
# ---
#
# ### 2. What "document structure" means and why NCERT PDFs are tricky
#
# **Document structure** = the logical hierarchy of a document:
#
# ```
# Chapter
#   └── Section (2.1, 2.2, ...)
#         └── Paragraph
#               └── Sentence
# ```
#
# Plus non-linear elements: sidebars, activities, figures, captions.
#
# **Why NCERT PDFs are tricky:**
#
# 1. **Designed for print, not machines.**  The layout has floating sidebars
#    ("Think It Over", "Threads of Curiosity"), multi-column areas, and
#    captioned figures that break the linear reading order.
#
# 2. **pdfplumber extracts by glyph position**, so sidebar text gets
#    interleaved with body text on the same "line."
#
# 3. **InDesign artefacts** — headers/footers repeat on every page with
#    doubled-character filenames (`CChhaapptteerr`) that look like gibberish.
#
# 4. **Font encoding quirks** — bullet points render as the letter `y`
#    instead of a bullet symbol (•) because of how the font maps glyphs.
#
# 5. **No semantic tags.**  Unlike HTML (`<h1>`, `<p>`, `<aside>`) or EPUB,
#    a PDF is just positioned glyphs on a canvas.  ALL structure must be
#    inferred from patterns, spacing, and heuristic rules.
#
# ---
#
# ### 3. The difference between "extraction" and "parsing"
#
# | Term            | What it does                                             | Example                                              |
# |-----------------|----------------------------------------------------------|------------------------------------------------------|
# | **Extraction**  | Pull raw text *out of* the PDF container                 | `pdfplumber.extract_text()` → flat string            |
# | **Parsing**     | Understand the *meaning* — headings, questions, sidebars | Our `classify_chunk()` function → typed JSON entries  |
#
# **Extraction is mechanical**: read bytes → decode glyphs → return string.
# **Parsing is semantic**: *this* paragraph is a definition, *that* one is
# a question, and *this* block is an Activity.
#
# Most tutorials stop at extraction.  **Parsing is where the real value is**
# because it lets you:
# - **Filter** chunks by type (only retrieve concepts, not exercises)
# - **Weight** chunks differently in your retriever
# - **Route** chunks to specialized prompts (e.g., "answer this question"
#   vs. "explain this concept")
#
# ---
#
# ### 4. Why save to JSON (not CSV, not plain .txt)
#
# | Format         | Problem                                                    |
# |----------------|------------------------------------------------------------|
# | **Plain .txt** | No metadata — you lose page numbers, types, IDs.  Cannot filter. |
# | **CSV**        | Breaks when content has commas, newlines, or quotes (always does). |
# | **JSON**       | [OK] Preserves nested structure, handles multiline natively.  |
#
# JSON also lets you **add fields later** (e.g., `embedding`, `token_count`,
# `retrieval_score`) without changing the schema.  CSV would require adding
# columns and re-escaping everything.
#
# Your downstream retriever, prompt builder, and evaluation harness all
# consume JSON natively.  **Use the format your pipeline speaks.**
#
# ---
#
# ### 5. What to expect next (Phase 2 preview)
#
# In Phase 2 we will:
# 1. **Chunk further** if needed (semantic splitting by topic)
# 2. **Embed** each chunk using a sentence-transformer model
# 3. **Index** embeddings for fast nearest-neighbour retrieval
# 4. **Build a retriever** that takes a student's question and
#    returns the most relevant chunks
#
# The quality of *this* corpus directly determines retrieval accuracy.
# A well-segmented corpus with clean text and meaningful types is the
# foundation everything else stands on.
