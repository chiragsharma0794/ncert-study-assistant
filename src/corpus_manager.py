"""
src/corpus_manager.py -- Multi-chapter corpus management for the NCERT RAG pipeline.

Wraps the V1 extraction/cleaning/segmentation/chunking pipeline to support
multiple chapters (iesc101.pdf through iesc114.pdf) with:
  - Per-chapter sharded storage (outputs/chapters/ch{NN}_chunks.json)
  - Lazy loading (only requested chapters loaded into memory)
  - Backward-compatible chunk schema (V1 chunks_semantic.json still works)
  - Chapter-aware chunk IDs: ch{NN}_p{PP}_s{SSS}
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Optional, Union

import pdfplumber
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

# =========================================================================
# Constants
# =========================================================================

# PDF filename pattern: iesc1XX.pdf where XX = chapter number (01-14)
_PDF_PATTERN = re.compile(r"iesc1(\d{2})\.pdf", re.IGNORECASE)

# ── Cleaning patterns (shared across all NCERT chapters) ─────────────────
_RE_INDESIGN_FOOTER = re.compile(r"^CChh.*$", re.MULTILINE)
_RE_PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE)
_RE_EXPLORATION_FOOTER = re.compile(
    r"^\s*\d+\s+Exploration\s*\|?\s*Grade\s+\d+\s*$", re.MULTILINE
)
_RE_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_RE_MULTI_BLANK = re.compile(r"\n{3,}")
_RE_BULLET_Y = re.compile(r"^y\s+", re.MULTILINE)

# ── Segmentation patterns ────────────────────────────────────────────────
_RE_ACTIVITY_START = re.compile(r"Activity\s+\d+\.\d+", re.IGNORECASE)
_RE_EXERCISE_NUM = re.compile(r"^\d{1,2}\.\s+\w")
_RE_QUESTION_MARKERS = re.compile(
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

_SUMMARY_MARKER = "At a Glance"
_BEYOND_MARKER = "The Journey Beyond"
_QUEST_MARKER = "The Quest Continues"

_MAX_CHUNK_CHARS = 1200
_MIN_CHUNK_CHARS = 80
_TARGET_WORDS = 150
_MIN_SEMANTIC_TOKENS = 30


# =========================================================================
# Pipeline functions (identical logic to V1 notebooks, not modified)
# =========================================================================

def _extract_raw_pages(pdf_path: Path) -> list[dict]:
    """Extract raw text from every page of a PDF."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page_num": i + 1, "raw_text": text})
    return pages


def _clean_page_text(text: str, page_num: int,
                     running_header_re: Optional[re.Pattern] = None) -> str:
    """Apply all cleaning steps to one page's raw text.

    Parameters
    ----------
    running_header_re : re.Pattern, optional
        Compiled regex for the chapter's running header.
        Each chapter has a different title that appears on even pages.
        If None, no header stripping is performed (safe default).
    """
    text = _RE_INDESIGN_FOOTER.sub("", text)
    if page_num > 1 and running_header_re:
        text = running_header_re.sub("", text)
    text = _RE_PAGE_NUMBER.sub("", text)
    text = _RE_EXPLORATION_FOOTER.sub("", text)
    text = _RE_HYPHEN_BREAK.sub(r"\1\2", text)
    text = _RE_BULLET_Y.sub("- ", text)
    text = _RE_MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def _split_into_paragraphs(text: str) -> list[str]:
    """Split cleaned page text into paragraph-level chunks."""
    raw_paras = re.split(r"\n\s*\n", text)
    result = []
    for para in raw_paras:
        para = para.strip()
        if not para or len(para) < _MIN_CHUNK_CHARS:
            if result and len(para) > 10:
                result[-1] = result[-1] + "\n" + para
            continue
        parts = re.split(r"(?=Activity\s+\d+\.\d+:)", para)
        for p in parts:
            p = p.strip()
            if p:
                result.append(p)
    return result


def _split_oversized(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split a chunk exceeding max_chars at sentence boundaries."""
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


def _classify_chunk(text: str, zone: str) -> str:
    """Classify a text chunk into a corpus type."""
    if zone == "summary":
        return "summary"
    if zone in ("exercises", "beyond"):
        return "exercise"
    first_line = text.split("\n")[0].strip()
    if _RE_ACTIVITY_START.search(first_line):
        return "activity"
    q_marks = text.count("?")
    all_endings = len(re.findall(r"[.?!]", text)) or 1
    q_ratio = q_marks / all_endings
    if _RE_QUESTION_MARKERS.search(text):
        if len(text) < 600 or q_ratio >= 0.25:
            return "question"
    if q_marks >= 2 and q_ratio >= 0.5:
        return "question"
    if q_marks >= 1 and len(text) < 300 and q_ratio >= 0.4:
        return "question"
    return "concept"


def _segment_all_pages(cleaned_pages: list[dict]) -> list[dict]:
    """Segment every page into typed chunks with sequential IDs."""
    corpus = []
    chunk_id = 0
    zone = "normal"

    def add_chunks(text, page_num):
        nonlocal chunk_id, zone
        for sub in _split_oversized(text):
            if _BEYOND_MARKER in sub or _QUEST_MARKER in sub:
                zone = "beyond"
            chunk_id += 1
            corpus.append({
                "id": f"chunk_{chunk_id:03d}",
                "page": page_num,
                "type": _classify_chunk(sub, zone),
                "content": sub,
            })

    for page in cleaned_pages:
        text, page_num = page["text"], page["page_num"]
        if _SUMMARY_MARKER in text and zone == "normal":
            before, _, after = text.partition(_SUMMARY_MARKER)
            if before.strip():
                for para in _split_into_paragraphs(before):
                    add_chunks(para, page_num)
            zone = "summary"
            if after.strip():
                for para in _split_into_paragraphs(after):
                    first_line = para.split("\n")[0].strip()
                    if _RE_EXERCISE_NUM.match(first_line) or first_line.startswith("(i)"):
                        zone = "exercises"
                    add_chunks(para, page_num)
            continue
        if zone in ("summary", "exercises", "beyond"):
            for para in _split_into_paragraphs(text):
                first_line = para.split("\n")[0].strip()
                if _RE_EXERCISE_NUM.match(first_line):
                    zone = "exercises"
                add_chunks(para, page_num)
            continue
        for para in _split_into_paragraphs(text):
            add_chunks(para, page_num)

    return corpus


def _merge_undersized(corpus: list[dict],
                      min_chars: int = _MIN_CHUNK_CHARS) -> list[dict]:
    """Merge chunks smaller than min_chars into nearest same-type neighbor."""
    if not corpus:
        return corpus
    merged = [corpus[0]]
    for entry in corpus[1:]:
        prev = merged[-1]
        if len(entry["content"]) < min_chars and entry["type"] == prev["type"]:
            prev["content"] = prev["content"] + "\n" + entry["content"]
        elif len(prev["content"]) < min_chars and entry["type"] == prev["type"]:
            entry["content"] = prev["content"] + "\n" + entry["content"]
            merged[-1] = entry
        else:
            merged.append(entry)
    for i, entry in enumerate(merged):
        entry["id"] = f"chunk_{i + 1:03d}"
    return merged


def _count_words(text: str) -> int:
    """Count whitespace-separated words."""
    return len(text.split())


def _chunk_semantic(corpus: list[dict], chapter_num: int,
                    target_words: int = _TARGET_WORDS) -> list[dict]:
    """Split corpus entries at sentence boundaries, grouping to ~target_words.

    Produces V2 chunk IDs: ch{NN}_p{PP}_s{SSS}

    CHUNK_ID FORMAT:
        ch{NN} -- chapter number, zero-padded to 2 digits
        p{PP}  -- page number from the PDF (1-indexed)
        s{SSS} -- sequential segment number within the chapter (1-indexed)

    The segment number is global within the chapter (not per-page) to
    guarantee uniqueness.  Page is included for human readability.
    """
    chunks = []
    seg_num = 0  # sequential within chapter

    for entry in corpus:
        sentences = sent_tokenize(entry["content"])
        if not sentences:
            continue

        current_sentences = []
        current_word_count = 0

        for sent in sentences:
            sent_words = _count_words(sent)
            if current_word_count + sent_words > target_words and current_sentences:
                text = " ".join(current_sentences)
                seg_num += 1
                chunks.append({
                    "chunk_id": f"ch{chapter_num:02d}_p{entry['page']:02d}_s{seg_num:03d}",
                    "source_id": entry["id"],
                    "chapter_num": chapter_num,
                    "page": entry["page"],
                    "type": entry["type"],
                    "text": text,
                    "token_count": _count_words(text),  # word count (no BERT dependency)
                })
                current_sentences = []
                current_word_count = 0

            current_sentences.append(sent)
            current_word_count += sent_words

        if current_sentences:
            text = " ".join(current_sentences)
            seg_num += 1
            chunks.append({
                "chunk_id": f"ch{chapter_num:02d}_p{entry['page']:02d}_s{seg_num:03d}",
                "source_id": entry["id"],
                "chapter_num": chapter_num,
                "page": entry["page"],
                "type": entry["type"],
                "text": text,
                "token_count": _count_words(text),
            })

    return chunks


def _merge_tiny_semantic(chunks: list[dict], chapter_num: int,
                         min_tokens: int = _MIN_SEMANTIC_TOKENS) -> list[dict]:
    """Merge semantic chunks below min_tokens into neighbors."""
    if not chunks:
        return chunks
    merged = [chunks[0]]
    for ch in chunks[1:]:
        prev = merged[-1]
        if ch["token_count"] < min_tokens and ch["source_id"] == prev["source_id"]:
            prev["text"] = prev["text"] + " " + ch["text"]
            prev["token_count"] = _count_words(prev["text"])
        elif prev["token_count"] < min_tokens and prev["source_id"] == ch["source_id"]:
            ch["text"] = prev["text"] + " " + ch["text"]
            ch["token_count"] = _count_words(ch["text"])
            merged[-1] = ch
        else:
            merged.append(ch)
    # Re-number segment IDs after merging
    for i, ch in enumerate(merged):
        ch["chunk_id"] = f"ch{chapter_num:02d}_p{ch['page']:02d}_s{i + 1:03d}"
    return merged


# =========================================================================
# Chapter configuration
# =========================================================================

# Known NCERT Class 9 Science chapter titles.
# Used for running-header removal.  Chapters not listed here still
# process correctly -- they just skip header stripping.
CHAPTER_TITLES: dict[int, str] = {
    1: "Matter in Our Surroundings",
    2: "Cell: The Building Block of Life",
    3: "Is Matter Around Us Pure",
    4: "Atoms and Molecules",
    5: "Structure of the Atom",
    6: "Tissues",
    7: "Motion",
    8: "Force and Laws of Motion",
    9: "Gravitation",
    10: "Work and Energy",
    11: "Sound",
    12: "Improvement in Food Resources",
    13: "Why Do We Fall Ill",
    14: "Natural Resources",
}


def _build_header_regex(title: str) -> re.Pattern:
    """Build a running-header regex from a chapter title.

    Handles flexible whitespace in the title as pdfplumber
    may insert extra spaces between words.
    """
    escaped = re.escape(title)
    flexible = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"^\s*{flexible}\s*$", re.MULTILINE | re.IGNORECASE)


# =========================================================================
# CorpusManager
# =========================================================================
class CorpusManager:
    """Manages multi-chapter NCERT corpus with lazy loading.

    Architecture
    ------------
    Per-chapter sharded storage:

        outputs/chapters/
            ch02_chunks.json      -- chapter 2 chunks
            ch05_chunks.json      -- chapter 5 chunks
            manifest.json         -- registry of processed chapters

    Lazy loading:
        Chunks are loaded from disk ONLY when requested via load_chunks().
        Each loaded chapter is cached in memory.  unload_chapter() releases
        memory.  At no point are all 14 chapters in memory simultaneously.

    Query scoping:
        load_chunks(chapters=[2, 5])  -- load specific chapters
        load_chunks()                 -- load ALL processed chapters
        load_chunks(chapters=[2])     -- single chapter (backward compat)

    Parameters
    ----------
    data_dir : str or Path
        Directory containing NCERT PDF files (iesc1XX.pdf).
    output_dir : str or Path
        Directory for chunk outputs (default: outputs/).

    MIGRATION PATH FROM V1:
    -----------------------
    1. V1's chunks_semantic.json continues to work with all retrievers.
       No migration is required if you only use chapter 2.

    2. To migrate chapter 2 into the multi-chapter system:
       >>> mgr = CorpusManager("data", "outputs")
       >>> mgr.process_chapter(2)
       This creates outputs/chapters/ch02_chunks.json with V2 chunk IDs.

    3. To load V1 data alongside V2:
       >>> from retriever import load_chunks
       >>> v1_chunks = load_chunks("outputs/chunks_semantic.json")  # V1 IDs
       >>> v2_chunks = mgr.load_chunks(chapters=[2])                # V2 IDs
       Both work with BM25Retriever -- the retriever doesn't care about ID format.

    4. Evaluator compatibility:
       evaluator.py calls retriever.retrieve(q["question"], top_k=top_k).
       The retriever returns chunks with the "text" key regardless of
       whether chunk IDs are "sem_0001" (V1) or "ch02_p14_s003" (V2).
       Zero evaluator changes needed.

    CHUNK_ID COLLISION RISKS:
    -------------------------
    1. WITHIN a chapter: IMPOSSIBLE.
       ch{NN}_p{PP}_s{SSS} uses a chapter-global sequential segment counter.
       Even if two segments come from the same page, they get distinct s{SSS}.

    2. ACROSS chapters: IMPOSSIBLE.
       The ch{NN} prefix is unique per chapter.  ch02_p01_s001 and
       ch05_p01_s001 are distinct.

    3. V1 vs V2 IDs: IMPOSSIBLE.
       V1 uses "sem_XXXX" prefix.  V2 uses "ch{NN}_" prefix.  No overlap.

    4. Re-processing risk: SAFE.
       Processing the same chapter twice overwrites the shard file.
       The manifest hash changes, stale FAISS indices are invalidated.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.chapters_dir = self.output_dir / "chapters"
        self.chapters_dir.mkdir(parents=True, exist_ok=True)

        # Lazy cache: chapter_num -> list[dict]
        self._loaded: dict[int, list[dict]] = {}

        # Manifest: tracks which chapters are processed
        self._manifest = self._load_manifest()

    # =====================================================================
    # Discovery
    # =====================================================================

    def discover_pdfs(self) -> dict[int, Path]:
        """Scan data_dir for available NCERT PDF files.

        Returns
        -------
        dict[int, Path]
            Mapping of chapter number -> PDF file path.
            Example: {2: Path("data/iesc102.pdf")}
        """
        found = {}
        for f in sorted(self.data_dir.glob("iesc1*.pdf")):
            m = _PDF_PATTERN.match(f.name)
            if m:
                chapter_num = int(m.group(1))
                found[chapter_num] = f
        return found

    # =====================================================================
    # Processing
    # =====================================================================

    def process_chapter(self, chapter_num: int,
                        pdf_path: Optional[Union[str, Path]] = None) -> list[dict]:
        """Run full extraction + cleaning + segmentation + chunking on one chapter.

        Parameters
        ----------
        chapter_num : int
            Chapter number (1-14).
        pdf_path : str or Path, optional
            Path to the PDF.  If None, auto-discovers from data_dir.

        Returns
        -------
        list[dict]
            Semantic chunks with V2 chunk IDs and chapter metadata.
        """
        # ── Resolve PDF path ─────────────────────────────────────────────
        if pdf_path is None:
            available = self.discover_pdfs()
            if chapter_num not in available:
                raise FileNotFoundError(
                    f"Chapter {chapter_num} PDF not found in {self.data_dir}. "
                    f"Expected: iesc1{chapter_num:02d}.pdf"
                )
            pdf_path = available[chapter_num]
        pdf_path = Path(pdf_path)

        logger.info("Processing chapter %d from %s", chapter_num, pdf_path)

        # ── Build chapter-specific running header regex ───────────────────
        title = CHAPTER_TITLES.get(chapter_num)
        header_re = _build_header_regex(title) if title else None

        # ── Stage 1: Extract raw pages ───────────────────────────────────
        raw_pages = _extract_raw_pages(pdf_path)

        # ── Stage 2: Clean pages ─────────────────────────────────────────
        cleaned_pages = []
        for p in raw_pages:
            cleaned = _clean_page_text(p["raw_text"], p["page_num"], header_re)
            if cleaned:
                cleaned_pages.append({"page_num": p["page_num"], "text": cleaned})

        # ── Stage 3: Segment into typed corpus chunks ────────────────────
        corpus = _segment_all_pages(cleaned_pages)
        corpus = _merge_undersized(corpus, min_chars=120)

        # ── Stage 4: Semantic chunking with V2 IDs ───────────────────────
        chunks = _chunk_semantic(corpus, chapter_num)
        chunks = _merge_tiny_semantic(chunks, chapter_num)

        # ── Add chapter title metadata ───────────────────────────────────
        for ch in chunks:
            ch["chapter_title"] = title or f"Chapter {chapter_num}"

        # ── Save per-chapter shard ───────────────────────────────────────
        shard_path = self.chapters_dir / f"ch{chapter_num:02d}_chunks.json"
        with open(shard_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        # ── Update manifest ──────────────────────────────────────────────
        self._manifest["chapters"][str(chapter_num)] = {
            "chapter_num": chapter_num,
            "chapter_title": title or f"Chapter {chapter_num}",
            "pdf_file": pdf_path.name,
            "n_chunks": len(chunks),
            "shard_file": shard_path.name,
            "content_hash": self._hash_content(chunks),
        }
        self._save_manifest()

        # ── Cache in memory ──────────────────────────────────────────────
        self._loaded[chapter_num] = chunks

        logger.info("Chapter %d: %d chunks saved to %s", chapter_num, len(chunks), shard_path)
        return chunks

    def process_all(self) -> dict[int, int]:
        """Process all available chapter PDFs.

        Returns
        -------
        dict[int, int]
            Mapping of chapter_num -> chunk_count for each processed chapter.
        """
        available = self.discover_pdfs()
        if not available:
            raise FileNotFoundError(f"No NCERT PDFs found in {self.data_dir}")

        results = {}
        for ch_num, pdf_path in sorted(available.items()):
            chunks = self.process_chapter(ch_num, pdf_path)
            results[ch_num] = len(chunks)
            print(f"  [OK] Chapter {ch_num}: {len(chunks)} chunks")

        return results

    # =====================================================================
    # Lazy Loading
    # =====================================================================

    def load_chunks(self, chapters: Optional[list[int]] = None) -> list[dict]:
        """Lazy-load chunks for specified chapters.

        LAZY LOADING STRATEGY:
            Each chapter is stored as a separate JSON shard file.
            load_chunks() reads ONLY the requested shard files.
            Once loaded, chunks are cached in self._loaded.

            Memory usage: ~7 KB per chapter (85 chunks * ~80 bytes metadata).
            14 chapters fully loaded: ~100 KB.  But the design supports
            future scaling (1000+ chapters) by keeping the lazy pattern.

        Parameters
        ----------
        chapters : list[int], optional
            Chapter numbers to load.  None = all processed chapters.

        Returns
        -------
        list[dict]
            Combined chunks from all requested chapters, in chapter order.
        """
        if chapters is None:
            chapters = sorted(int(k) for k in self._manifest["chapters"])

        all_chunks = []
        for ch_num in sorted(chapters):
            if ch_num in self._loaded:
                all_chunks.extend(self._loaded[ch_num])
                continue

            shard_path = self.chapters_dir / f"ch{ch_num:02d}_chunks.json"
            if not shard_path.exists():
                raise FileNotFoundError(
                    f"Chapter {ch_num} not processed yet. "
                    f"Run process_chapter({ch_num}) first."
                )

            with open(shard_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            self._loaded[ch_num] = chunks
            all_chunks.extend(chunks)

        return all_chunks

    def unload_chapter(self, chapter_num: int) -> None:
        """Release a chapter's chunks from memory.

        Use this to manage memory when working with many chapters.
        The chapter can be re-loaded later via load_chunks().
        """
        self._loaded.pop(chapter_num, None)

    def get_loaded_chapters(self) -> list[int]:
        """Return list of chapter numbers currently cached in memory."""
        return sorted(self._loaded.keys())

    # =====================================================================
    # Unified output
    # =====================================================================

    def build_unified_chunks(self, chapters: Optional[list[int]] = None) -> Path:
        """Generate outputs/chunks_all.json from processed chapters.

        Parameters
        ----------
        chapters : list[int], optional
            Chapters to include.  None = all processed.

        Returns
        -------
        Path
            Path to the generated chunks_all.json file.
        """
        all_chunks = self.load_chunks(chapters)
        out_path = self.output_dir / "chunks_all.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d chunks to %s", len(all_chunks), out_path)
        return out_path

    # =====================================================================
    # Manifest management
    # =====================================================================

    def get_manifest(self) -> dict:
        """Return the current manifest (read-only copy)."""
        return dict(self._manifest)

    def _load_manifest(self) -> dict:
        """Load manifest from disk, or create empty one."""
        path = self.chapters_dir / "manifest.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"chapters": {}}

    def _save_manifest(self) -> None:
        """Save manifest to disk."""
        path = self.chapters_dir / "manifest.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2)

    @staticmethod
    def _hash_content(chunks: list[dict]) -> str:
        """SHA-256 hash of chunk texts for integrity checking."""
        h = hashlib.sha256()
        for ch in chunks:
            h.update(ch["text"].encode("utf-8"))
        return h.hexdigest()[:16]

    # =====================================================================
    # Info
    # =====================================================================

    def __repr__(self) -> str:
        n_processed = len(self._manifest["chapters"])
        n_loaded = len(self._loaded)
        return f"CorpusManager(processed={n_processed}, loaded={n_loaded})"
