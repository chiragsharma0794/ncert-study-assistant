"""
src/retrieval_sufficiency.py -- Multi-signal retrieval sufficiency assessment.

Determines whether retrieved chunks contain enough information to
construct a grounded answer, and produces structured partial-answer
responses when context is insufficient.

DETECTION METHOD: Multi-Signal Heuristic (Option B)
====================================================
The original implementation used a single score threshold (SCORE_THRESHOLD=0.3)
calibrated for cross-encoder normalized scores.  This broke when V2 migrated
to weighted BM25 + FAISS score fusion, which inflates scores to 0.5-0.6
even for completely unrelated queries.

The multi-signal approach uses FIVE independent signals:

  A. Chunk count        -- zero chunks = always insufficient
  B. Top score          -- uncalibrated heuristic, not a probability
  C. Score margin       -- gap between top-1 and top-3 scores
  D. Lexical overlap    -- query term coverage in retrieved text
  E. Source diversity   -- unique pages/chapters as supporting signal

Each signal contributes a weighted vote to a final confidence score.
The decision is conservative: high hybrid scores alone do NOT guarantee
sufficiency if lexical overlap is near zero.

WHY NOT A SINGLE THRESHOLD?
----------------------------
Hybrid retriever scores are NOT calibrated probabilities.  They are
weighted fusions of BM25 min-max normalized scores and FAISS cosine
similarities shifted to [0,1].  This means:
  - A score of 0.6 does NOT mean "60% relevant"
  - Unrelated queries can produce scores of 0.5-0.6
  - The score distribution depends on corpus size and vocabulary

A single threshold cannot distinguish "high score because relevant"
from "high score because of score fusion arithmetic".  Lexical overlap
provides an orthogonal signal that catches this failure mode.

TODO: Future calibration using labeled validation data
------------------------------------------------------
The weights and thresholds below are empirically reasonable but not
optimized.  A proper calibration would:
  1. Collect ~100 labeled (query, chunks, is_relevant) examples
  2. Fit logistic regression on the signal vector
  3. Replace the hand-tuned weights with learned coefficients
  4. Set the decision threshold using precision-recall analysis

Usage:
    from retrieval_sufficiency import assess_retrieval_sufficiency
    result = assess_retrieval_sufficiency(chunks, query="What is a cell?")
    if not result.is_sufficient:
        return build_insufficient_response(query, chunks, result)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional


# =========================================================================
# Configuration
# =========================================================================

# SCORE THRESHOLD (uncalibrated heuristic)
# Used as ONE signal among many.  Not a hard decision boundary.
#
# NOTE: Hybrid retriever scores are NOT calibrated probabilities.
# This threshold is a soft heuristic, not a precise cutoff.
# See module docstring for rationale.
SCORE_THRESHOLD = 0.3

# PAGE COVERAGE THRESHOLD
# Supporting signal only.  Not required for single-definition queries.
PAGE_COVERAGE_THRESHOLD = 2

# MINIMUM CHUNKS for assessment
MIN_CHUNKS_REQUIRED = 1

# CONFIDENCE BANDS (for reporting, NOT for decision-making)
# These categorize the top score for human-readable output.
# The actual sufficiency decision uses multi-signal logic.
CONFIDENCE_BANDS = {
    "high":     (0.7, float("inf")),
    "medium":   (0.5, 0.7),
    "low":      (0.3, 0.5),
    "critical": (0.0, 0.3),
}

# ── Multi-signal weights ─────────────────────────────────────────────────
# These control relative importance of each signal in the final confidence.
# Sum to 1.0 for interpretability.
#
# Lexical overlap is weighted highest because it is the most retriever-
# agnostic signal: if query terms don't appear in retrieved chunks, the
# chunks are unlikely to answer the question regardless of score.
_W_SCORE    = 0.25   # Top retrieval score (uncalibrated)
_W_MARGIN   = 0.10   # Score margin (top-1 vs top-3)
_W_OVERLAP  = 0.45   # Lexical overlap between query and chunks
_W_DIVERSITY = 0.10  # Source page diversity
_W_COUNT    = 0.10   # Chunk count signal

# Decision threshold on weighted confidence
# Below this → insufficient.  Conservative: prefer false negatives
# (marking as insufficient) over false positives (hallucinating).
_CONFIDENCE_THRESHOLD = 0.40

# Lexical overlap floor: if overlap is below this, mark insufficient
# regardless of other signals.  This catches the "inflated hybrid score
# but zero topical relevance" failure mode.
_OVERLAP_FLOOR = 0.05

# ── Stopwords for lexical overlap ────────────────────────────────────────
# Minimal English stopword set.  We only need to filter function words
# that appear in every chunk regardless of topic.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "its", "this", "that", "these", "those",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "about", "between",
    "and", "but", "or", "nor", "not", "no", "so", "if", "then",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "all", "each", "every", "both", "few", "more", "most", "some",
    "such", "only", "very", "also", "just",
})

_WORD_RE = re.compile(r"\w+")


# =========================================================================
# SufficiencyResult dataclass
# =========================================================================
@dataclass
class SufficiencyResult:
    """Result of retrieval sufficiency assessment.

    Fields
    ------
    is_sufficient : bool
        True if retrieved context is adequate for a grounded answer.
    confidence : float
        Weighted multi-signal confidence in [0.0, 1.0].
        NOT a probability.  Higher = more evidence of relevance.
    confidence_band : str
        One of: "high", "medium", "low", "critical".
        Based on top score only, for backward-compatible reporting.
    max_score : float
        Maximum retrieval score among retrieved chunks.
    avg_score : float
        Mean retrieval score across all chunks.
    unique_pages : int
        Number of unique pages covered by retrieved chunks.
    n_chunks : int
        Total number of chunks assessed.
    reason : str
        Human-readable explanation of the assessment.
    found_topics : list[str]
        Chunk types/topics that WERE found in the context.
    page_numbers : list[int]
        Sorted list of page numbers covered by retrieved chunks.
    signals : dict
        Raw signal values used in the decision.  Shape:
        {
            "num_chunks": int,
            "top_score": float,
            "score_margin": float,
            "lexical_overlap": float,
            "source_diversity": int,
        }
    """
    is_sufficient: bool
    confidence: float = 0.0
    confidence_band: str = "high"
    max_score: float = 0.0
    avg_score: float = 0.0
    unique_pages: int = 0
    n_chunks: int = 0
    reason: str = ""
    found_topics: list[str] = field(default_factory=list)
    page_numbers: list[int] = field(default_factory=list)
    signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# =========================================================================
# InsufficientContextResponse dataclass
# =========================================================================
@dataclass
class InsufficientContextResponse:
    """Structured response when retrieved context is insufficient.

    This response tells the student:
    1. What WAS found in the context (partial information).
    2. What was NOT found (the gap).
    3. A follow-up suggestion (page numbers or related terms).

    Fields
    ------
    answer : str
        Partial answer constructed from available context.
        Prefixed with a confidence warning.
    found_summary : str
        What information was found in the context.
    missing_info : str
        What information is missing or insufficient.
    follow_up : str
        Actionable suggestion for the student.
    sources : list[str]
        chunk_ids from the partial context used.
    refused : bool
        Always False (partial answer, not refusal).
    is_partial : bool
        Always True (marks this as an insufficient-context response).
    sufficiency : dict
        The raw SufficiencyResult for audit/debugging.
    """
    answer: str = ""
    found_summary: str = ""
    missing_info: str = ""
    follow_up: str = ""
    sources: list[str] = field(default_factory=list)
    refused: bool = False
    is_partial: bool = True
    sufficiency: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# =========================================================================
# Signal computation helpers
# =========================================================================

def _tokenize_important(text: str) -> set[str]:
    """Extract important (non-stopword, len >= 3) tokens from text.

    Parameters
    ----------
    text : str
        Input text to tokenize.

    Returns
    -------
    set[str]
        Lowercased tokens with stopwords and short tokens removed.
    """
    tokens = _WORD_RE.findall(text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) >= 3}


def _compute_lexical_overlap(query: str, chunks: list[dict]) -> float:
    """Compute fraction of important query terms found in chunk text.

    This is the most retriever-agnostic signal: if the query asks about
    "mitochondria" but no chunk contains that word, the chunks are almost
    certainly insufficient regardless of their retrieval score.

    Parameters
    ----------
    query : str
        The student's question.
    chunks : list[dict]
        Retrieved chunks with 'content' or 'text' key.

    Returns
    -------
    float
        Fraction of important query tokens found in chunk text.
        Range: [0.0, 1.0].  Returns 0.0 if query has no important tokens.
    """
    query_terms = _tokenize_important(query)
    if not query_terms:
        return 0.0

    # Combine all chunk text into a single searchable string
    chunk_text_combined = " ".join(
        (c.get("content") or c.get("text", "")).lower()
        for c in chunks
    )

    matched = sum(1 for t in query_terms if t in chunk_text_combined)
    return matched / len(query_terms)


def _compute_score_margin(scores: list[float]) -> float:
    """Compute score margin between top-1 and top-3 (or last) score.

    A large margin indicates the top result is significantly more relevant
    than lower-ranked results, which is a positive signal.
    A very small margin suggests no chunk stands out — weak retrieval.

    Parameters
    ----------
    scores : list[float]
        Sorted descending retrieval scores.

    Returns
    -------
    float
        Score margin (top_1 - reference_score).
        Returns 0.0 if fewer than 2 scores.
    """
    if len(scores) < 2:
        return 0.0

    top_score = scores[0]
    # Compare against 3rd score if available, otherwise last score
    ref_idx = min(2, len(scores) - 1)
    return top_score - scores[ref_idx]


def _score_to_signal(score: float) -> float:
    """Convert a raw retrieval score to a [0, 1] signal value.

    Uses a simple clamp since hybrid scores are already in [0, 1].
    For raw BM25 scores (unbounded), this clamps to [0, 1].

    This is NOT a calibration — it's a normalization.
    The score is still treated as an uncalibrated heuristic.
    """
    return max(0.0, min(1.0, score))


def _diversity_to_signal(unique_pages: int) -> float:
    """Convert page diversity count to a [0, 1] signal value.

    Saturates at 3 pages (most NCERT queries are answered within 3 pages).
    """
    return min(1.0, unique_pages / 3.0)


def _count_to_signal(n_chunks: int) -> float:
    """Convert chunk count to a [0, 1] signal value.

    Saturates at 3 chunks (typical top_k is 3-5).
    """
    return min(1.0, n_chunks / 3.0)


def _margin_to_signal(margin: float) -> float:
    """Convert score margin to a [0, 1] signal value.

    A margin of 0.2+ is considered strong discrimination.
    """
    return min(1.0, margin / 0.2)


# =========================================================================
# Core assessment function
# =========================================================================
def assess_retrieval_sufficiency(
    chunks: list[dict],
    query: str = "",
    score_threshold: float = SCORE_THRESHOLD,
    page_threshold: int = PAGE_COVERAGE_THRESHOLD,
) -> SufficiencyResult:
    """Assess whether retrieved chunks are sufficient for a grounded answer.

    Uses a multi-signal heuristic that combines:
      A. Chunk count (zero chunks = always insufficient)
      B. Top retrieval score (uncalibrated heuristic)
      C. Score margin (top-1 vs lower-ranked scores)
      D. Lexical overlap (query terms found in chunk text)
      E. Source/page diversity (supporting signal)

    IMPORTANT: Hybrid retriever scores are NOT calibrated probabilities.
    A score of 0.6 does NOT mean "60% relevant".  Score fusion (BM25 +
    FAISS weighted combination) can produce scores of 0.5-0.6 for
    completely unrelated queries.  This function uses lexical overlap as
    the primary grounding signal to catch these false positives.

    Parameters
    ----------
    chunks : list[dict]
        Retrieved chunks, each with at least 'score' and either
        'content' or 'text'.  May also have 'page', 'type', 'chunk_id'.
    query : str, optional
        The original query string.  If provided, enables lexical overlap
        signal.  If empty, the function falls back to score-only logic.
    score_threshold : float
        Soft score threshold (used as one signal, not a hard cutoff).
    page_threshold : int
        Minimum unique pages for broad queries (supporting signal).

    Returns
    -------
    SufficiencyResult
        Assessment with multi-signal confidence, signals dict, and
        human-readable reason.

    Backward Compatibility
    ----------------------
    When ``query`` is empty (legacy callers), the function falls back to
    a score-only heuristic that matches the original V1 behavior:
      - max_score >= score_threshold → SUFFICIENT
      - max_score < score_threshold → INSUFFICIENT (with page-based reason)
    """
    # ── Case 1: No chunks → always insufficient ─────────────────────
    if not chunks or len(chunks) < MIN_CHUNKS_REQUIRED:
        return SufficiencyResult(
            is_sufficient=False,
            confidence=0.0,
            confidence_band="critical",
            reason="No chunks retrieved. The topic may not be in the corpus.",
            signals={
                "num_chunks": 0,
                "top_score": 0.0,
                "score_margin": 0.0,
                "lexical_overlap": 0.0,
                "source_diversity": 0,
            },
        )

    # ── Extract base metrics ─────────────────────────────────────────
    scores = sorted(
        [c.get("score", 0.0) for c in chunks], reverse=True
    )
    max_score = scores[0]
    avg_score = sum(scores) / len(scores)
    pages = sorted(set(
        c.get("page", 0) for c in chunks if c.get("page")
    ))
    unique_pages = len(pages)
    types_found = sorted(set(c.get("type", "unknown") for c in chunks))
    n_chunks = len(chunks)

    # ── Confidence band (for reporting only) ─────────────────────────
    confidence_band = "critical"
    for band, (low, high) in CONFIDENCE_BANDS.items():
        if low <= max_score < high:
            confidence_band = band
            break

    # ── Compute signals ──────────────────────────────────────────────
    score_margin = _compute_score_margin(scores)
    lexical_overlap = (
        _compute_lexical_overlap(query, chunks) if query else -1.0
    )

    signals = {
        "num_chunks": n_chunks,
        "top_score": round(max_score, 4),
        "score_margin": round(score_margin, 4),
        "lexical_overlap": round(lexical_overlap, 4),
        "source_diversity": unique_pages,
    }

    base_result = dict(
        max_score=round(max_score, 2),
        avg_score=round(avg_score, 2),
        unique_pages=unique_pages,
        n_chunks=n_chunks,
        confidence_band=confidence_band,
        found_topics=types_found,
        page_numbers=pages,
        signals=signals,
    )

    # ── Decision logic ───────────────────────────────────────────────
    #
    # Two paths:
    #   Path A (query provided): Multi-signal weighted confidence
    #   Path B (no query / legacy): Score-only fallback

    if query:
        # ── Path A: Multi-signal decision ────────────────────────────
        #
        # Hard floor: if lexical overlap is near zero, the chunks almost
        # certainly don't address the query.  Inflated hybrid scores
        # cannot override this.
        if lexical_overlap < _OVERLAP_FLOOR:
            return SufficiencyResult(
                is_sufficient=False,
                confidence=round(lexical_overlap, 4),
                reason=(
                    f"Near-zero lexical overlap ({lexical_overlap:.2f}): "
                    f"query terms not found in retrieved chunks despite "
                    f"top_score={max_score:.2f}. The retriever score is "
                    f"likely inflated by score fusion arithmetic."
                ),
                **base_result,
            )

        # Compute weighted confidence from all signals
        confidence = (
            _W_SCORE    * _score_to_signal(max_score)
            + _W_MARGIN   * _margin_to_signal(score_margin)
            + _W_OVERLAP  * lexical_overlap
            + _W_DIVERSITY * _diversity_to_signal(unique_pages)
            + _W_COUNT    * _count_to_signal(n_chunks)
        )
        confidence = round(max(0.0, min(1.0, confidence)), 4)

        is_sufficient = confidence >= _CONFIDENCE_THRESHOLD

        if is_sufficient:
            reason = (
                f"Multi-signal assessment: confidence={confidence:.2f} "
                f"(>= {_CONFIDENCE_THRESHOLD}). "
                f"Signals: overlap={lexical_overlap:.2f}, "
                f"top_score={max_score:.2f}, margin={score_margin:.2f}, "
                f"pages={unique_pages}, chunks={n_chunks}."
            )
        else:
            reason = (
                f"Insufficient multi-signal confidence: {confidence:.2f} "
                f"(< {_CONFIDENCE_THRESHOLD}). "
                f"Signals: overlap={lexical_overlap:.2f}, "
                f"top_score={max_score:.2f}, margin={score_margin:.2f}, "
                f"pages={unique_pages}, chunks={n_chunks}."
            )

        return SufficiencyResult(
            is_sufficient=is_sufficient,
            confidence=confidence,
            reason=reason,
            **base_result,
        )

    # ── Path B: Legacy score-only fallback ───────────────────────────
    # When no query is provided, fall back to original V1 logic.
    # This preserves backward compatibility for callers that don't
    # pass the query string.

    if max_score >= score_threshold:
        return SufficiencyResult(
            is_sufficient=True,
            confidence=round(_score_to_signal(max_score), 4),
            reason=(
                f"Score-only assessment (no query provided): "
                f"max_score={max_score:.2f} >= {score_threshold}."
            ),
            **base_result,
        )

    # Low score fallback
    if unique_pages < page_threshold:
        reason = (
            f"Low retrieval confidence: max_score={max_score:.1f} "
            f"(< {score_threshold}) with only {unique_pages} page(s). "
            f"The specific information may not be in the retrieved context."
        )
    else:
        reason = (
            f"Topical drift detected: max_score={max_score:.1f} "
            f"(< {score_threshold}) despite {unique_pages} pages. "
            f"The chunks may be related but do not directly address the query."
        )

    return SufficiencyResult(
        is_sufficient=False,
        confidence=round(_score_to_signal(max_score), 4),
        reason=reason,
        **base_result,
    )


# =========================================================================
# Insufficient context response builder
# =========================================================================
def build_insufficient_response(
    query: str,
    chunks: list[dict],
    sufficiency: SufficiencyResult,
) -> InsufficientContextResponse:
    """Build a structured partial-answer response for insufficient context.

    Follows three rules:
    1. Do NOT fabricate. Only report what is in the chunks.
    2. Tell the student what was found AND what was not found.
    3. Suggest a follow-up using page numbers from chunk metadata.

    Parameters
    ----------
    query : str
        The original student query.
    chunks : list[dict]
        The retrieved chunks (may be partially relevant).
    sufficiency : SufficiencyResult
        The assessment result from assess_retrieval_sufficiency().

    Returns
    -------
    InsufficientContextResponse
        Structured partial answer with follow-up suggestions.
    """
    # ── What was found ───────────────────────────────────────────────
    if chunks and sufficiency.max_score > 0:
        # Extract brief snippets from top chunks (first 80 chars each)
        top_chunks = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)[:3]
        found_snippets = []
        for c in top_chunks:
            text = (c.get("content") or c.get("text", ""))[:100].replace("\n", " ").strip()
            cid = c.get("chunk_id", "?")
            found_snippets.append(f"[{cid}] {text}...")
        found_summary = (
            f"I found some related content ({sufficiency.n_chunks} chunks "
            f"from {sufficiency.unique_pages} page(s)), but it may not "
            f"fully cover your question:\n" + "\n".join(found_snippets)
        )
    else:
        found_summary = "I could not find relevant content for this question."

    # ── What was NOT found ───────────────────────────────────────────
    missing_info = (
        f"The retrieved context has low relevance to '{query}' "
        f"(confidence: {sufficiency.confidence_band}, "
        f"max score: {sufficiency.max_score:.1f}). "
        f"The specific information you're looking for may not be "
        f"in the retrieved chunks."
    )

    # ── Follow-up suggestion ─────────────────────────────────────────
    follow_up_parts = []

    # Suggest page numbers if available
    if sufficiency.page_numbers:
        page_str = ", ".join(str(p) for p in sufficiency.page_numbers[:5])
        follow_up_parts.append(
            f"Check page(s) {page_str} in your textbook for more details"
        )

    # Suggest related terms based on chunk types found
    if sufficiency.found_topics:
        topics_str = ", ".join(sufficiency.found_topics)
        follow_up_parts.append(
            f"Try searching for related terms: {topics_str}"
        )

    if follow_up_parts:
        follow_up = ". ".join(follow_up_parts) + "."
    else:
        follow_up = "Try rephrasing your question or checking the textbook index."

    # ── Build answer text ────────────────────────────────────────────
    answer = (
        f"I found limited information about this in the textbook. "
        f"{found_summary}\n\n"
        f"What's missing: {missing_info}\n\n"
        f"Suggestion: {follow_up}"
    )

    # ── Extract source chunk_ids ─────────────────────────────────────
    sources = [c.get("chunk_id", "") for c in chunks[:5] if c.get("chunk_id")]

    return InsufficientContextResponse(
        answer=answer,
        found_summary=found_summary,
        missing_info=missing_info,
        follow_up=follow_up,
        sources=sources,
        sufficiency=sufficiency.to_dict(),
    )


# =========================================================================
# System prompt addition for insufficient context
# =========================================================================
INSUFFICIENT_CONTEXT_PROMPT_ADDITION = """\

HANDLING INSUFFICIENT CONTEXT:
If the context chunks do not contain enough information to fully answer
the query, you MUST:
1. Answer ONLY what the context supports. Do NOT fabricate or infer.
2. Explicitly state what information is MISSING: "The context does not
   contain information about [specific aspect]."
3. Set is_partial to true in your JSON response.
4. If page numbers are available in the chunk metadata, suggest:
   "Try checking page [X] for more details."
5. NEVER guess, speculate, or fill gaps with training data.
"""
