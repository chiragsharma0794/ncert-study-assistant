"""
src/summarizer.py -- Grounded summarization for the NCERT RAG pipeline.

Produces structured, citation-backed summaries from retrieved chunks.
Every claim traces back to a specific chunk_id.  No fabrication.

Usage:
    from summarizer import GroundedSummarizer
    summarizer = GroundedSummarizer(retriever, api_key="...")
    result = summarizer.summarize("cell membrane")
    print(result.overview)
    print(result.bullets)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from google import genai
from google.genai import types


# =========================================================================
# SummaryResponse dataclass
# =========================================================================
@dataclass
class SummaryResponse:
    """Structured summary output with citation tracking.

    Fields
    ------
    topic : str
        The topic that was summarized.
    overview : str
        3-sentence grounded overview of the topic.
    bullets : list[str]
        Key concepts as bullet points (max 5).  Each bullet contains
        at least one [chunk_id] citation.
    chunk_ids : list[str]
        All chunk_ids cited across overview and bullets (deduplicated).
    is_partial : bool
        True if the context was insufficient to fully cover the topic.
        A partial summary still contains whatever WAS found.
    missing_topics : list[str]
        Aspects of the topic the model identified as NOT covered by
        the retrieved context.  Empty if is_partial is False.
    refused : bool
        True if NO relevant context was found at all (full refusal).
    model : str
        Model name used for generation.
    n_chunks_used : int
        Number of chunks retrieved and provided as context.
    """

    topic: str
    overview: str = ""
    bullets: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    is_partial: bool = False
    missing_topics: list[str] = field(default_factory=list)
    refused: bool = False
    model: str = ""
    n_chunks_used: int = 0

    def to_dict(self) -> dict:
        """Convert to plain dict (JSON-serializable)."""
        return asdict(self)


# =========================================================================
# Summarization system prompt
# =========================================================================

# DESIGN: HOW THIS DIFFERS FROM Q&A
#
# Q&A (GroundedGenerator):
#   - Input: a specific QUESTION with a single expected answer
#   - Output: a direct answer paragraph with inline [1], [2] citations
#   - Refusal: binary (answer or refuse)
#
# Summarization (GroundedSummarizer):
#   - Input: a TOPIC (not a question) -- broader, multi-faceted
#   - Output: STRUCTURED format (overview + bullet points)
#   - Citations: per-bullet, using actual [chunk_id] strings
#   - Refusal: GRADUATED -- can be partial (some aspects found, others missing)
#   - Unique prompt elements:
#     * Explicit instruction to produce exactly 3 overview sentences
#     * Bullet-point format enforcement with per-bullet citation requirement
#     * "MISSING" section for transparency about gaps
#     * JSON output format for reliable parsing

SUMMARIZATION_SYSTEM_PROMPT = """\
You are a study assistant that creates structured summaries from NCERT textbook content.

Summarize ONLY from the context chunks provided below. Context is labeled with
chunk_ids in the format [N] {chunk_id: "..."}.
If the context does not contain information about the topic, respond with the
refusal JSON shown below.

RULES -- follow these STRICTLY:
1. Every claim must cite the chunk it came from using [chunk_id] notation.
2. If the context does not contain enough information about the topic,
   state explicitly what aspects are MISSING.

You MUST NOT:
- Use any knowledge from your training data. Your ONLY knowledge source is
  the context below.
- Add facts, definitions, or examples not present in the provided context.
- Paraphrase the refusal phrase. Use it exactly: "I could not find this in the textbook."
- Make claims without citing a source chunk_id.

OUTPUT FORMAT -- respond ONLY with valid JSON. No preamble, no explanation,
no markdown fences.
{
  "overview": "Three sentences summarizing the topic. Each sentence cites its source [chunk_id].",
  "bullets": [
    "Key concept 1 with citation [chunk_id]",
    "Key concept 2 with citation [chunk_id]"
  ],
  "missing": ["aspect not covered", "another gap"],
  "is_partial": false
}

RULES FOR EACH SECTION:
- OVERVIEW: Exactly 3 sentences. Each sentence must cite at least one [chunk_id].
- BULLETS: 1 to 5 key concepts. Each bullet must cite at least one [chunk_id].
- MISSING: List any aspects of the topic NOT covered in the context.
  If everything is covered, use an empty list [].
- IS_PARTIAL: Set to true if the context covers less than half of what a
  student would need to understand this topic. Set to false otherwise.

IF NO RELEVANT CONTEXT EXISTS:
{
  "overview": "I could not find this in the textbook.",
  "bullets": [],
  "missing": ["entire topic not found in context"],
  "is_partial": true
}
"""

# ── Summarization user prompt template ───────────────────────────────────────
SUMMARIZATION_USER_TEMPLATE = """\
CONTEXT CHUNKS (from NCERT textbook):
{context_block}

User query: Summarize {topic}
"""


# =========================================================================
# Sufficient context threshold
# =========================================================================

# DESIGN DECISION: SUFFICIENT CONTEXT
#
# "Sufficient context" is defined by TWO heuristics:
#
# 1. RELEVANCE SCORE THRESHOLD (retriever-level):
#    If the top-1 retrieved chunk has a HybridRetriever score < 0.15
#    (sigmoid-normalized cross-encoder), the topic likely has no
#    relevant content in the corpus.  Trigger FULL REFUSAL.
#
# 2. COVERAGE (LLM-level):
#    The LLM itself judges whether the context covers the topic,
#    via the is_partial and missing fields in its JSON output.
#    This is more nuanced than a score threshold -- the model can
#    identify WHICH aspects are missing.
#
# The two-layer approach catches both:
#   - Retrieval failure (no relevant chunks found) -> fast refusal
#   - Partial coverage (some chunks found but topic not fully covered)
#     -> partial summary with transparency

_MIN_RELEVANCE_SCORE = 0.15  # below this, top chunk is likely irrelevant


# =========================================================================
# GroundedSummarizer
# =========================================================================
class GroundedSummarizer:
    """Grounded summarization engine for NCERT topics.

    Flow:
        1. Retrieve top_k=8 chunks for the topic using the provided retriever
        2. Check relevance threshold (fast refusal if no relevant chunks)
        3. Build context prompt with chunk_ids as citation labels
        4. Call Gemini with SUMMARIZATION_SYSTEM_PROMPT (temperature=0)
        5. Parse structured JSON response into SummaryResponse

    Parameters
    ----------
    retriever : object
        Any retriever with retrieve(query, top_k) method.
        Typically HybridRetriever for best results.
    api_key : str, optional
        Gemini API key.  If None, reads from GEMINI_API_KEY env var.
    model_name : str
        Gemini model to use.
    top_k : int
        Number of chunks to retrieve for context.  Default: 8.
        Higher than Q&A (5) because summaries need broader coverage.
    min_relevance : float
        Minimum top-1 score for the topic to be considered in-scope.
    """

    def __init__(
        self,
        retriever: object,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash-lite",
        top_k: int = 8,
        min_relevance: float = _MIN_RELEVANCE_SCORE,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set GEMINI_API_KEY in .env or "
                "pass api_key= to GroundedSummarizer()."
            )
        self.model_name = model_name
        self.top_k = top_k
        self.min_relevance = min_relevance
        self._retriever = retriever
        self._client = genai.Client(api_key=self.api_key)

    def summarize(self, topic: str) -> SummaryResponse:
        """Produce a grounded summary for a topic.

        Parameters
        ----------
        topic : str
            The topic to summarize (e.g., "cell membrane", "mitochondria").

        Returns
        -------
        SummaryResponse
            Structured summary with citations and coverage metadata.
        """
        # ── Stage 1: Retrieve chunks ─────────────────────────────────────
        retrieved = self._retriever.retrieve(topic, top_k=self.top_k)

        # ── Stage 2: Relevance check (fast refusal) ─────────────────────
        if not retrieved or retrieved[0].get("score", 0) < self.min_relevance:
            return SummaryResponse(
                topic=topic,
                overview="I could not find this in the textbook.",
                refused=True,
                is_partial=True,
                missing_topics=[f"'{topic}' not found in context"],
                model=self.model_name,
                n_chunks_used=0,
            )

        # ── Stage 3: Build prompt ────────────────────────────────────────
        context_block = self._build_context(retrieved)
        user_prompt = SUMMARIZATION_USER_TEMPLATE.format(
            context_block=context_block,
            topic=topic,
        )

        # ── Stage 4: Generate (temperature=0, deterministic) ─────────────
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SUMMARIZATION_SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=1024,
            ),
        )

        raw_text = response.text.strip() if response.text else ""

        # ── Stage 5: Parse JSON response ─────────────────────────────────
        return self._parse_response(raw_text, topic, retrieved)

    def _build_context(self, chunks: list[dict]) -> str:
        """Build context block with chunk_ids as citation labels."""
        lines = []
        for i, chunk in enumerate(chunks, 1):
            cid = chunk["chunk_id"]
            page = f" (page {chunk['page']})" if "page" in chunk else ""
            ctype = f" [{chunk['type']}]" if "type" in chunk else ""
            lines.append(
                f"[{i}] {{chunk_id: \"{cid}\"}}{ctype}{page} \u2014 {chunk.get('content') or chunk.get('text', '')}"
            )
        return "\n\n".join(lines)

    def _parse_response(self, raw: str, topic: str,
                        chunks: list[dict]) -> SummaryResponse:
        """Parse the LLM's JSON response into a SummaryResponse."""
        # ── Detect refusal ───────────────────────────────────────────────
        refused = "could not find" in raw.lower()

        # ── Extract JSON from response ───────────────────────────────────
        # The model may wrap JSON in ```json ... ``` code fences
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if not json_match:
            # Fallback: treat entire response as plain text answer
            return SummaryResponse(
                topic=topic,
                overview=raw,
                refused=refused,
                model=self.model_name,
                n_chunks_used=len(chunks),
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return SummaryResponse(
                topic=topic,
                overview=raw,
                refused=refused,
                model=self.model_name,
                n_chunks_used=len(chunks),
            )

        # ── Extract cited chunk_ids from overview + bullets ──────────────
        all_text = data.get("overview", "") + " " + " ".join(data.get("bullets", []))
        # Match both V1 IDs [sem_0001] and V2 IDs [ch02_p14_s003]
        cited_ids = list(dict.fromkeys(re.findall(r'\[([a-z0-9_]+)\]', all_text)))

        overview = data.get("overview", "")
        is_partial = data.get("is_partial", False)
        missing = data.get("missing", [])

        # Refusal detection: check both explicit flag and overview content
        if not refused:
            refused = "could not find" in overview.lower()

        return SummaryResponse(
            topic=topic,
            overview=overview,
            bullets=data.get("bullets", []),
            chunk_ids=cited_ids,
            is_partial=is_partial,
            missing_topics=missing if isinstance(missing, list) else [missing],
            refused=refused,
            model=self.model_name,
            n_chunks_used=len(chunks),
        )

    def __repr__(self) -> str:
        return f"GroundedSummarizer(model={self.model_name!r}, top_k={self.top_k})"
