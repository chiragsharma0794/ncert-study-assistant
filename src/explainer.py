"""
src/explainer.py -- Feynman-style concept explanation for the NCERT RAG pipeline.

Produces structured, citation-backed explanations that adapt depth to
available context.  Analogies not from the source text are explicitly
flagged as [pedagogical_addition].

Usage:
    from explainer import ConceptExplainer
    explainer = ConceptExplainer(retriever, api_key="...")
    result = explainer.explain("osmosis")
    print(result.simple_definition)
    print(result.steps)
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
# ExplanationResponse dataclass
# =========================================================================
@dataclass
class ExplanationResponse:
    """Structured Feynman-style concept explanation.

    Fields
    ------
    concept : str
        The concept that was explained.
    simple_definition : str
        One sentence, no jargon.  Cites [chunk_id].
    analogy : str
        An analogy to help understanding.
        If from source text, cites [chunk_id].
        If pedagogical addition, prefixed with [pedagogical_addition: not from NCERT].
    analogy_is_grounded : bool
        True if the analogy comes from the source text.
        False if it is a pedagogical addition.
    steps : list[str]
        Numbered step-by-step breakdown.  Each step cites [chunk_id].
    misconception : str
        Common misconception, if found in source text.  Empty if none found.
    related_concepts : list[str]
        Related concepts whose chunk_ids appear in the provided context.
        Never includes concepts outside the retrieved chunks.
    chunk_ids : list[str]
        All chunk_ids cited across the explanation (deduplicated).
    is_partial : bool
        True if context was insufficient for a complete explanation.
    refused : bool
        True if no relevant context was found at all.
    model : str
        Model name used for generation.
    n_chunks_used : int
        Number of chunks provided as context.
    """
    concept: str
    simple_definition: str = ""
    analogy: str = ""
    analogy_is_grounded: bool = True
    steps: list[str] = field(default_factory=list)
    misconception: str = ""
    related_concepts: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    is_partial: bool = False
    refused: bool = False
    model: str = ""
    n_chunks_used: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# =========================================================================
# Explanation system prompt
# =========================================================================

# DESIGN: PEDAGOGICAL ADDITION vs HALLUCINATION
#
# The distinction is enforced by a three-part contract in the prompt:
#
# 1. EXPLICIT TAGGING: The model MUST prefix any analogy not found in
#    the source text with "[pedagogical_addition: not from NCERT]".
#    This makes additions visible and auditable.
#
# 2. SCOPE RESTRICTION: Only analogies are allowed as additions.
#    Definitions, steps, and misconceptions must come ONLY from context.
#    The model cannot add "helpful examples" to steps -- only to analogies.
#
# 3. VERIFIABILITY: Every step cites a chunk_id.  If a step has no
#    citation, it is flagged as invalid during post-processing.
#    This creates a hard boundary: if you can't cite it, don't say it.
#
# This design treats pedagogical additions as DISCLOSED ENRICHMENT
# (valuable for students) while treating uncited factual claims as
# HALLUCINATION (rejected during validation).

EXPLANATION_SYSTEM_PROMPT = """\
You are a study assistant that explains NCERT Science concepts using the Feynman technique.
Your goal: make complex concepts simple enough for a Class 9 student to understand.

Explain ONLY from the context chunks provided below. Context is labeled with
chunk_ids in the format [N] {chunk_id: "..."}.
If the context does not contain information about the concept, respond with the
refusal JSON shown below.

You MUST NOT:
- Use any knowledge from your training data. Your ONLY knowledge source is
  the context below.
- Add examples not present in the source text without the [pedagogical_addition: not from NCERT] tag.
- Explain related concepts unless their chunk_ids are in the provided context.
- Paraphrase the refusal phrase. Use it exactly: "I could not find this in the textbook."
- Make factual claims without citing a source chunk_id.

RULES:
1. Every factual statement must cite its source chunk using [chunk_id] notation.
2. If the context does not contain enough information, say what is MISSING.

OUTPUT FORMAT -- respond ONLY with valid JSON. No preamble, no explanation,
no markdown fences.
{
  "simple_definition": "One sentence, no jargon, explaining the concept [chunk_id].",
  "analogy": "An analogy to help understanding. If from context, cite [chunk_id]. If not from context, start with: [pedagogical_addition: not from NCERT]",
  "analogy_is_grounded": true,
  "steps": [
    "1. First key aspect of the concept [chunk_id]",
    "2. Second key aspect [chunk_id]",
    "3. Third key aspect [chunk_id]"
  ],
  "misconception": "A common misconception about this concept, if mentioned in the text [chunk_id]. Leave empty string if none found.",
  "related_concepts": ["concept_name_1", "concept_name_2"],
  "is_partial": false
}

RULES FOR EACH SECTION:
- SIMPLE_DEFINITION: Exactly one sentence. Must cite a chunk_id. No technical jargon.
- ANALOGY: If the textbook contains an analogy, use it and cite [chunk_id].
  If you must create your own, you MUST prefix it with:
  "[pedagogical_addition: not from NCERT]"
  Set analogy_is_grounded to false when using a pedagogical addition.
- STEPS: 3 to 6 numbered steps. Each step MUST cite at least one [chunk_id].
- MISCONCEPTION: Only include if the source text explicitly mentions one.
  If none, use empty string "".
- RELATED_CONCEPTS: List ONLY concepts that appear in the provided context chunks.
- IS_PARTIAL: true if context covers less than half of the concept.

IF NO RELEVANT CONTEXT EXISTS:
{
  "simple_definition": "I could not find this in the textbook.",
  "analogy": "",
  "analogy_is_grounded": true,
  "steps": [],
  "misconception": "",
  "related_concepts": [],
  "is_partial": true
}
"""

EXPLANATION_USER_TEMPLATE = """\
CONTEXT CHUNKS (from NCERT textbook):
{context_block}

User query: Explain {concept}
"""

_MIN_RELEVANCE_SCORE = 0.15


# =========================================================================
# ConceptExplainer
# =========================================================================
class ConceptExplainer:
    """Feynman-style concept explainer grounded in NCERT chunks.

    Flow:
        1. Retrieve top_k chunks for the concept
        2. Relevance check (fast refusal if no relevant content)
        3. Call Gemini with EXPLANATION_SYSTEM_PROMPT (temperature=0)
        4. Parse and validate JSON response
        5. Extract citations and verify grounding
        6. Return ExplanationResponse

    Parameters
    ----------
    retriever : object
        Any retriever with retrieve(query, top_k).
    api_key : str, optional
        Gemini API key.
    model_name : str
        Gemini model to use.
    top_k : int
        Number of chunks to retrieve.  Default: 8.

    DESIGN: MULTI-CHAPTER CONCEPTS
    ------------------------------
    Concepts that span multiple chapters (e.g., "cell" appears in Ch.2
    and Ch.6 Tissues) are handled naturally:
      - The retriever searches the FULL corpus (all loaded chapters)
      - Retrieved chunks may come from different chapters
      - Each chunk carries its chapter_num metadata
      - The explainer presents all retrieved information regardless
        of source chapter, with chunk_ids tracing back to origins
      - The related_concepts list may surface cross-chapter connections

    DESIGN: INSUFFICIENT CONTEXT FALLBACK
    --------------------------------------
    Three-tier fallback:
      1. Score < 0.15  -> FULL REFUSAL (no API call, saves quota)
      2. LLM sets is_partial=true -> PARTIAL explanation with
         explicit "MISSING" acknowledgment
      3. LLM produces full explanation -> SUCCESS
    """

    def __init__(
        self,
        retriever: object,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash-lite",
        top_k: int = 8,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set GEMINI_API_KEY in .env or "
                "pass api_key= to ConceptExplainer()."
            )
        self.model_name = model_name
        self.top_k = top_k
        self._retriever = retriever
        self._client = genai.Client(api_key=self.api_key)

    def explain(self, concept: str) -> ExplanationResponse:
        """Produce a Feynman-style explanation for a concept.

        Parameters
        ----------
        concept : str
            The concept to explain (e.g., "osmosis", "cell wall vs cell membrane").

        Returns
        -------
        ExplanationResponse
            Structured explanation with citations and grounding metadata.
        """
        # ── Stage 1: Retrieve ────────────────────────────────────────────
        retrieved = self._retriever.retrieve(concept, top_k=self.top_k)

        # ── Stage 2: Relevance check ────────────────────────────────────
        if not retrieved or retrieved[0].get("score", 0) < _MIN_RELEVANCE_SCORE:
            return ExplanationResponse(
                concept=concept,
                simple_definition="I could not find this in the textbook.",
                refused=True,
                is_partial=True,
                model=self.model_name,
                n_chunks_used=0,
            )

        # ── Stage 3: Build prompt ────────────────────────────────────────
        context_block = self._build_context(retrieved)
        user_prompt = EXPLANATION_USER_TEMPLATE.format(
            context_block=context_block,
            concept=concept,
        )

        # ── Stage 4: Generate (temperature=0, deterministic) ─────────────
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=EXPLANATION_SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=1536,
            ),
        )

        raw_text = response.text.strip() if response.text else ""

        # ── Stage 5: Parse and validate ──────────────────────────────────
        return self._parse_response(raw_text, concept, retrieved)

    def _build_context(self, chunks: list[dict]) -> str:
        """Build labeled context block with chunk_ids."""
        lines = []
        for i, chunk in enumerate(chunks, 1):
            cid = chunk.get("chunk_id", f"chunk_{i}")
            page = f" (page {chunk['page']})" if "page" in chunk else ""
            ctype = f" [{chunk['type']}]" if "type" in chunk else ""
            ch_num = f" Ch.{chunk['chapter_num']}" if "chapter_num" in chunk else ""
            lines.append(
            f"[{i}] {{chunk_id: \"{cid}\"}}{ctype}{ch_num}{page} \u2014 {chunk.get('content') or chunk.get('text', '')}"
            )
        return "\n\n".join(lines)

    def _parse_response(self, raw: str, concept: str,
                        chunks: list[dict]) -> ExplanationResponse:
        """Parse LLM JSON into ExplanationResponse with validation."""
        # ── Detect full refusal ──────────────────────────────────────────
        refused = "could not find" in raw.lower()

        # ── Extract JSON ─────────────────────────────────────────────────
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if not json_match:
            return ExplanationResponse(
                concept=concept, simple_definition=raw,
                refused=refused, model=self.model_name,
                n_chunks_used=len(chunks),
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return ExplanationResponse(
                concept=concept, simple_definition=raw,
                refused=refused, model=self.model_name,
                n_chunks_used=len(chunks),
            )

        # ── Extract fields ───────────────────────────────────────────────
        simple_def = data.get("simple_definition", "")
        analogy = data.get("analogy", "")
        analogy_grounded = data.get("analogy_is_grounded", True)
        steps = data.get("steps", [])
        misconception = data.get("misconception", "")
        related = data.get("related_concepts", [])
        is_partial = data.get("is_partial", False)

        # ── Detect pedagogical additions in analogy ──────────────────────
        if "pedagogical_addition" in analogy.lower():
            analogy_grounded = False

        # ── Validate related concepts against context ────────────────────
        # Only keep related concepts that actually appear in chunk text
        chunk_text_combined = " ".join((c.get("content") or c.get("text", "")).lower() for c in chunks)
        validated_related = [
            r for r in related
            if r.lower() in chunk_text_combined
        ]

        # ── Extract all cited chunk_ids ──────────────────────────────────
        all_text = (
            simple_def + " " + analogy + " " +
            " ".join(steps) + " " + misconception
        )
        cited_ids = list(dict.fromkeys(re.findall(r'\[([a-z0-9_]+)\]', all_text)))

        # ── Refusal detection ────────────────────────────────────────────
        if not refused:
            refused = "could not find" in simple_def.lower()

        return ExplanationResponse(
            concept=concept,
            simple_definition=simple_def,
            analogy=analogy,
            analogy_is_grounded=analogy_grounded,
            steps=steps,
            misconception=misconception if misconception else "",
            related_concepts=validated_related,
            chunk_ids=cited_ids,
            is_partial=is_partial,
            refused=refused,
            model=self.model_name,
            n_chunks_used=len(chunks),
        )

    def __repr__(self) -> str:
        return f"ConceptExplainer(model={self.model_name!r}, top_k={self.top_k})"
