"""
src/generator.py -- Grounded answer generation for the NCERT RAG pipeline V2.

Uses Google Gemini to generate answers that are STRICTLY grounded in
retrieved context chunks.  If the context doesn't contain the answer,
the model is instructed to refuse politely.

V2 additions over V1:
  - Pluggable LLM backend via LLMBackend protocol (GeminiBackend, MockLLMBackend)
  - Optional ResponseCache integration (cache_hit in return schema)
  - Structured refusal_reason field in return schema

V1 invariants preserved:
  - SYSTEM_PROMPT constant at module level
  - Exact refusal phrase: "I could not find this in the textbook"
  - temperature=0 hardcoded (not configurable)
  - build_prompt() labels chunks as [1],[2],...[N] with chunk_ids embedded

Usage:
    from generator import GroundedGenerator
    gen = GroundedGenerator(api_key="...")
    result = gen.generate(query, retrieved_chunks)
    print(result["answer"])
    print(result["cache_hit"])  # True if served from cache
"""

from __future__ import annotations

import os
import re
from typing import Optional, Protocol, runtime_checkable


# =========================================================================
# Soft-refusal threshold  (V1 constant — preserved)
# =========================================================================
# If average retrieval score of provided chunks is below this value,
# prepend a confidence warning to the answer.
SOFT_REFUSAL_THRESHOLD = 2.0


# =========================================================================
# System prompt — the grounding contract  (V1 constant — preserved)
# =========================================================================
SYSTEM_PROMPT = """\
You are a helpful study assistant for NCERT Class 9 Science.

Answer ONLY from the context provided below. Context is labeled [1], [2]... [N].
Each label includes a chunk_id for traceability.
If the answer is not in the context, respond EXACTLY with:
"I could not find this in the textbook."

RULES — follow these STRICTLY:
1. Answer the student's question using ONLY the context chunks provided.
2. Cite your sources using the chunk labels [1], [2], etc. for every factual claim.
3. Keep answers clear, concise, and at a Class 9 reading level.
4. If the question is ambiguous, answer the most likely interpretation
   and note the ambiguity.

You MUST NOT:
- Use any knowledge from your training data. Your ONLY knowledge source is
  the context below.
- Add examples, facts, or definitions not present in the provided context.
- Paraphrase the refusal phrase. Use it exactly: "I could not find this in the textbook."
- Make claims without citing a source chunk [1], [2], etc.
"""


# =========================================================================
# User prompt template  (V1 constant — preserved)
# =========================================================================
USER_PROMPT_TEMPLATE = """\
CONTEXT (retrieved from the textbook):
{context_block}

User query: {query}
"""


# =========================================================================
# LLMBackend protocol — pluggable generation backend
# =========================================================================
@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for pluggable LLM backends.

    Any object that implements ``generate(system, user) -> str`` can be
    used as a backend for GroundedGenerator.  This enables:
      - Live API calls (GeminiBackend)
      - Deterministic mock responses (MockLLMBackend)
      - Future backends (OpenAI, Anthropic, local models)
    """

    def generate(self, system: str, user: str) -> str:
        """Generate a text response given system and user prompts.

        Parameters
        ----------
        system : str
            The system instruction (grounding contract).
        user : str
            The user prompt (context + query).

        Returns
        -------
        str
            The raw text response from the LLM.
        """
        ...  # pragma: no cover


# =========================================================================
# GeminiBackend — default live API backend
# =========================================================================
class GeminiBackend:
    """Live Gemini API backend.

    Uses ``google-genai`` SDK with temperature=0 (hardcoded, not
    configurable) for deterministic, reproducible generation.

    Parameters
    ----------
    api_key : str
        Gemini API key.
    model_name : str
        Gemini model ID.  Default: ``gemini-2.5-flash-lite``.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash-lite",
    ) -> None:
        from google import genai
        from google.genai import types as _gentypes
        self._types = _gentypes
        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, system: str, user: str) -> str:
        """Call the Gemini API with temperature=0.

        Returns the raw text response, or empty string if no text
        is returned (e.g., safety filter triggered).
        """
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=user,
            config=self._types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.0,
                max_output_tokens=1024,
            ),
        )
        return response.text.strip() if response.text else ""

    def __repr__(self) -> str:
        return f"GeminiBackend(model={self.model_name!r})"


# =========================================================================
# MockLLMBackend — deterministic mock for testing
# =========================================================================

# Default mock response map — keys are matched against query substrings
_DEFAULT_MOCK_MAP: dict[str, str] = {
    "__in_scope__": (
        "Based on the provided context, the cell is the basic structural "
        "and functional unit of all living organisms. Robert Hooke first "
        "observed cells in 1665 using a cork slice under a microscope [1]. "
        "All living organisms are composed of cells, which carry out "
        "essential life processes [2]."
    ),
    "__out_of_scope__": "I could not find this in the textbook.",
}


class MockLLMBackend:
    """Deterministic mock backend for offline testing.

    Returns pre-configured responses based on keyword matching.
    No API key or network access required.

    Parameters
    ----------
    response_map : dict, optional
        Mapping of query substrings to response strings.  If a query
        contains a key from this map, the corresponding value is returned.
        If no key matches, falls back to a heuristic based on chunk
        overlap.  Special keys:
          - ``"__in_scope__"``: default for in-scope queries
          - ``"__out_of_scope__"``: default for out-of-scope queries

    Example
    -------
    >>> backend = MockLLMBackend({"osmosis": "Osmosis is the movement of water [1]."})
    >>> backend.generate("system prompt", "...User query: What is osmosis?")
    'Osmosis is the movement of water [1].'
    """

    def __init__(self, response_map: Optional[dict[str, str]] = None) -> None:
        self._map = response_map or dict(_DEFAULT_MOCK_MAP)
        self.model_name = "mock-llm-v2"

    def generate(self, system: str, user: str) -> str:
        """Return a mock response based on keyword matching.

        Checks the user prompt against each key in the response map.
        Returns the first match, or the ``__in_scope__`` / ``__out_of_scope__``
        default based on a simple heuristic.
        """
        user_lower = user.lower()

        # Check explicit response map entries (skip special keys)
        for key, response in self._map.items():
            if key.startswith("__"):
                continue
            if key.lower() in user_lower:
                return response

        # Heuristic: if the user prompt contains "CONTEXT" with chunk data,
        # assume it's in-scope; otherwise out-of-scope
        if "chunk_id:" in user_lower and len(user) > 200:
            return self._map.get(
                "__in_scope__",
                "I could not find this in the textbook.",
            )
        else:
            return self._map.get(
                "__out_of_scope__",
                "I could not find this in the textbook.",
            )

    def __repr__(self) -> str:
        return f"MockLLMBackend(keys={list(self._map.keys())})"


# =========================================================================
# GroundedGenerator — the main generation engine
# =========================================================================
class GroundedGenerator:
    """Generate grounded answers using retrieved context chunks.

    Supports pluggable LLM backends and optional response caching.

    Parameters
    ----------
    api_key : str, optional
        Gemini API key.  If None, reads from ``GEMINI_API_KEY`` env var.
        Ignored if a custom ``backend`` is provided.
    model_name : str
        Gemini model to use.  Default: ``gemini-2.5-flash-lite``.
        Ignored if a custom ``backend`` is provided.
    backend : LLMBackend, optional
        Pluggable LLM backend.  If None, a ``GeminiBackend`` is created
        using ``api_key`` and ``model_name``.
    cache : ResponseCache, optional
        If provided, responses are cached to disk.  Cache hits are
        returned without making API calls.

    Example
    -------
    >>> gen = GroundedGenerator(api_key="...")
    >>> result = gen.generate("What is a cell?", chunks)
    >>> result["answer"]
    'A cell is the basic structural and functional unit...'
    >>> result["cache_hit"]
    False

    V1 INVARIANTS PRESERVED:
    - SYSTEM_PROMPT at module level
    - Refusal phrase: "I could not find this in the textbook"
    - temperature=0 hardcoded in GeminiBackend
    - build_prompt() labels chunks [1],[2],...[N] with chunk_ids
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash-lite",
        backend: Optional[LLMBackend] = None,
        cache: Optional[object] = None,
    ) -> None:
        # ── Backend setup ─────────────────────────────────────────────
        if backend is not None:
            self._backend = backend
            self.model_name = getattr(backend, "model_name", "custom")
        else:
            self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
            if not self.api_key:
                raise ValueError(
                    "No API key provided. Set GEMINI_API_KEY in .env or "
                    "pass api_key= to GroundedGenerator()."
                )
            self.model_name = model_name
            self._backend = GeminiBackend(
                api_key=self.api_key, model_name=model_name
            )

        # ── Cache setup (optional) ────────────────────────────────────
        self._cache = cache

    # -----------------------------------------------------------------
    # Prompt construction  (V1 logic preserved)
    # -----------------------------------------------------------------
    def build_prompt(self, query: str, chunks: list[dict]) -> str:
        """Build the full user prompt with labeled context chunks.

        Each chunk is labeled ``[N] {chunk_id: "..."} [type] (page P) — text``
        for the LLM to cite using ``[1]``, ``[2]``, etc.

        Handles both V1 chunk dicts (``text`` key) and V2 retriever
        output (``content`` key) transparently.

        Parameters
        ----------
        query : str
            The student's question.
        chunks : list[dict]
            Retrieved chunks.  Must have ``chunk_id`` and either
            ``text`` or ``content`` key.

        Returns
        -------
        str
            The formatted prompt string.
        """
        context_lines = []
        for i, chunk in enumerate(chunks, 1):
            cid = chunk.get("chunk_id", f"chunk_{i}")
            page_info = f" (page {chunk['page']})" if "page" in chunk else ""
            ctype = f" [{chunk['type']}]" if "type" in chunk else ""

            # V2 retriever returns "content"; V1 chunks have "text"
            text = chunk.get("content") or chunk.get("text", "")

            context_lines.append(
                f'[{i}] {{chunk_id: "{cid}"}}{ctype}{page_info} — {text}'
            )

        context_block = "\n\n".join(context_lines)

        return USER_PROMPT_TEMPLATE.format(
            context_block=context_block,
            query=query,
        )

    # -----------------------------------------------------------------
    # Refusal detection  (V1 logic preserved)
    # -----------------------------------------------------------------
    def _is_refusal(self, answer: str) -> bool:
        """Detect if the answer is a refusal.

        Uses exact substring match against the canonical refusal phrase.
        This is intentionally strict — paraphrased refusals (e.g.,
        "I couldn't find") are NOT detected, forcing the LLM to use
        the exact phrase.

        Parameters
        ----------
        answer : str
            The generated answer text.

        Returns
        -------
        bool
            True if the answer contains the exact refusal phrase.
        """
        return "I could not find this in the textbook" in answer

    # -----------------------------------------------------------------
    # Source extraction
    # -----------------------------------------------------------------
    @staticmethod
    def _extract_sources(answer: str, chunks: list[dict]) -> list[str]:
        """Extract cited chunk_ids from [N] labels in the answer.

        Maps integer labels back to the corresponding chunk_id from
        the input chunk list.

        Parameters
        ----------
        answer : str
            The generated answer text containing [1], [2], etc.
        chunks : list[dict]
            The chunks passed to build_prompt (same order).

        Returns
        -------
        list[str]
            Sorted list of unique chunk_ids that were cited.
        """
        cited_indices = set(int(m) for m in re.findall(r"\[(\d+)\]", answer))
        sources = []
        for idx in sorted(cited_indices):
            if 1 <= idx <= len(chunks):
                sources.append(chunks[idx - 1]["chunk_id"])
        return sources

    # -----------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------
    def generate(self, query: str, chunks: list[dict]) -> dict:
        """Generate a grounded answer from query + retrieved chunks.

        Flow:
          1. Check cache (if configured) → return on hit
          2. Build prompt from query + chunks
          3. Call LLM backend
          4. Detect refusal, extract sources, apply soft refusal
          5. Store in cache (if configured)
          6. Return structured response

        Parameters
        ----------
        query : str
            The student's question.
        chunks : list[dict]
            Retrieved chunks (typically top_k from retriever).

        Returns
        -------
        dict
            {
                "answer":         str,            # generated text
                "sources":        list[str],       # chunk_ids cited
                "refused":        bool,            # hard refusal
                "cache_hit":      bool,            # True if from cache
                "refusal_reason": str | None,      # "out_of_scope" | "low_confidence" | None
                "model":          str,             # model name used
            }
        """
        chunk_ids = [c.get("chunk_id", "") for c in chunks]

        # ── Step 1: Check cache ──────────────────────────────────────
        if self._cache is not None:
            cached = self._cache.get(query, chunk_ids)
            if cached is not None:
                # Ensure V2 fields are present in cached response
                cached.setdefault("cache_hit", True)
                cached.setdefault("refusal_reason", None)
                cached["cache_hit"] = True
                return cached

        # ── Step 2: Build prompt ─────────────────────────────────────
        user_prompt = self.build_prompt(query, chunks)

        # ── Step 3: Call LLM backend ─────────────────────────────────
        answer_text = self._backend.generate(SYSTEM_PROMPT, user_prompt)

        # ── Step 4: Detect refusal ───────────────────────────────────
        refused = self._is_refusal(answer_text)
        refusal_reason: Optional[str] = None

        if refused:
            refusal_reason = "out_of_scope"

        # ── Step 4b: Soft refusal (low-confidence retrieval) ─────────
        avg_score = (
            sum(c.get("score", 0) for c in chunks) / len(chunks)
            if chunks else 0
        )
        soft_refused = False
        if not refused and avg_score < SOFT_REFUSAL_THRESHOLD and chunks:
            answer_text = (
                "The context I found may not fully answer this question. "
                "Here is what I found: " + answer_text +
                " Please verify against your textbook."
            )
            soft_refused = True
            refusal_reason = "low_confidence"

        # ── Step 5: Extract cited sources ────────────────────────────
        sources = self._extract_sources(answer_text, chunks)

        # ── Step 6: Build response ───────────────────────────────────
        response = {
            "answer":         answer_text,
            "sources":        sources,
            "refused":        refused,
            "cache_hit":      False,
            "refusal_reason": refusal_reason,
            "model":          self.model_name,
        }

        # ── Step 7: Store in cache ───────────────────────────────────
        if self._cache is not None:
            self._cache.set(query, chunk_ids, response)

        return response

    def __repr__(self) -> str:
        cache_status = f", cache={self._cache}" if self._cache else ""
        return f"GroundedGenerator(model={self.model_name!r}{cache_status})"
