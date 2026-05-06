"""
src/flashcard_generator.py -- Grounded flashcard generation for the NCERT RAG pipeline.

Generates Anki-compatible flashcards from retrieved NCERT chunks.
Every flashcard is traceable to a specific source chunk_id.

Usage:
    from flashcard_generator import FlashcardGenerator
    gen = FlashcardGenerator(retriever, api_key="...")
    result = gen.generate("cell membrane")
    for card in result.flashcards:
        print(f"Q: {card['front']}")
        print(f"A: {card['back']}")
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
# Response schema
# =========================================================================
@dataclass
class Flashcard:
    """Single flashcard with source tracing."""
    id: str                    # e.g., "fc_001"
    type: str                  # "definition" | "fill_blank" | "true_false"
    front: str                 # question / prompt
    back: str                  # answer
    source_chunk_id: str       # chunk this fact came from
    difficulty: str = "medium" # "easy" | "medium" | "hard"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FlashcardResult:
    """Complete flashcard generation result.

    Fields
    ------
    flashcards : list[Flashcard]
        Generated flashcards, each traceable to a source chunk.
    total_generated : int
        Number of flashcards produced.
    topics_covered : list[str]
        Topics/concepts found in the context and covered by cards.
    topics_missing : list[str]
        Aspects the model identified as not present in context.
    refused : bool
        True if no relevant context was found at all.
    model : str
        Model name used for generation.
    """
    flashcards: list[Flashcard] = field(default_factory=list)
    total_generated: int = 0
    topics_covered: list[str] = field(default_factory=list)
    topics_missing: list[str] = field(default_factory=list)
    refused: bool = False
    model: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# =========================================================================
# Flashcard system prompt
# =========================================================================
FLASHCARD_SYSTEM_PROMPT = """\
You are a study assistant that creates flashcards from NCERT textbook content.

Create flashcards ONLY from the context chunks provided below.
Context is labeled with chunk_ids in the format [N] {chunk_id: "..."}.
If the context does not contain relevant information, respond with the empty JSON below.

You MUST NOT:
- Use any knowledge from your training data. Your ONLY knowledge source is
  the context below.
- Add facts, definitions, or examples not present in the provided context.
- Generate two flashcards that test the same fact.
- Include information from outside the retrieved chunks.

FLASHCARD TYPES -- vary across these three types:
- "definition": Front asks "What is X?", back gives the definition from the text.
- "fill_blank": Front has a sentence with a key term replaced by "___",
  back gives the missing term.
- "true_false": Front states a claim (may be true or false), back states
  "True" or "False" with a brief explanation.

DIFFICULTY LEVELS:
- "easy": Direct recall of a single fact or term.
- "medium": Requires understanding a concept or relationship.
- "hard": Requires comparing, contrasting, or applying knowledge.

OUTPUT FORMAT -- respond ONLY with valid JSON. No preamble, no explanation,
no markdown fences.
{
  "flashcards": [
    {
      "id": "fc_001",
      "type": "definition",
      "front": "What is the cell membrane?",
      "back": "The cell membrane is a thin, flexible barrier...",
      "source_chunk_id": "sem_0023",
      "difficulty": "easy"
    }
  ],
  "topics_covered": ["cell membrane", "plasma membrane"],
  "topics_missing": ["aspects not found in context"]
}

IF NO RELEVANT CONTEXT EXISTS:
{
  "flashcards": [],
  "topics_covered": [],
  "topics_missing": ["topic not found in provided context"]
}

QUALITY RULES:
- Front should be a clear, unambiguous question or prompt.
- Back should be concise (1-3 sentences max).
- Generate 5 to 10 flashcards when sufficient context exists.
- Mix all three card types and difficulty levels.
- Do NOT repeat the same information across cards.
"""

FLASHCARD_USER_TEMPLATE = """\
CONTEXT CHUNKS (from NCERT textbook):
{context_block}

User query: Generate flashcards about {topic}
"""

# ── Validation constants ─────────────────────────────────────────────────
_VALID_TYPES = {"definition", "fill_blank", "true_false"}
_VALID_DIFFICULTIES = {"easy", "medium", "hard"}
_MIN_RELEVANCE_SCORE = 0.15


# =========================================================================
# FlashcardGenerator
# =========================================================================
class FlashcardGenerator:
    """Generates grounded, Anki-compatible flashcards from NCERT chunks.

    Flow:
        1. Retrieve top_k=10 chunks for the topic
        2. Check relevance (fast refusal if no relevant content)
        3. Call Gemini with FLASHCARD_SYSTEM_PROMPT (temperature=0)
        4. Parse and validate JSON response
        5. Deduplicate cards testing the same fact
        6. Return FlashcardResult with source tracing

    Parameters
    ----------
    retriever : object
        Any retriever with retrieve(query, top_k).
    api_key : str, optional
        Gemini API key.
    model_name : str
        Gemini model to use.
    top_k : int
        Number of chunks to retrieve.  Default: 10 (more than Q&A/summary
        because flashcards need broad coverage of a topic).
    """

    def __init__(
        self,
        retriever: object,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash-lite",
        top_k: int = 10,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set GEMINI_API_KEY in .env or "
                "pass api_key= to FlashcardGenerator()."
            )
        self.model_name = model_name
        self.top_k = top_k
        self._retriever = retriever
        self._client = genai.Client(api_key=self.api_key)

    def generate(self, topic: str) -> FlashcardResult:
        """Generate flashcards for a topic.

        Parameters
        ----------
        topic : str
            Topic string (e.g., "cell membrane") or chapter reference.

        Returns
        -------
        FlashcardResult
            Validated flashcards with source tracing and coverage metadata.
        """
        # ── Stage 1: Retrieve ────────────────────────────────────────────
        retrieved = self._retriever.retrieve(topic, top_k=self.top_k)

        # ── Stage 2: Relevance check ────────────────────────────────────
        if not retrieved or retrieved[0].get("score", 0) < _MIN_RELEVANCE_SCORE:
            return FlashcardResult(
                refused=True,
                topics_missing=[f"'{topic}' not found in context"],
                model=self.model_name,
            )

        # ── Stage 3: Build prompt ────────────────────────────────────────
        context_block = self._build_context(retrieved)
        user_prompt = FLASHCARD_USER_TEMPLATE.format(
            context_block=context_block,
            topic=topic,
        )

        # ── Stage 4: Generate (temperature=0) ────────────────────────────
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=FLASHCARD_SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=2048,
            ),
        )

        raw_text = response.text.strip() if response.text else ""

        # ── Stage 5: Parse, validate, deduplicate ────────────────────────
        return self._parse_and_validate(raw_text, topic, retrieved)

    def _build_context(self, chunks: list[dict]) -> str:
        """Build labeled context block."""
        lines = []
        for i, chunk in enumerate(chunks, 1):
            cid = chunk["chunk_id"]
            page = f" (page {chunk['page']})" if "page" in chunk else ""
            ctype = f" [{chunk['type']}]" if "type" in chunk else ""
            lines.append(
                f"[{i}] {{chunk_id: \"{cid}\"}}{ctype}{page} \u2014 {chunk.get('content') or chunk.get('text', '')}"
            )
        return "\n\n".join(lines)

    def _parse_and_validate(self, raw: str, topic: str,
                            chunks: list[dict]) -> FlashcardResult:
        """Parse LLM JSON, validate each flashcard, deduplicate."""
        # ── Extract JSON ─────────────────────────────────────────────────
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if not json_match:
            return FlashcardResult(
                refused=True,
                topics_missing=["Failed to parse model response"],
                model=self.model_name,
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return FlashcardResult(
                refused=True,
                topics_missing=["Invalid JSON in model response"],
                model=self.model_name,
            )

        # ── Validate each flashcard ──────────────────────────────────────
        valid_chunk_ids = {c["chunk_id"] for c in chunks}
        raw_cards = data.get("flashcards", [])
        validated = []
        seen_fronts = set()  # for dedup

        for i, card in enumerate(raw_cards):
            errors = self._validate_card(card, valid_chunk_ids)
            if errors:
                continue  # skip invalid cards silently

            # ── Deduplication ────────────────────────────────────────────
            front_norm = card["front"].strip().lower()
            if front_norm in seen_fronts:
                continue
            seen_fronts.add(front_norm)

            validated.append(Flashcard(
                id=f"fc_{len(validated) + 1:03d}",
                type=card["type"],
                front=card["front"].strip(),
                back=card["back"].strip(),
                source_chunk_id=card["source_chunk_id"],
                difficulty=card.get("difficulty", "medium"),
            ))

        # ── Detect refusal ───────────────────────────────────────────────
        refused = len(validated) == 0 and "could not find" in raw.lower()

        return FlashcardResult(
            flashcards=validated,
            total_generated=len(validated),
            topics_covered=data.get("topics_covered", []),
            topics_missing=data.get("topics_missing", []),
            refused=refused,
            model=self.model_name,
        )

    @staticmethod
    def _validate_card(card: dict, valid_chunk_ids: set[str]) -> list[str]:
        """Validate a single flashcard dict against the schema.

        Returns
        -------
        list[str]
            List of validation errors.  Empty = valid.
        """
        errors = []

        # Required fields
        for key in ("type", "front", "back", "source_chunk_id"):
            if key not in card or not isinstance(card[key], str) or not card[key].strip():
                errors.append(f"Missing or empty field: {key}")

        if errors:
            return errors

        # Type validation
        if card["type"] not in _VALID_TYPES:
            errors.append(f"Invalid type: {card['type']}. Must be one of {_VALID_TYPES}")

        # Difficulty validation
        diff = card.get("difficulty", "medium")
        if diff not in _VALID_DIFFICULTIES:
            errors.append(f"Invalid difficulty: {diff}. Must be one of {_VALID_DIFFICULTIES}")

        # Source chunk validation
        if card["source_chunk_id"] not in valid_chunk_ids:
            errors.append(f"Unknown source_chunk_id: {card['source_chunk_id']}")

        # Content quality checks
        if len(card["front"]) < 10:
            errors.append("Front is too short (< 10 chars)")
        if len(card["back"]) < 3:
            errors.append("Back is too short (< 3 chars)")

        return errors

    def __repr__(self) -> str:
        return f"FlashcardGenerator(model={self.model_name!r}, top_k={self.top_k})"
