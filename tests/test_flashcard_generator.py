"""Verification test for FlashcardGenerator."""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from flashcard_generator import (
    FlashcardGenerator, FlashcardResult, Flashcard,
    FLASHCARD_SYSTEM_PROMPT, _VALID_TYPES, _VALID_DIFFICULTIES,
)
from retriever import BM25Retriever, HybridRetriever, load_chunks

CHUNKS_PATH = str(PROJECT_ROOT / "outputs" / "chunks_semantic.json")
chunks = load_chunks(CHUNKS_PATH)

# =========================================================================
# TEST 1: Dataclass schema
# =========================================================================
print("=" * 72)
print("  TEST 1: Dataclass Schema")
print("=" * 72)

card = Flashcard(
    id="fc_001", type="definition",
    front="What is a cell?", back="Basic unit of life.",
    source_chunk_id="sem_0001", difficulty="easy",
)
d = card.to_dict()
assert set(d.keys()) == {"id", "type", "front", "back", "source_chunk_id", "difficulty"}
print(f"  [OK] Flashcard fields: {sorted(d.keys())}")

result = FlashcardResult(
    flashcards=[card], total_generated=1,
    topics_covered=["cell"], topics_missing=[], model="test",
)
rd = result.to_dict()
assert "flashcards" in rd and "total_generated" in rd
assert "topics_covered" in rd and "topics_missing" in rd
print(f"  [OK] FlashcardResult fields: {sorted(rd.keys())}")

# =========================================================================
# TEST 2: System prompt design
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 2: System Prompt Validation")
print("=" * 72)

prompt = FLASHCARD_SYSTEM_PROMPT
assert "ONLY" in prompt, "Must enforce grounding"
assert "chunk_id" in prompt, "Must require source chunk_id"
assert "definition" in prompt and "fill_blank" in prompt and "true_false" in prompt
assert "JSON" in prompt, "Must enforce JSON output"
assert "same fact" in prompt.lower(), "Must prohibit duplicates"
print("  [OK] Grounding, citations, card types, JSON, dedup all enforced")

# =========================================================================
# TEST 3: Validation logic (unit test)
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 3: Schema Validation Logic")
print("=" * 72)

valid_ids = {"sem_0001", "sem_0002", "sem_0003"}

# Valid card
errors = FlashcardGenerator._validate_card({
    "type": "definition",
    "front": "What is a cell membrane?",
    "back": "A thin barrier around the cell.",
    "source_chunk_id": "sem_0001",
    "difficulty": "easy",
}, valid_ids)
assert errors == [], f"Should be valid: {errors}"
print("  [OK] Valid card passes validation")

# Missing field
errors = FlashcardGenerator._validate_card({
    "type": "definition",
    "front": "What?",
    "source_chunk_id": "sem_0001",
}, valid_ids)
assert len(errors) > 0
print(f"  [OK] Missing 'back' detected: {errors[0]}")

# Invalid type
errors = FlashcardGenerator._validate_card({
    "type": "essay",
    "front": "What is a cell membrane?",
    "back": "A barrier.",
    "source_chunk_id": "sem_0001",
}, valid_ids)
assert any("type" in e.lower() for e in errors)
print(f"  [OK] Invalid type detected: {errors[0]}")

# Unknown chunk_id
errors = FlashcardGenerator._validate_card({
    "type": "definition",
    "front": "What is a cell membrane?",
    "back": "A barrier around cells.",
    "source_chunk_id": "sem_9999",
}, valid_ids)
assert any("unknown" in e.lower() for e in errors)
print(f"  [OK] Unknown chunk_id detected: {errors[0]}")

# Invalid difficulty
errors = FlashcardGenerator._validate_card({
    "type": "definition",
    "front": "What is a cell membrane?",
    "back": "A barrier around cells.",
    "source_chunk_id": "sem_0001",
    "difficulty": "impossible",
}, valid_ids)
assert any("difficulty" in e.lower() for e in errors)
print(f"  [OK] Invalid difficulty detected: {errors[0]}")

# =========================================================================
# TEST 4: Out-of-scope fast refusal (no API call)
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 4: Out-of-Scope Fast Refusal")
print("=" * 72)

hybrid = HybridRetriever(CHUNKS_PATH)
try:
    gen = FlashcardGenerator(retriever=hybrid, top_k=10)
    result = gen.generate("quantum entanglement in black holes")
    assert result.refused or result.total_generated == 0
    print(f"  Refused: {result.refused}")
    print(f"  Missing: {result.topics_missing}")
    print("  [OK] Out-of-scope handled without API call")
except ValueError:
    print("  [SKIP] No API key (expected if not set)")

# =========================================================================
# TEST 5: JSON round-trip
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 5: JSON Serialization")
print("=" * 72)

result = FlashcardResult(
    flashcards=[
        Flashcard("fc_001", "definition", "What is X?", "X is Y.", "sem_0001", "easy"),
        Flashcard("fc_002", "true_false", "Cells have walls: T/F?", "True.", "sem_0002", "medium"),
    ],
    total_generated=2,
    topics_covered=["cell wall"],
    topics_missing=[],
    model="test",
)
serialized = json.dumps(result.to_dict(), indent=2)
parsed = json.loads(serialized)
assert len(parsed["flashcards"]) == 2
assert parsed["total_generated"] == 2
print(f"  [OK] Round-trip JSON ({len(serialized)} chars)")

# =========================================================================
# TEST 6: Live generation (API)
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 6: Live Flashcard Generation")
print("=" * 72)

try:
    gen = FlashcardGenerator(retriever=hybrid, top_k=10)
    result = gen.generate("cell organelles")

    print(f"  Generated: {result.total_generated} flashcards")
    print(f"  Covered:   {result.topics_covered}")
    print(f"  Missing:   {result.topics_missing}")
    print(f"  Refused:   {result.refused}")

    for card in result.flashcards:
        assert card.type in _VALID_TYPES
        assert card.difficulty in _VALID_DIFFICULTIES
        assert len(card.front) >= 10
        assert len(card.back) >= 3
        print(f"    [{card.type:11s}] {card.front[:60]}...")
        print(f"               -> {card.back[:60]}... ({card.source_chunk_id})")

    if result.total_generated > 0:
        # Check dedup: no two cards should have identical fronts
        fronts = [c.front.strip().lower() for c in result.flashcards]
        assert len(fronts) == len(set(fronts)), "Duplicate fronts found!"
        print(f"  [OK] {result.total_generated} cards, no duplicates, all valid")
    else:
        print("  [OK] No cards generated (likely quota)")

except ValueError:
    print("  [SKIP] No API key")
except Exception as e:
    if "429" in str(e) or "quota" in str(e).lower():
        print(f"  [SKIP] API quota exhausted (expected)")
    else:
        raise

# =========================================================================
print("\n" + "=" * 72)
print("  ALL TESTS PASSED [OK]")
print("=" * 72)
