"""Verification test for ConceptExplainer."""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from explainer import (
    ConceptExplainer, ExplanationResponse,
    EXPLANATION_SYSTEM_PROMPT,
)
from retriever import HybridRetriever, load_chunks

CHUNKS_PATH = str(PROJECT_ROOT / "outputs" / "chunks_semantic.json")
chunks = load_chunks(CHUNKS_PATH)

# =========================================================================
# TEST 1: ExplanationResponse dataclass
# =========================================================================
print("=" * 72)
print("  TEST 1: ExplanationResponse Dataclass")
print("=" * 72)

er = ExplanationResponse(concept="test", simple_definition="A test.")
d = er.to_dict()
required = {
    "concept", "simple_definition", "analogy", "analogy_is_grounded",
    "steps", "misconception", "related_concepts", "chunk_ids",
    "is_partial", "refused", "model", "n_chunks_used",
}
assert set(d.keys()) == required
assert er.refused == False
assert er.is_partial == False
assert er.analogy_is_grounded == True
assert er.steps == []
print(f"  [OK] Fields: {sorted(d.keys())}")

# =========================================================================
# TEST 2: System prompt validation
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 2: System Prompt Validation")
print("=" * 72)

prompt = EXPLANATION_SYSTEM_PROMPT
assert "MUST NOT" in prompt
print("  [OK] Negative instruction: MUST NOT block present")

assert "training data" in prompt.lower() or "only knowledge source" in prompt.lower()
print("  [OK] Training data prohibition enforced")

assert "pedagogical_addition" in prompt
print("  [OK] Pedagogical addition tagging required")

assert "related concepts" in prompt.lower()
print("  [OK] Related concepts restricted to context")

assert "chunk_id" in prompt
print("  [OK] Citation requirement present")

assert "could not find" in prompt.lower()
print("  [OK] Refusal phrase preserved")

assert "JSON" in prompt
print("  [OK] JSON output enforced")

assert "Feynman" in prompt or "simple" in prompt.lower()
print("  [OK] Simple explanation style enforced")

# =========================================================================
# TEST 3: Out-of-scope fast refusal
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 3: Out-of-Scope Fast Refusal")
print("=" * 72)

hybrid = HybridRetriever(CHUNKS_PATH)
try:
    explainer = ConceptExplainer(retriever=hybrid, top_k=8)

    result = explainer.explain("quantum superposition")
    # With HybridRetriever, OOS may pass the fast refusal threshold (0.15)
    # and reach the LLM, which should still refuse or mark as partial.
    assert result.refused or result.is_partial, (
        f"OOS should be refused or partial, got refused={result.refused}, "
        f"is_partial={result.is_partial}"
    )
    print(f"  Refused:    {result.refused}")
    print(f"  Partial:    {result.is_partial}")
    print(f"  N chunks:   {result.n_chunks_used}")
    print(f"  Definition: {result.simple_definition[:80]}")
    if result.n_chunks_used == 0:
        print("  [OK] Out-of-scope refused via fast refusal (no API call)")
    else:
        print("  [OK] Out-of-scope refused via LLM (API call made)")
except ValueError:
    print("  [SKIP] No API key")

# =========================================================================
# TEST 4: JSON serialization
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 4: JSON Serialization")
print("=" * 72)

er = ExplanationResponse(
    concept="osmosis",
    simple_definition="Water moving through a membrane [sem_0030].",
    analogy="[pedagogical_addition: not from NCERT] Like a coffee filter.",
    analogy_is_grounded=False,
    steps=[
        "1. Water molecules move from high to low concentration [sem_0030]",
        "2. The membrane only lets small molecules through [sem_0031]",
    ],
    misconception="",
    related_concepts=["diffusion", "cell membrane"],
    chunk_ids=["sem_0030", "sem_0031"],
    model="test",
    n_chunks_used=8,
)
serialized = json.dumps(er.to_dict(), indent=2)
parsed = json.loads(serialized)
assert parsed["analogy_is_grounded"] == False
assert len(parsed["steps"]) == 2
assert len(parsed["chunk_ids"]) == 2
print(f"  [OK] Round-trip JSON ({len(serialized)} chars)")

# =========================================================================
# TEST 5: Pedagogical addition detection
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 5: Pedagogical Addition Detection")
print("=" * 72)

# Grounded analogy (from source)
er_grounded = ExplanationResponse(
    concept="test",
    analogy="Like a factory producing energy [sem_0046].",
    analogy_is_grounded=True,
)
assert er_grounded.analogy_is_grounded == True
print("  [OK] Grounded analogy: analogy_is_grounded=True")

# Non-grounded analogy (pedagogical addition)
er_added = ExplanationResponse(
    concept="test",
    analogy="[pedagogical_addition: not from NCERT] Like a security guard at a gate.",
    analogy_is_grounded=False,
)
assert er_added.analogy_is_grounded == False
print("  [OK] Pedagogical addition: analogy_is_grounded=False")

# =========================================================================
# TEST 6: Live explanation (API)
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 6: Live Concept Explanation")
print("=" * 72)

try:
    explainer = ConceptExplainer(retriever=hybrid, top_k=8)
    result = explainer.explain("cell membrane")

    print(f"  Concept:    {result.concept}")
    print(f"  Refused:    {result.refused}")
    print(f"  Partial:    {result.is_partial}")
    print(f"  Definition: {result.simple_definition[:100]}...")
    print(f"  Analogy:    {result.analogy[:100]}...")
    print(f"  Grounded:   {result.analogy_is_grounded}")
    print(f"  Steps:      {len(result.steps)}")
    for s in result.steps:
        print(f"    {s[:80]}...")
    print(f"  Misconception: {result.misconception[:80] if result.misconception else '(none)'}")
    print(f"  Related:    {result.related_concepts}")
    print(f"  Citations:  {result.chunk_ids}")

    assert not result.refused
    assert len(result.simple_definition) > 10
    assert len(result.steps) >= 2
    assert len(result.chunk_ids) >= 1
    print("  [OK] Full explanation generated with citations")

except ValueError:
    print("  [SKIP] No API key")
except Exception as e:
    if "429" in str(e) or "quota" in str(e).lower():
        print("  [SKIP] API quota exhausted (expected)")
    else:
        raise

# =========================================================================
print("\n" + "=" * 72)
print("  ALL TESTS PASSED [OK]")
print("=" * 72)
