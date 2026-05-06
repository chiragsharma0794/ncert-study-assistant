"""Verification test for ResponseCache, CachedGenerator, and MockLLMGenerator."""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cache import ResponseCache, CachedGenerator, MockLLMGenerator
from retriever import BM25Retriever, load_chunks

CHUNKS_PATH = PROJECT_ROOT / "outputs" / "chunks_semantic.json"
TEST_CACHE = PROJECT_ROOT / "outputs" / "test_response_cache.json"

# Clean up any previous test cache
if TEST_CACHE.exists():
    TEST_CACHE.unlink()

chunks = load_chunks(CHUNKS_PATH)
bm25 = BM25Retriever(str(CHUNKS_PATH))

# =========================================================================
# TEST 1: Cache basic operations
# =========================================================================
print("=" * 72)
print("  TEST 1: Cache Set / Get")
print("=" * 72)

cache = ResponseCache(TEST_CACHE)
query = "What is a cell?"
retrieved = bm25.retrieve(query, top_k=3)
chunk_ids = [c["chunk_id"] for c in retrieved]

fake_response = {
    "answer": "A cell is the basic unit of life.",
    "sources": [chunk_ids[0]],
    "refused": False,
    "model": "test-model",
}

# Miss before set
result = cache.get(query, chunk_ids)
assert result is None, "Should be a miss"
print("  [OK] Cache miss before set")

# Set
cache.set(query, chunk_ids, fake_response)
print("  [OK] Response cached")

# Hit after set
result = cache.get(query, chunk_ids)
assert result is not None, "Should be a hit"
assert result["answer"] == fake_response["answer"]
assert result["sources"] == fake_response["sources"]
assert result["refused"] == fake_response["refused"]
assert result["model"] == fake_response["model"]
print("  [OK] Cache hit -- response matches exactly")

# =========================================================================
# TEST 2: Deterministic keys
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 2: Key Determinism")
print("=" * 72)

# Same query, same chunk_ids in different order -> same key
key1 = cache._make_key("What is a cell?", ["sem_0001", "sem_0003", "sem_0005"])
key2 = cache._make_key("What is a cell?", ["sem_0005", "sem_0001", "sem_0003"])
assert key1 == key2, "Sorted chunk_ids should produce same key"
print("  [OK] Order-independent: different chunk order -> same key")

# Whitespace normalization
key3 = cache._make_key("  What is a cell?  ", ["sem_0001", "sem_0003", "sem_0005"])
assert key1 == key3, "Whitespace should be normalized"
print("  [OK] Whitespace normalized")

# Case normalization
key4 = cache._make_key("WHAT IS A CELL?", ["sem_0001", "sem_0003", "sem_0005"])
assert key1 == key4, "Case should be normalized"
print("  [OK] Case normalized")

# Different query -> different key
key5 = cache._make_key("What is osmosis?", ["sem_0001", "sem_0003", "sem_0005"])
assert key1 != key5, "Different query should produce different key"
print("  [OK] Different query -> different key")

# =========================================================================
# TEST 3: Persistence
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 3: Disk Persistence")
print("=" * 72)

assert TEST_CACHE.exists(), "Cache file should exist"
size = TEST_CACHE.stat().st_size
print(f"  [OK] Cache file exists ({size} bytes)")

# Load fresh cache from disk
cache2 = ResponseCache(TEST_CACHE)
result2 = cache2.get(query, chunk_ids)
assert result2 is not None, "Should load from disk"
assert result2["answer"] == fake_response["answer"]
print("  [OK] Fresh load from disk -> same response")

# Verify JSON is inspectable
with open(TEST_CACHE) as f:
    raw = json.load(f)
assert isinstance(raw, dict)
print(f"  [OK] JSON inspectable: {len(raw)} entries")

# =========================================================================
# TEST 4: Invalidation
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 4: Invalidation")
print("=" * 72)

removed = cache.invalidate("What is a cell?", chunk_ids)
assert removed == True, f"Should return True for existing entry"
result = cache.get(query, chunk_ids)
assert result is None, "Should be gone after invalidation"
print(f"  [OK] invalidate() removed entry")

# =========================================================================
# TEST 5: Stats
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 5: Stats")
print("=" * 72)

stats = cache.stats()
print(f"  Entries:  {stats['total_entries']}")
print(f"  Hits:     {stats['hit_count']}")
print(f"  Misses:   {stats['miss_count']}")
print(f"  Hit rate: {stats['hit_rate']:.2%}")
print(f"  Size:     {stats['cache_file_size_bytes']} bytes")
print("  [OK] Stats computed")

# =========================================================================
# TEST 6: MockLLMGenerator
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 6: MockLLMGenerator")
print("=" * 72)

mock = MockLLMGenerator()
retrieved = bm25.retrieve("What is a cell membrane?", top_k=5)
result = mock.generate("What is a cell membrane?", retrieved)

assert "answer" in result
assert "sources" in result
assert "refused" in result
assert "model" in result
assert result["model"] == "mock-llm-v1"
assert not result["refused"], "In-scope query should not be refused"
print(f"  [OK] In-scope response: refused={result['refused']}, sources={result['sources']}")

# Out-of-scope
oos_result = mock.generate("Who invented the telephone?", retrieved)
assert oos_result["refused"], "Out-of-scope should be refused"
assert "could not find" in oos_result["answer"].lower()
print(f"  [OK] Out-of-scope response: refused={oos_result['refused']}")

# Schema matches GroundedGenerator exactly
required_keys = {"answer", "sources", "refused", "model"}
assert set(result.keys()) == required_keys
print(f"  [OK] Schema matches: {sorted(required_keys)}")

# =========================================================================
# TEST 7: CachedGenerator with MockLLM
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 7: CachedGenerator Integration")
print("=" * 72)

cache.clear_all()
cache = ResponseCache(TEST_CACHE)
cached_gen = CachedGenerator(mock, cache, offline=False)

retrieved = bm25.retrieve("What is a cell?", top_k=3)

# First call: cache miss -> calls mock
r1 = cached_gen.generate("What is a cell?", retrieved)
s1 = cache.stats()
assert s1["miss_count"] == 1 and s1["hit_count"] == 0
print(f"  [OK] First call: miss (hits={s1['hit_count']}, misses={s1['miss_count']})")

# Second call: cache hit
r2 = cached_gen.generate("What is a cell?", retrieved)
s2 = cache.stats()
assert s2["hit_count"] == 1
assert r1["answer"] == r2["answer"]
print(f"  [OK] Second call: hit (hits={s2['hit_count']}, misses={s2['miss_count']})")

# =========================================================================
# TEST 8: Offline mode
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 8: Offline Mode")
print("=" * 72)

offline_gen = CachedGenerator(mock, cache, offline=True)

# Cached query works in offline mode
r3 = offline_gen.generate("What is a cell?", retrieved)
assert r3["answer"] == r1["answer"]
print("  [OK] Cached query works in offline mode")

# Uncached query raises error
try:
    new_retrieved = bm25.retrieve("What is osmosis?", top_k=3)
    offline_gen.generate("What is osmosis?", new_retrieved)
    assert False, "Should have raised RuntimeError"
except RuntimeError as e:
    assert "offline mode" in str(e).lower()
    print(f"  [OK] Uncached query raises RuntimeError in offline mode")

# =========================================================================
# Cleanup
# =========================================================================
cache.clear_all()
if TEST_CACHE.exists():
    TEST_CACHE.unlink()

print("\n" + "=" * 72)
print("  ALL TESTS PASSED [OK]")
print("=" * 72)
