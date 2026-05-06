"""Verification test for CorpusManager."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from corpus_manager import CorpusManager
from retriever import BM25Retriever

mgr = CorpusManager(PROJECT_ROOT / "data", PROJECT_ROOT / "outputs")

# =========================================================================
# TEST 1: PDF discovery
# =========================================================================
print("=" * 72)
print("  TEST 1: PDF Discovery")
print("=" * 72)
available = mgr.discover_pdfs()
print(f"  [OK] Found {len(available)} PDFs: {available}")

# =========================================================================
# TEST 2: Process chapter 2
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 2: Process Chapter 2")
print("=" * 72)
chunks = mgr.process_chapter(2)
print(f"  [OK] Produced {len(chunks)} chunks")
print(f"  First ID: {chunks[0]['chunk_id']}")
print(f"  Last ID:  {chunks[-1]['chunk_id']}")
print(f"  Schema:   {sorted(chunks[0].keys())}")

# =========================================================================
# TEST 3: V2 chunk_id format
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 3: Chunk ID Format")
print("=" * 72)
import re
pattern = re.compile(r"^ch\d{2}_p\d{2}_s\d{3}$")
for ch in chunks:
    assert pattern.match(ch["chunk_id"]), f"Bad ID: {ch['chunk_id']}"
print(f"  [OK] All {len(chunks)} IDs match ch{{NN}}_p{{PP}}_s{{SSS}}")

# No duplicates
ids = [ch["chunk_id"] for ch in chunks]
assert len(ids) == len(set(ids)), "Duplicate chunk IDs found!"
print(f"  [OK] Zero collisions in {len(ids)} IDs")

# =========================================================================
# TEST 4: Schema has chapter metadata
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 4: Chapter Metadata")
print("=" * 72)
required = {"chunk_id", "source_id", "chapter_num", "chapter_title", "page", "type", "text", "token_count"}
for ch in chunks:
    missing = required - set(ch.keys())
    assert not missing, f"Missing keys: {missing}"
assert all(ch["chapter_num"] == 2 for ch in chunks)
assert all(ch["chapter_title"] == "Cell: The Building Block of Life" for ch in chunks)
print(f"  [OK] All chunks have chapter_num=2, chapter_title set")

# =========================================================================
# TEST 5: Shard file saved
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 5: Shard Persistence")
print("=" * 72)
shard = PROJECT_ROOT / "outputs" / "chapters" / "ch02_chunks.json"
assert shard.exists(), "Shard file not found"
size_kb = shard.stat().st_size / 1024
print(f"  [OK] ch02_chunks.json saved ({size_kb:.1f} KB)")

manifest_path = PROJECT_ROOT / "outputs" / "chapters" / "manifest.json"
assert manifest_path.exists(), "Manifest not found"
print(f"  [OK] manifest.json exists")

# =========================================================================
# TEST 6: Lazy loading
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 6: Lazy Loading")
print("=" * 72)

mgr2 = CorpusManager(PROJECT_ROOT / "data", PROJECT_ROOT / "outputs")
assert len(mgr2.get_loaded_chapters()) == 0, "Should start empty"
print(f"  [OK] Fresh manager: {mgr2.get_loaded_chapters()} loaded")

loaded = mgr2.load_chunks(chapters=[2])
assert len(loaded) == len(chunks), "Loaded count mismatch"
assert mgr2.get_loaded_chapters() == [2], "Should have ch2 loaded"
print(f"  [OK] load_chunks([2]) -> {len(loaded)} chunks, cached: {mgr2.get_loaded_chapters()}")

mgr2.unload_chapter(2)
assert len(mgr2.get_loaded_chapters()) == 0, "Should be empty after unload"
print(f"  [OK] unload_chapter(2) -> cached: {mgr2.get_loaded_chapters()}")

# =========================================================================
# TEST 7: Backward compatibility with BM25Retriever
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 7: BM25 Compatibility")
print("=" * 72)

# V2 BM25Retriever takes a file path -- write chunks to temp file
tmp_chunks = PROJECT_ROOT / "outputs" / "_test_corpus_bm25.json"
with open(tmp_chunks, "w", encoding="utf-8") as f:
    import json as _json
    _json.dump(chunks, f)
try:
    retriever = BM25Retriever(str(tmp_chunks))
    results = retriever.retrieve("What is a cell membrane?", top_k=3)
    assert len(results) == 3
    assert "score" in results[0]
    assert "content" in results[0]  # V2 returns 'content' not 'text'
    print(f"  [OK] BM25Retriever works with V2 chunks")
    print(f"       Top result: {results[0]['chunk_id']} (score={results[0]['score']:.3f})")
finally:
    if tmp_chunks.exists():
        tmp_chunks.unlink()

# =========================================================================
# TEST 8: Build unified chunks_all.json
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 8: Unified Chunk File")
print("=" * 72)

out_path = mgr.build_unified_chunks()
assert out_path.exists()
import json
with open(out_path) as f:
    all_chunks = json.load(f)
print(f"  [OK] chunks_all.json: {len(all_chunks)} chunks ({out_path.stat().st_size / 1024:.1f} KB)")

# =========================================================================
# TEST 9: Type distribution
# =========================================================================
print("\n" + "=" * 72)
print("  TEST 9: Type Distribution")
print("=" * 72)
from collections import Counter
types = Counter(ch["type"] for ch in chunks)
for t, count in types.most_common():
    print(f"    {t:12s} : {count}")
print(f"  [OK] {len(types)} types found")

# =========================================================================
print("\n" + "=" * 72)
print("  ALL TESTS PASSED [OK]")
print("=" * 72)
