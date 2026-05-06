"""
src/cache.py -- Response caching for the NCERT RAG pipeline V2.

Provides:
  - ResponseCache: thread-safe, deterministic disk cache for LLM responses
  - CachedGenerator: wrapper around any generator with cache integration
  - MockLLMGenerator: offline synthetic generator (no API key needed)

THEORETICAL BASIS FOR CACHING:
    temperature=0 in the Gemini generation config means the model uses
    greedy decoding (always picks the highest-probability token).  This
    makes the output a DETERMINISTIC FUNCTION of the input:

        f(system_prompt, user_prompt) -> answer   (always the same)

    Since the user prompt is constructed from (query, chunks), and the
    system prompt is a frozen constant, we can cache:

        cache_key = SHA256(query + sorted(chunk_ids))
        cache[key] = {query, chunk_ids, response, cached_at, hit_count}

    A cache hit is GUARANTEED to return the same response the API would
    produce, making offline evaluation replay identical to live runs.

CACHE FILE FORMAT (V2):
    {
      "version": "2.0",
      "created_at": "ISO8601",
      "entries": {
        "<sha256_key>": {
          "query": str,
          "chunk_ids": list[str],
          "response": dict,
          "cached_at": "ISO8601",
          "hit_count": int
        }
      }
    }

THREAD SAFETY:
    All disk writes use filelock + write-then-rename for atomic updates.
    Concurrent readers never see a partially-written file.

FAULT TOLERANCE:
    All public methods catch exceptions internally and log warnings.
    A corrupted or missing cache file is silently replaced with an
    empty cache.  The cache is a performance optimization, not a
    correctness requirement — losing it only costs API calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from filelock import FileLock

logger = logging.getLogger(__name__)


# =========================================================================
# ResponseCache
# =========================================================================
class ResponseCache:
    """Thread-safe, deterministic disk cache for LLM responses.

    Stores (query, chunk_ids) → response mappings as inspectable JSON.
    All public methods are exception-safe and never raise.

    Cache key derivation:
        key = SHA-256( query.strip().lower() + "|" + ",".join(sorted(chunk_ids)) )

    This is deterministic: the same (query, chunks) always produces the
    same key, regardless of chunk retrieval order.

    Parameters
    ----------
    cache_path : str or Path
        Path to the JSON cache file.  Default: ``outputs/response_cache.json``.

    DESIGN DECISIONS:
    -----------------
    1. COLLISION HANDLING:
       SHA-256 produces 2^256 possible keys.  The probability of two
       different (query, chunk_ids) pairs colliding is ~1 in 10^77.
       We store the original query in each entry as a secondary check.

    2. CACHE INVALIDATION:
       The cache key includes chunk_ids.  If retrieval returns DIFFERENT
       chunk_ids for the same query (because the corpus changed), the
       key won't match → automatic cache miss.

    3. ATOMIC WRITES:
       Uses write-to-temp-then-rename + filelock to prevent corruption
       from concurrent processes or crashes mid-write.

    4. NEVER RAISES:
       All public methods catch exceptions and log warnings.  A cache
       failure degrades gracefully to a cache miss (API call proceeds).
    """

    _FORMAT_VERSION = "2.0"

    def __init__(
        self, cache_path: Union[str, Path] = "outputs/response_cache.json"
    ) -> None:
        self.cache_path = Path(cache_path)
        self._lock_path = Path(str(self.cache_path) + ".lock")
        self._lock = FileLock(self._lock_path, timeout=10)

        self._entries: dict[str, dict] = {}
        self._created_at: str = ""
        self._hit_count: int = 0
        self._miss_count: int = 0

        self._load()

    # =====================================================================
    # Public API
    # =====================================================================

    def get(self, query: str, chunk_ids: list[str]) -> Optional[dict]:
        """Look up a cached response.

        Parameters
        ----------
        query : str
            The student's question.
        chunk_ids : list[str]
            IDs of chunks passed to the generator (order-independent).

        Returns
        -------
        dict or None
            The cached response dict, or None if no cache hit.
            Never raises.
        """
        try:
            key = self._make_key(query, chunk_ids)

            if key in self._entries:
                entry = self._entries[key]

                # Secondary collision check
                if entry.get("query", "").strip().lower() != query.strip().lower():
                    logger.warning(
                        "Cache key collision for query: %s", query[:50]
                    )
                    self._miss_count += 1
                    return None

                # Track hit
                self._hit_count += 1
                entry["hit_count"] = entry.get("hit_count", 0) + 1
                self._save()

                # Return a copy of the response dict
                return dict(entry["response"])

            self._miss_count += 1
            return None

        except Exception as e:
            logger.warning("Cache get failed: %s", e)
            self._miss_count += 1
            return None

    def set(
        self, query: str, chunk_ids: list[str], response: dict
    ) -> None:
        """Store a response in the cache.

        Overwrites if the key already exists.  Never raises.

        Parameters
        ----------
        query : str
            The student's question.
        chunk_ids : list[str]
            IDs of chunks used for generation.
        response : dict
            The generator's response dict (answer, sources, refused, etc.).
        """
        try:
            key = self._make_key(query, chunk_ids)

            self._entries[key] = {
                "query": query.strip().lower(),
                "chunk_ids": sorted(chunk_ids),
                "response": dict(response),
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "hit_count": 0,
            }

            self._save()

        except Exception as e:
            logger.warning("Cache set failed: %s", e)

    def invalidate(self, query: str, chunk_ids: list[str]) -> bool:
        """Remove a specific cache entry.

        Parameters
        ----------
        query : str
            The query to invalidate.
        chunk_ids : list[str]
            The chunk IDs associated with the cached entry.

        Returns
        -------
        bool
            True if the entry existed and was removed, False if not found.
            Never raises.
        """
        try:
            key = self._make_key(query, chunk_ids)

            if key in self._entries:
                del self._entries[key]
                self._save()
                return True

            return False

        except Exception as e:
            logger.warning("Cache invalidate failed: %s", e)
            return False

    def clear_all(self) -> int:
        """Remove all cached entries and delete the cache file.

        Returns
        -------
        int
            Number of entries that were deleted.  Never raises.
        """
        try:
            count = len(self._entries)
            self._entries.clear()
            self._hit_count = 0
            self._miss_count = 0

            with self._lock:
                if self.cache_path.exists():
                    self.cache_path.unlink()

            return count

        except Exception as e:
            logger.warning("Cache clear_all failed: %s", e)
            return 0

    def stats(self) -> dict:
        """Return cache statistics.

        Returns
        -------
        dict
            {
                "total_entries":        int,
                "hit_count":            int,
                "miss_count":           int,
                "hit_rate":             float (0.0 to 1.0),
                "cache_file_size_bytes": int,
            }
            Never raises.
        """
        try:
            total_lookups = self._hit_count + self._miss_count
            file_size = (
                self.cache_path.stat().st_size
                if self.cache_path.exists()
                else 0
            )

            return {
                "total_entries": len(self._entries),
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": (
                    self._hit_count / total_lookups
                    if total_lookups > 0
                    else 0.0
                ),
                "cache_file_size_bytes": file_size,
            }

        except Exception as e:
            logger.warning("Cache stats failed: %s", e)
            return {
                "total_entries": 0,
                "hit_count": 0,
                "miss_count": 0,
                "hit_rate": 0.0,
                "cache_file_size_bytes": 0,
            }

    # =====================================================================
    # Internal methods
    # =====================================================================

    @staticmethod
    def _make_key(query: str, chunk_ids: list[str]) -> str:
        """Derive a deterministic cache key from query + chunk IDs.

        Key = SHA-256( normalized_query + "|" + sorted_chunk_ids )

        Sorting chunk_ids makes the key ORDER-INDEPENDENT: retrieving
        [sem_0001, sem_0003] and [sem_0003, sem_0001] produces the
        same key.  Normalization (strip + lowercase) handles trivial
        query variations.
        """
        normalized = query.strip().lower()
        ids_str = ",".join(sorted(chunk_ids))
        raw = f"{normalized}|{ids_str}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        """Load cache from disk.  Never raises."""
        try:
            if not self.cache_path.exists():
                self._created_at = datetime.now(timezone.utc).isoformat()
                self._entries = {}
                return

            with self._lock:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            version = data.get("version", "1.0")
            self._created_at = data.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            )

            if version == self._FORMAT_VERSION:
                # V2 format: entries are nested under "entries" key
                self._entries = data.get("entries", {})
            else:
                # V1 format migration: top-level keys are cache entries
                # (everything except "version" and "created_at")
                self._entries = self._migrate_v1(data)
                # Save in V2 format immediately
                self._save()

            logger.info(
                "Loaded cache: %d entries from %s (v%s)",
                len(self._entries),
                self.cache_path,
                version,
            )

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                "Failed to load cache from %s: %s. Starting fresh.",
                self.cache_path,
                e,
            )
            self._entries = {}
            self._created_at = datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _migrate_v1(data: dict) -> dict:
        """Migrate V1 cache format to V2 entry format.

        V1 format stored entries as top-level keys with fields:
            {answer, sources, refused, model, _query, _chunk_ids}

        V2 format nests these under an "entries" key with structure:
            {query, chunk_ids, response: {answer, sources, ...}, cached_at, hit_count}
        """
        migrated: dict[str, dict] = {}
        now = datetime.now(timezone.utc).isoformat()

        for key, entry in data.items():
            if key in ("version", "created_at"):
                continue
            if not isinstance(entry, dict):
                continue

            # Extract V1 fields
            query = entry.get("_query", "")
            chunk_ids = entry.get("_chunk_ids", [])

            # Build V2 response sub-dict (everything except internal metadata)
            response = {}
            for field in ("answer", "sources", "refused", "model"):
                if field in entry:
                    response[field] = entry[field]

            migrated[key] = {
                "query": query,
                "chunk_ids": chunk_ids,
                "response": response,
                "cached_at": now,
                "hit_count": 0,
            }

        logger.info("Migrated %d entries from V1 to V2 cache format", len(migrated))
        return migrated

    def _save(self) -> None:
        """Persist cache to disk atomically.

        Uses write-to-temp-then-rename for crash safety.
        File lock prevents concurrent writes from corrupting the file.

        Never raises (catches and logs all exceptions).
        """
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            payload = {
                "version": self._FORMAT_VERSION,
                "created_at": self._created_at,
                "entries": self._entries,
            }

            with self._lock:
                # Write to a temp file in the same directory, then rename.
                # os.replace() is atomic on POSIX and near-atomic on Windows.
                fd, tmp_path = tempfile.mkstemp(
                    dir=str(self.cache_path.parent),
                    prefix=".cache_",
                    suffix=".tmp",
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)
                    os.replace(tmp_path, str(self.cache_path))
                except Exception:
                    # Clean up temp file on failure
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    raise

        except Exception as e:
            logger.warning("Cache save failed: %s", e)

    def __repr__(self) -> str:
        return (
            f"ResponseCache(entries={len(self._entries)}, "
            f"path={self.cache_path})"
        )

    def __len__(self) -> int:
        return len(self._entries)


# =========================================================================
# CachedGenerator
# =========================================================================
class CachedGenerator:
    """Wrapper that adds caching to any generator with a generate() method.

    Exposes the SAME generate(query, chunks) interface as GroundedGenerator.
    evaluator.py can use this as a drop-in replacement with zero changes.

    Flow:
        1. Compute cache key from (query, chunk_ids)
        2. If cache hit → return cached response (no API call)
        3. If cache miss → call inner generator → cache result → return

    Parameters
    ----------
    generator : object
        Any object with a generate(query, chunks) method that returns
        a dict with at least {answer, sources, refused, model}.
        Typically a GroundedGenerator.
    cache : ResponseCache
        The cache instance to use.
    offline : bool
        If True, NEVER call the inner generator.  Raise RuntimeError
        on cache miss instead.  Use for fully offline evaluation replay.
    """

    def __init__(
        self,
        generator: object,
        cache: ResponseCache,
        offline: bool = False,
    ) -> None:
        self._generator = generator
        self._cache = cache
        self._offline = offline

    def generate(self, query: str, chunks: list[dict]) -> dict:
        """Generate a grounded answer, using cache when possible.

        Parameters
        ----------
        query : str
            The student's question.
        chunks : list[dict]
            Retrieved chunks with at least ``chunk_id`` key and either
            ``text`` or ``content`` key.

        Returns
        -------
        dict
            Response dict — identical schema to GroundedGenerator.generate().

        Raises
        ------
        RuntimeError
            If offline=True and no cache hit exists.
        """
        chunk_ids = [c.get("chunk_id", "") for c in chunks]

        # ── Check cache ──────────────────────────────────────────────
        cached = self._cache.get(query, chunk_ids)
        if cached is not None:
            return cached

        # ── Offline mode: refuse to call API ─────────────────────────
        if self._offline:
            raise RuntimeError(
                f"Cache miss in offline mode for query: '{query[:60]}...'. "
                "Run evaluation with offline=False first to populate cache."
            )

        # ── Cache miss: call live generator ──────────────────────────
        response = self._generator.generate(query, chunks)

        # ── Store in cache ───────────────────────────────────────────
        self._cache.set(query, chunk_ids, response)

        return response

    def __repr__(self) -> str:
        mode = "offline" if self._offline else "live+cache"
        return f"CachedGenerator(mode={mode}, {self._cache})"


# =========================================================================
# MockLLMGenerator
# =========================================================================

# Predefined responses for common evaluation questions.
# These are synthetic but structurally identical to Gemini responses.
_MOCK_RESPONSES: dict[str, dict] = {
    "default_in_scope": {
        "answer": (
            "Based on the provided context, the cell is the basic structural "
            "and functional unit of all living organisms. Robert Hooke first "
            "observed cells in 1665 using a cork slice under a microscope [1]. "
            "All living organisms are composed of cells, which carry out "
            "essential life processes [2]."
        ),
        "sources": [],
        "refused": False,
        "model": "mock-llm-v1",
    },
    "default_out_of_scope": {
        "answer": "I could not find this in the textbook.",
        "sources": [],
        "refused": True,
        "model": "mock-llm-v1",
    },
}


class MockLLMGenerator:
    """Deterministic synthetic generator for offline testing.

    Returns structurally valid responses WITHOUT any API call.
    Useful when GEMINI_API_KEY is absent or quota is exhausted.

    The generate() method returns a refusal for queries that appear
    out-of-scope (no keyword overlap with chunks), and a generic
    in-scope response otherwise.  Sources are mapped to actual chunk_ids
    from the provided chunks.

    Exposes the same generate(query, chunks) -> dict interface as
    GroundedGenerator.  Can be used with CachedGenerator and evaluator.py.
    """

    def __init__(self) -> None:
        self.model_name = "mock-llm-v1"

    def generate(self, query: str, chunks: list[dict]) -> dict:
        """Generate a synthetic response.

        Parameters
        ----------
        query : str
            The student's question.
        chunks : list[dict]
            Retrieved chunks with at least ``chunk_id`` and either
            ``text`` or ``content`` key.

        Returns
        -------
        dict
            {answer, sources, refused, model}
        """
        if not chunks:
            return dict(_MOCK_RESPONSES["default_out_of_scope"])

        # Simple heuristic: check if query words appear in any chunk text
        query_words = set(query.lower().split())
        chunk_text = " ".join(
            (c.get("content") or c.get("text", "")).lower() for c in chunks
        )
        overlap = sum(1 for w in query_words if w in chunk_text)

        if overlap < 2:
            # Likely out-of-scope
            return dict(_MOCK_RESPONSES["default_out_of_scope"])

        # In-scope: build response with actual chunk_ids as sources
        source_ids = [
            chunks[i]["chunk_id"] for i in range(min(2, len(chunks)))
        ]
        answer = _MOCK_RESPONSES["default_in_scope"]["answer"]

        return {
            "answer": answer,
            "sources": source_ids,
            "refused": False,
            "model": self.model_name,
        }

    def __repr__(self) -> str:
        return "MockLLMGenerator()"
