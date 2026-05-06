"""
src/embedder.py -- Dense retrieval via sentence-transformers + FAISS.

Provides FAISSRetriever: a dense retriever that embeds chunks using a
sentence-transformer model and indexes them in a FAISS IndexFlatIP index.

Exposes retrieve(query, top_k) with identical return schema to BM25Retriever.

Index files:
  - outputs/faiss_index.bin       -- serialized FAISS index
  - outputs/faiss_index.meta.json -- metadata (chunk hash, model, dim)
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from pathlib import Path
from typing import Union

import numpy as np

# ---------------------------------------------------------------------------
# Lazy imports for optional heavy dependencies.
# FAISS and sentence-transformers are V2 additions.
# ---------------------------------------------------------------------------
_DEPS_AVAILABLE = False
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    _DEPS_AVAILABLE = True
except ImportError:
    faiss = None
    SentenceTransformer = None

logger = logging.getLogger(__name__)


# =========================================================================
# Configuration defaults
# =========================================================================

# ── Embedding model ──────────────────────────────────────────────────────
#
# MODEL: all-MiniLM-L6-v2
#
# This is a BI-ENCODER (not a cross-encoder).  It produces a single 384-dim
# embedding per text, enabling fast approximate nearest-neighbor search.
#
# Why this model:
#   - 384-dim output: small index size (85 chunks * 384 * 4 bytes = 130 KB)
#   - ~22M params: loads in <2 seconds, ~80 MB download
#   - Trained on 1B sentence pairs: strong semantic similarity performance
#   - Max sequence length: 256 tokens (our chunks average ~150 words = ~180
#     BERT tokens, comfortably under the limit)
#   - Widely used baseline in RAG literature, making results reproducible
#
# Alternative: all-mpnet-base-v2 (768-dim, higher quality, 2x larger index,
# 3x slower inference).  Overkill for 85 chunks.
#
_DEFAULT_MODEL = "all-MiniLM-L6-v2"

# ── Batch size ───────────────────────────────────────────────────────────
#
# BATCH SIZE: 32
#
# Memory analysis for a laptop with 8 GB RAM:
#   Model weights (all-MiniLM-L6-v2):     ~80 MB
#   Per-batch activations:
#     32 samples * 256 tokens * 384 hidden * 4 bytes = ~12.6 MB
#   Total peak during inference:           ~100 MB
#   Python + OS overhead:                  ~1-2 GB
#   Available for other processes:         ~6 GB
#
# 85 chunks / 32 = 3 batches.  Even batch_size=85 (all at once) would use
# only ~33 MB of activation memory -- safe on 8 GB.  We use 32 for forward
# compatibility with multi-chapter corpora (1000+ chunks) where a single
# batch would consume ~400 MB.
#
_DEFAULT_BATCH_SIZE = 32

# ── Index metadata filename ──────────────────────────────────────────────
_META_SUFFIX = ".meta.json"


# =========================================================================
# FAISSRetriever
# =========================================================================
class FAISSRetriever:
    """Dense retriever using sentence-transformer embeddings + FAISS IndexFlatIP.

    Two phases:
      1. BUILD: Load chunks, compute embeddings, L2-normalize, build FAISS
         index, save to disk.
      2. QUERY: Load index from disk (or use in-memory), embed the query,
         L2-normalize, search for top_k nearest neighbors.

    Parameters
    ----------
    chunks_path : str or Path
        Path to chunks_semantic.json (the authoritative chunk source).
    index_path : str or Path
        Path for the FAISS index file (e.g., outputs/faiss_index.bin).
        A sidecar metadata file (*.meta.json) is stored alongside.
    model_name : str, optional
        HuggingFace model ID for the sentence-transformer.
        Default: ``all-MiniLM-L6-v2`` (384-dim, ~22M params).
    batch_size : int, optional
        Embedding batch size.  Default: 32 (safe for 8 GB RAM laptops).

    Example
    -------
    >>> retriever = FAISSRetriever("outputs/chunks_semantic.json",
    ...                            "outputs/faiss_index.bin")
    >>> results = retriever.retrieve("What is a cell membrane?", top_k=3)
    >>> results[0]["score"]     # cosine similarity in [-1, 1]
    0.742
    >>> results[0]["chunk_id"]  # preserves chunk metadata
    'sem_0023'

    Return Schema (identical to BM25Retriever)
    -------------------------------------------
    Each result dict contains ALL original chunk keys from chunks_semantic.json:
        chunk_id    : str   -- e.g., "sem_0001"
        source_id   : str   -- e.g., "chunk_001"
        page        : int   -- 1-indexed PDF page
        type        : str   -- "concept", "activity", "question", etc.
        text        : str   -- chunk content
        token_count : int   -- BERT token count
    Plus one additional key:
        score       : float -- cosine similarity (higher = more relevant)

    INVARIANTS PRESERVED:
        1. retrieve(query, top_k) signature -- identical to BM25Retriever
        2. Return schema -- list[dict] with chunk keys + "score"
        3. chunks_semantic.json is never modified
        4. Index is deterministically rebuildable from chunks + model
        5. evaluator.py compatibility -- zero changes required

    KNOWN LIMITATIONS:
        1. First build downloads the embedding model (~80 MB).
        2. IndexFlatIP is exact (brute-force) search -- O(n) per query.
           For 85 chunks this is <1ms.  For 10,000+ chunks, switch to
           IndexIVFFlat or IndexHNSW.
        3. Cosine similarity scores can be negative (unlike BM25).
           The notebook's GOOD/WEAK/FAIL thresholds do not apply.
        4. If chunks_semantic.json changes, the index is automatically
           rebuilt on next load (detected via SHA-256 hash).
    """

    def __init__(
        self,
        chunks_path: Union[str, Path],
        index_path: Union[str, Path],
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        if not _DEPS_AVAILABLE:
            raise ImportError(
                "FAISSRetriever requires 'faiss-cpu' and 'sentence-transformers'. "
                "Install with: pip install faiss-cpu sentence-transformers"
            )

        self.chunks_path = Path(chunks_path)
        self.index_path = Path(index_path)
        self.meta_path = Path(str(index_path) + _META_SUFFIX)
        self.model_name = model_name
        self.batch_size = batch_size

        # ── Load chunks (read-only, never modified) ──────────────────────
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunk file not found: {self.chunks_path}")
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks: list[dict] = json.load(f)
        self._n_docs = len(self.chunks)

        # ── Compute chunk file hash for integrity checking ───────────────
        # SHA-256 of the raw file bytes.  If chunks_semantic.json is
        # modified (even whitespace), the hash changes and the index
        # is rebuilt.  This guarantees the index always matches the
        # current chunk content.
        self._chunks_hash = self._compute_file_hash(self.chunks_path)

        # ── Load embedding model ─────────────────────────────────────────
        self._model = SentenceTransformer(model_name)
        self._embedding_dim = self._model.get_embedding_dimension()

        # ── Load or build the FAISS index ────────────────────────────────
        self._index: faiss.IndexFlatIP = None
        self._load_or_build()

    # =====================================================================
    # Public API
    # =====================================================================

    def build_index(self) -> None:
        """Compute embeddings for all chunks and build a fresh FAISS index.

        This method is idempotent: calling it twice with the same
        chunks_semantic.json and model produces a bit-identical index.

        Steps:
          1. Encode all chunk texts → embeddings matrix (N x D)
          2. L2-normalize each embedding to unit length
          3. Build FAISS IndexFlatIP (inner product = cosine after L2 norm)
          4. Save index + metadata to disk
        """
        logger.info("Building FAISS index from %d chunks...", self._n_docs)

        # ── Step 1: Compute embeddings ───────────────────────────────────
        texts = [c["text"] for c in self.chunks]

        # SentenceTransformer.encode() with deterministic settings:
        #   - normalize_embeddings=False (we normalize manually for clarity)
        #   - show_progress_bar=True (user feedback during build)
        #   - convert_to_numpy=True (default, returns np.ndarray)
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we do this explicitly below
        )
        # embeddings shape: (n_docs, embedding_dim), dtype: float32

        # ── Step 2: L2-normalize ─────────────────────────────────────────
        #
        # MATHEMATICAL JUSTIFICATION:
        #
        #   Cosine similarity:  cos(a, b) = (a . b) / (||a|| * ||b||)
        #
        #   If we L2-normalize both vectors:  a' = a / ||a||,  b' = b / ||b||
        #   Then: ||a'|| = ||b'|| = 1
        #   And:  cos(a, b) = a' . b'  =  inner_product(a', b')
        #
        #   FAISS IndexFlatIP computes inner product.
        #   Therefore: L2-normalized vectors + IndexFlatIP = cosine similarity.
        #
        #   This is more efficient than using IndexFlatL2 and converting
        #   L2 distances to cosine: cos = 1 - d^2/2 (requires extra math).
        #
        #   We use faiss.normalize_L2() which normalizes IN-PLACE.
        #   This modifies the embeddings array directly (no copy).
        #
        faiss.normalize_L2(embeddings)

        # ── Step 3: Build FAISS index ────────────────────────────────────
        #
        # IndexFlatIP: exact inner-product search (brute-force).
        # For 85 chunks (130 KB index), this is <1ms per query.
        # No training required, no approximation, deterministic.
        #
        self._index = faiss.IndexFlatIP(self._embedding_dim)
        self._index.add(embeddings)

        # ── Step 4: Save to disk ─────────────────────────────────────────
        self._save_index()

        logger.info(
            "FAISS index built: %d vectors, %d dims, saved to %s",
            self._index.ntotal, self._embedding_dim, self.index_path,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top_k most relevant chunks for *query*.

        Each returned dict is a copy of the original chunk dict
        with an added ``score`` key (float, cosine similarity).

        Parameters
        ----------
        query : str
            The student's question.
        top_k : int
            Number of results to return.  Default: 5.

        Returns
        -------
        list[dict]
            Top-k chunks ranked by cosine similarity (descending).
            Each dict has all original chunk keys plus ``score``.

            Schema (identical to BM25Retriever):
            {
                "chunk_id":    str,   # e.g., "sem_0001"
                "source_id":   str,   # e.g., "chunk_001"
                "page":        int,   # 1-indexed PDF page
                "type":        str,   # "concept", "activity", etc.
                "text":        str,   # chunk content
                "token_count": int,   # BERT token count
                "score":       float  # cosine similarity (higher = better)
            }
        """
        # ── Embed the query ──────────────────────────────────────────────
        query_embedding = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        # L2-normalize query vector (must match index normalization)
        faiss.normalize_L2(query_embedding)

        # ── Search ───────────────────────────────────────────────────────
        # Clamp top_k to corpus size (FAISS raises error if k > ntotal)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_embedding, k)

        # scores shape: (1, k), indices shape: (1, k)
        scores = scores[0]    # flatten to 1D
        indices = indices[0]  # flatten to 1D

        # ── Build result dicts ───────────────────────────────────────────
        #
        # FAISS INTEGER ID MAPPING:
        #
        #   FAISS uses sequential integer IDs: 0, 1, 2, ..., N-1.
        #   These correspond directly to the order of chunks in
        #   chunks_semantic.json (which we loaded into self.chunks).
        #
        #   Mapping:  FAISS ID i  <-->  self.chunks[i]
        #
        #   This is guaranteed because:
        #     1. We load chunks in order from JSON (list preserves order)
        #     2. We add embeddings in the same order via index.add()
        #     3. FAISS assigns sequential IDs starting from 0
        #
        results = []
        for idx, score in zip(indices, scores):
            if idx < 0:
                # FAISS returns -1 for empty results (shouldn't happen
                # with IndexFlatIP, but defensive)
                continue
            entry = dict(self.chunks[idx])   # shallow copy
            entry["score"] = float(score)    # cosine similarity
            results.append(entry)

        return results

    # =====================================================================
    # Private methods
    # =====================================================================

    def _load_or_build(self) -> None:
        """Load the FAISS index from disk, or build it if necessary.

        RECOVERY BEHAVIOR when index exists but chunks have changed:

          The metadata sidecar file stores a SHA-256 hash of
          chunks_semantic.json at build time.  On load, we recompute the
          hash of the current chunks file and compare:

            - Hash MATCHES:   Load the existing index (fast path, ~5ms).
            - Hash MISMATCH:  Log a warning, delete stale index, rebuild.
            - Index MISSING:  Build from scratch.

          This ensures the index always reflects the current chunk content,
          even if chunks_semantic.json was modified externally.
        """
        if self.index_path.exists() and self.meta_path.exists():
            # ── Try loading existing index ───────────────────────────────
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                stored_hash = meta.get("chunks_hash", "")
                stored_model = meta.get("model_name", "")
                stored_n = meta.get("n_chunks", 0)

                # Validate: hash match, model match, count match
                if (stored_hash == self._chunks_hash
                        and stored_model == self.model_name
                        and stored_n == self._n_docs):
                    self._index = faiss.read_index(str(self.index_path))
                    logger.info(
                        "Loaded FAISS index from %s (%d vectors)",
                        self.index_path, self._index.ntotal,
                    )
                    return
                else:
                    # ── Stale index detected ─────────────────────────────
                    reasons = []
                    if stored_hash != self._chunks_hash:
                        reasons.append("chunks_semantic.json changed")
                    if stored_model != self.model_name:
                        reasons.append(f"model changed: {stored_model} -> {self.model_name}")
                    if stored_n != self._n_docs:
                        reasons.append(f"chunk count changed: {stored_n} -> {self._n_docs}")

                    warnings.warn(
                        f"Stale FAISS index detected ({', '.join(reasons)}). "
                        "Rebuilding...",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            except (json.JSONDecodeError, KeyError, Exception) as e:
                logger.warning("Failed to read index metadata: %s. Rebuilding.", e)

        # ── Build fresh index ────────────────────────────────────────────
        self.build_index()

    def _save_index(self) -> None:
        """Save the FAISS index and metadata to disk."""
        # Ensure output directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(self.index_path))

        # Save metadata sidecar
        meta = {
            "chunks_hash": self._chunks_hash,
            "chunks_path": str(self.chunks_path),
            "model_name": self.model_name,
            "embedding_dim": self._embedding_dim,
            "n_chunks": self._n_docs,
            "index_type": "IndexFlatIP",
            "normalized": True,
            "note": "Embeddings are L2-normalized. IndexFlatIP inner product = cosine similarity.",
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        """Compute SHA-256 hex digest of a file's raw bytes.

        Used to detect whether chunks_semantic.json has changed since
        the last index build.  Deterministic: same file content always
        produces the same hash.
        """
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()

    def __repr__(self) -> str:
        return (
            f"FAISSRetriever(n_docs={self._n_docs}, "
            f"dim={self._embedding_dim}, model={self.model_name})"
        )
