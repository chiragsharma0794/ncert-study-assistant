"""
src/retriever.py -- Retrieval engines for the NCERT RAG pipeline V2.

Three retrievers with an identical interface:
  - BM25Retriever    : Okapi BM25 sparse retrieval (V1 logic preserved)
  - FAISSRetriever   : Dense retrieval via sentence-transformers + FAISS
  - HybridRetriever  : Weighted fusion of BM25 (sparse) + FAISS (dense)

All accept ``chunks_path`` in __init__ and expose:
  .retrieve(query, top_k) -> list[dict]

Return schema (identical for all three):
  [{"chunk_id": str, "content": str, "score": float, "page": int, "type": str}]

Usage:
    from retriever import BM25Retriever
    r = BM25Retriever(chunks_path="outputs/chunks_semantic.json")
    results = r.retrieve("What is a cell?", top_k=5)
    print(results[0]["content"])  # chunk text
    print(results[0]["score"])    # relevance score
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports for optional dense-retrieval dependencies.
# BM25Retriever works with zero optional dependencies.
# FAISSRetriever and HybridRetriever require faiss-cpu + sentence-transformers.
# ---------------------------------------------------------------------------
_FAISS_AVAILABLE = False
try:
    import faiss as _faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _faiss = None

_ST_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _SentenceTransformer = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization for sparse retrievers.

    Uses a simple regex tokenizer rather than NLTK/BERT because
    BM25 operates on word-level tokens, not subwords.
    Lowercasing ensures case-insensitive matching.
    """
    return _WORD_RE.findall(text.lower())


def load_chunks(path: Union[str, Path]) -> list[dict]:
    """Load a chunk JSON file and return the list of chunk dicts.

    Parameters
    ----------
    path : str or Path
        Path to a JSON file containing a list of chunk dicts.
        Each dict must have at least ``text`` and ``chunk_id`` keys.

    Returns
    -------
    list[dict]
        The parsed list of chunk dictionaries.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_result(chunk: dict, score: float) -> dict:
    """Map a raw chunk dict into the standard V2 return schema.

    The V2 schema uses ``content`` (not ``text``) for the chunk body.
    The raw chunk's ``page`` field is coerced to int for consistency.

    Parameters
    ----------
    chunk : dict
        A raw chunk dict from chunks_semantic.json.
    score : float
        The retrieval relevance score for this chunk.

    Returns
    -------
    dict
        ``{"chunk_id": str, "content": str, "score": float,
          "page": int, "type": str}``
    """
    page_raw = chunk.get("page", 0)
    try:
        page = int(page_raw)
    except (TypeError, ValueError):
        page = 0

    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "content":  chunk.get("text", ""),
        "score":    float(score),
        "page":     page,
        "type":     chunk.get("type", ""),
    }


# =========================================================================
# BM25 Retriever  (V1 logic preserved)
# =========================================================================
class BM25Retriever:
    """Okapi BM25 sparse retriever.

    Uses the rank_bm25 BM25Okapi implementation with lowercase word
    tokenization.  Deterministic: same query + same corpus = same results.

    Parameters
    ----------
    chunks_path : str
        Path to the chunks JSON file (e.g. ``"outputs/chunks_semantic.json"``).

    Example
    -------
    >>> retriever = BM25Retriever(chunks_path="outputs/chunks_semantic.json")
    >>> results = retriever.retrieve("What is osmosis?", top_k=3)
    >>> results[0]["score"]
    4.237
    """

    def __init__(self, chunks_path: str) -> None:
        self._chunks_path = Path(chunks_path)
        self._chunks = load_chunks(self._chunks_path)
        self._corpus_tokens = [_tokenize(c["text"]) for c in self._chunks]
        self._index = BM25Okapi(self._corpus_tokens)
        self._n_docs = len(self._chunks)
        logger.info(
            "BM25Retriever initialized: %d documents from %s",
            self._n_docs, self._chunks_path,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top_k most relevant chunks for *query*.

        Scores are raw BM25 scores (unbounded, higher = more relevant).
        The ranking is deterministic for a given query and corpus.

        Parameters
        ----------
        query : str
            The student's question.
        top_k : int
            Number of results to return.  Default: 5.

        Returns
        -------
        list[dict]
            Top-k chunks, each with keys:
            ``chunk_id``, ``content``, ``score``, ``page``, ``type``.
        """
        query_tokens = _tokenize(query)
        scores = self._index.get_scores(query_tokens)

        # Rank by score descending, take top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(_format_result(self._chunks[idx], scores[idx]))

        return results

    @property
    def chunks(self) -> list[dict]:
        """Access the raw chunk list (read-only)."""
        return self._chunks

    @property
    def n_docs(self) -> int:
        """Number of indexed documents."""
        return self._n_docs

    def __repr__(self) -> str:
        return f"BM25Retriever(n_docs={self._n_docs})"


# =========================================================================
# FAISS Retriever  (dense, lazy-loading)
# =========================================================================

# ── Default embedding model ─────────────────────────────────────────────
#
# MODEL: all-MiniLM-L6-v2
#
#   - 384-dim output: small index size (85 chunks × 384 × 4 bytes = 130 KB)
#   - ~22M params: loads in <2 seconds, ~80 MB download
#   - Trained on 1B sentence pairs: strong semantic similarity
#   - Max sequence length: 256 tokens (our chunks average ~150 words
#     ≈ ~180 BERT tokens, comfortably under the limit)
#
_DEFAULT_FAISS_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_BATCH_SIZE = 32
_META_SUFFIX = ".meta.json"


class FAISSRetriever:
    """Dense retriever using sentence-transformer embeddings + FAISS.

    Two-phase operation:
      1. **BUILD**: Load chunks, compute embeddings, L2-normalize,
         build FAISS IndexFlatIP, save to disk.
      2. **QUERY**: Load index from disk (or use in-memory), embed the
         query, L2-normalize, search for top_k nearest neighbors.

    Lazy-loading: the FAISS index is built on the first ``retrieve()``
    call if no pre-built index file is found on disk.  Subsequent calls
    reuse the in-memory index.  If ``chunks_semantic.json`` changes
    (detected via SHA-256 hash), the index is automatically rebuilt.

    Parameters
    ----------
    chunks_path : str
        Path to the chunks JSON file.
    index_path : str, optional
        Path for the FAISS index file.  If None, defaults to
        ``<chunks_path_stem>.faiss`` in the same directory.
    model_name : str, optional
        HuggingFace model ID for the sentence-transformer.
        Default: ``all-MiniLM-L6-v2`` (384-dim, ~22M params).
    batch_size : int, optional
        Embedding batch size.  Default: 32 (safe for 8 GB RAM).

    Example
    -------
    >>> retriever = FAISSRetriever("outputs/chunks_semantic.json")
    >>> results = retriever.retrieve("What gives flowers colour?", top_k=3)
    >>> results[0]["score"]   # cosine similarity
    0.742
    """

    def __init__(
        self,
        chunks_path: str,
        index_path: Optional[str] = None,
        model_name: str = _DEFAULT_FAISS_MODEL,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self._chunks_path = Path(chunks_path)
        self._model_name = model_name
        self._batch_size = batch_size

        # Derive default index path from chunks path
        if index_path:
            self._index_path = Path(index_path)
        else:
            self._index_path = self._chunks_path.with_suffix(".faiss")
        self._meta_path = Path(str(self._index_path) + _META_SUFFIX)

        # Load chunks immediately (lightweight JSON read)
        self._chunks = load_chunks(self._chunks_path)
        self._n_docs = len(self._chunks)

        # Lazy-initialized state (set on first retrieve call)
        self._model: Optional[object] = None
        self._faiss_index = None
        self._initialized = False
        self._embedding_dim: int = 0

        logger.info(
            "FAISSRetriever created: %d documents (lazy init, model=%s)",
            self._n_docs, self._model_name,
        )

    # -----------------------------------------------------------------
    # Lazy initialization
    # -----------------------------------------------------------------
    def _ensure_initialized(self) -> None:
        """Ensure the embedding model and FAISS index are loaded.

        Called automatically on the first ``retrieve()`` call.
        Raises ImportError if required dependencies are missing.
        """
        if self._initialized:
            return

        if not _FAISS_AVAILABLE:
            raise ImportError(
                "FAISSRetriever requires 'faiss-cpu'. "
                "Install with: pip install faiss-cpu"
            )
        if not _ST_AVAILABLE:
            raise ImportError(
                "FAISSRetriever requires 'sentence-transformers'. "
                "Install with: pip install sentence-transformers"
            )

        # Load embedding model
        logger.info("Loading embedding model: %s", self._model_name)
        self._model = _SentenceTransformer(self._model_name)
        self._embedding_dim = self._model.get_embedding_dimension()

        # Load or build the FAISS index
        self._load_or_build()
        self._initialized = True

    def _load_or_build(self) -> None:
        """Load the FAISS index from disk, or build if stale/missing.

        Staleness is detected by comparing the SHA-256 hash of the
        current chunks file against the hash stored in the metadata
        sidecar file.  If they differ (or metadata is missing), the
        index is rebuilt.
        """
        chunks_hash = self._compute_file_hash(self._chunks_path)

        if self._index_path.exists() and self._meta_path.exists():
            try:
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                stored_hash = meta.get("chunks_hash", "")
                stored_model = meta.get("model_name", "")
                stored_n = meta.get("n_chunks", 0)

                if (stored_hash == chunks_hash
                        and stored_model == self._model_name
                        and stored_n == self._n_docs):
                    self._faiss_index = _faiss.read_index(
                        str(self._index_path)
                    )
                    logger.info(
                        "Loaded FAISS index from %s (%d vectors)",
                        self._index_path, self._faiss_index.ntotal,
                    )
                    return
                else:
                    reasons = []
                    if stored_hash != chunks_hash:
                        reasons.append("chunks file changed")
                    if stored_model != self._model_name:
                        reasons.append(
                            f"model changed: {stored_model} → {self._model_name}"
                        )
                    if stored_n != self._n_docs:
                        reasons.append(
                            f"chunk count: {stored_n} → {self._n_docs}"
                        )
                    warnings.warn(
                        f"Stale FAISS index detected ({', '.join(reasons)}). "
                        "Rebuilding...",
                        RuntimeWarning,
                        stacklevel=3,
                    )
            except (json.JSONDecodeError, KeyError, Exception) as e:
                logger.warning(
                    "Failed to read index metadata: %s. Rebuilding.", e
                )

        # Build fresh index
        self._build_index(chunks_hash)

    def _build_index(self, chunks_hash: str) -> None:
        """Compute embeddings and build a fresh FAISS index.

        Steps:
          1. Encode all chunk texts → embeddings matrix (N × D)
          2. L2-normalize each embedding to unit length
          3. Build FAISS IndexFlatIP (inner product = cosine similarity
             after L2 normalization)
          4. Save index + metadata sidecar to disk
        """
        logger.info("Building FAISS index from %d chunks...", self._n_docs)

        texts = [c["text"] for c in self._chunks]
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        # L2-normalize: inner product on unit vectors = cosine similarity
        _faiss.normalize_L2(embeddings)

        self._faiss_index = _faiss.IndexFlatIP(self._embedding_dim)
        self._faiss_index.add(embeddings)

        # Save to disk
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        _faiss.write_index(self._faiss_index, str(self._index_path))

        meta = {
            "chunks_hash": chunks_hash,
            "chunks_path": str(self._chunks_path),
            "model_name": self._model_name,
            "embedding_dim": self._embedding_dim,
            "n_chunks": self._n_docs,
            "index_type": "IndexFlatIP",
            "normalized": True,
        }
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            "FAISS index built: %d vectors, %d dims, saved to %s",
            self._faiss_index.ntotal, self._embedding_dim, self._index_path,
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top_k most relevant chunks for *query*.

        Uses cosine similarity via FAISS inner-product search on
        L2-normalized embeddings.  The index is lazily built on the
        first call if not already available on disk.

        Parameters
        ----------
        query : str
            The student's question.
        top_k : int
            Number of results to return.  Default: 5.

        Returns
        -------
        list[dict]
            Top-k chunks, each with keys:
            ``chunk_id``, ``content``, ``score``, ``page``, ``type``.
        """
        self._ensure_initialized()

        # Embed and normalize query
        query_embedding = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        _faiss.normalize_L2(query_embedding)

        # Search (clamp top_k to corpus size)
        k = min(top_k, self._faiss_index.ntotal)
        scores, indices = self._faiss_index.search(query_embedding, k)

        scores = scores[0]     # flatten (1, k) → (k,)
        indices = indices[0]

        results = []
        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            results.append(_format_result(self._chunks[idx], score))

        return results

    @property
    def chunks(self) -> list[dict]:
        """Access the raw chunk list (read-only)."""
        return self._chunks

    @property
    def n_docs(self) -> int:
        """Number of indexed documents."""
        return self._n_docs

    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        """Compute SHA-256 hex digest of a file's raw bytes."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()

    def __repr__(self) -> str:
        status = "ready" if self._initialized else "lazy"
        return (
            f"FAISSRetriever(n_docs={self._n_docs}, "
            f"model={self._model_name}, status={status})"
        )


# =========================================================================
# Hybrid Retriever  (weighted score fusion)
# =========================================================================
class HybridRetriever:
    """Weighted fusion of BM25 (sparse) + FAISS (dense) retrieval.

    Retrieves candidates from both BM25 and FAISS, normalizes scores
    to [0, 1], and combines them using configurable weights:

        final_score = bm25_weight × norm_bm25 + dense_weight × norm_faiss

    BM25 scores are min-max normalized within the candidate set.
    FAISS scores are cosine similarities (already in [-1, 1]) shifted
    to [0, 1] via ``(score + 1) / 2``.

    Parameters
    ----------
    chunks_path : str
        Path to the chunks JSON file.
    bm25_weight : float
        Weight for the BM25 sparse score.  Default: 0.6.
    dense_weight : float
        Weight for the FAISS dense score.  Default: 0.4.
    model_name : str, optional
        HuggingFace model ID for the dense encoder.
        Default: ``all-MiniLM-L6-v2``.
    index_path : str, optional
        Path for the FAISS index file.  If None, derived from chunks_path.

    Example
    -------
    >>> retriever = HybridRetriever("outputs/chunks_semantic.json")
    >>> results = retriever.retrieve("What gives flowers colour?", top_k=5)
    >>> results[0]["score"]   # fused score in [0, 1]
    0.847

    DESIGN: SCORE FUSION vs. CROSS-ENCODER RE-RANKING
    --------------------------------------------------
    Score fusion combines pre-computed scores from independent systems.
    Cross-encoder re-ranking runs each (query, doc) pair through a
    joint transformer.  Re-ranking is more accurate but 10-50× slower.

    For this corpus (~85 chunks), score fusion provides a good balance:
    fast enough for interactive use, and meaningfully improves recall
    over BM25 alone (catches semantic matches that BM25 misses).
    """

    def __init__(
        self,
        chunks_path: str,
        bm25_weight: float = 0.6,
        dense_weight: float = 0.4,
        model_name: str = _DEFAULT_FAISS_MODEL,
        index_path: Optional[str] = None,
    ) -> None:
        self._chunks_path = chunks_path
        self._bm25_weight = bm25_weight
        self._dense_weight = dense_weight

        # Build sub-retrievers (share the same chunks_path)
        self._bm25 = BM25Retriever(chunks_path)
        self._faiss = FAISSRetriever(
            chunks_path,
            model_name=model_name,
            index_path=index_path,
        )
        self._n_docs = self._bm25.n_docs

        logger.info(
            "HybridRetriever initialized: bm25_weight=%.2f, "
            "dense_weight=%.2f, n_docs=%d",
            bm25_weight, dense_weight, self._n_docs,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top_k most relevant chunks via score fusion.

        Algorithm:
          1. Retrieve ``top_k × 3`` candidates from both BM25 and FAISS.
          2. Min-max normalize BM25 scores to [0, 1].
          3. Shift FAISS cosine scores from [-1, 1] to [0, 1].
          4. Compute fused score per candidate.
          5. Return top_k by fused score, descending.

        Parameters
        ----------
        query : str
            The student's question.
        top_k : int
            Number of results to return.  Default: 5.

        Returns
        -------
        list[dict]
            Top-k chunks, each with keys:
            ``chunk_id``, ``content``, ``score``, ``page``, ``type``.
            Scores are fused values in [0, 1].
        """
        # Retrieve more candidates than needed for better fusion quality.
        # 3× is standard in two-stage retrieval: high enough to capture
        # relevant chunks that one retriever misses, bounded enough to
        # keep latency low.
        n_candidates = min(top_k * 3, self._n_docs)

        bm25_results = self._bm25.retrieve(query, top_k=n_candidates)
        faiss_results = self._faiss.retrieve(query, top_k=n_candidates)

        # ── Normalize BM25 scores to [0, 1] via min-max ──────────────
        bm25_scores = {r["chunk_id"]: r["score"] for r in bm25_results}
        if bm25_scores:
            max_bm25 = max(bm25_scores.values())
            min_bm25 = min(bm25_scores.values())
            range_bm25 = max_bm25 - min_bm25
            if range_bm25 > 0:
                bm25_norm = {
                    k: (v - min_bm25) / range_bm25
                    for k, v in bm25_scores.items()
                }
            else:
                # All BM25 scores identical → uniform 1.0
                bm25_norm = {k: 1.0 for k in bm25_scores}
        else:
            bm25_norm = {}

        # ── Normalize FAISS cosine scores from [-1, 1] to [0, 1] ─────
        #
        # Cosine similarity ∈ [-1, 1].  We shift to [0, 1] via:
        #   normalized = (cosine + 1) / 2
        # This preserves ordering and produces a uniform scale.
        #
        faiss_norm = {
            r["chunk_id"]: (r["score"] + 1.0) / 2.0
            for r in faiss_results
        }

        # ── Merge all candidate chunk_ids ────────────────────────────
        all_ids = set(bm25_norm.keys()) | set(faiss_norm.keys())

        # Build chunk data lookup (prefer FAISS data since it's richer
        # in semantic matching, but both have the same schema)
        chunk_data = {}
        for r in bm25_results + faiss_results:
            if r["chunk_id"] not in chunk_data:
                chunk_data[r["chunk_id"]] = r

        # ── Compute fused scores ─────────────────────────────────────
        fused = []
        for cid in all_ids:
            bm25_s = bm25_norm.get(cid, 0.0)
            faiss_s = faiss_norm.get(cid, 0.0)
            combined = (self._bm25_weight * bm25_s
                        + self._dense_weight * faiss_s)
            fused.append((cid, combined))

        # Sort by fused score descending
        fused.sort(key=lambda x: x[1], reverse=True)

        # Build results with fused score
        results = []
        for cid, score in fused[:top_k]:
            r = chunk_data[cid].copy()
            r["score"] = float(score)
            results.append(r)

        return results

    @property
    def chunks(self) -> list[dict]:
        """Access the raw chunk list (read-only, from BM25 sub-retriever)."""
        return self._bm25.chunks

    @property
    def n_docs(self) -> int:
        """Number of indexed documents."""
        return self._n_docs

    def __repr__(self) -> str:
        return (
            f"HybridRetriever(n_docs={self._n_docs}, "
            f"bm25_w={self._bm25_weight}, dense_w={self._dense_weight})"
        )
