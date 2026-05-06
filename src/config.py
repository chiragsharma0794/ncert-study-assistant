"""
src/config.py -- Central configuration for the NCERT RAG pipeline.

Provides RETRIEVER_TYPE selection and a factory function to instantiate
the configured retriever without changing calling code.

DESIGN: ENVIRONMENT VARIABLE OVERRIDE
--------------------------------------
RETRIEVER_TYPE can be overridden via environment variable:
    RETRIEVER_TYPE=hybrid python notebooks/05_evaluation.py

This enables A/B testing without editing config.py.

DESIGN: GRACEFUL FALLBACK
--------------------------
If the requested retriever cannot be instantiated (e.g., FAISS index
missing, sentence-transformers not installed), the factory falls back
to BM25Retriever with a warning.  The system NEVER crashes on retriever
selection.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional, Union


# =========================================================================
# Configuration values
# =========================================================================

# Retriever type: "bm25" | "hybrid" | "faiss"
# Override via environment variable: RETRIEVER_TYPE=hybrid
RETRIEVER_TYPE = os.getenv("RETRIEVER_TYPE", "bm25")

# Shared paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "outputs" / "chunks_semantic.json"
FAISS_INDEX_PATH = PROJECT_ROOT / "outputs" / "faiss_index.bin"

# Retriever parameters
HYBRID_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HYBRID_RECALL_FACTOR = 3
FAISS_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5


# =========================================================================
# Factory function
# =========================================================================

def create_retriever(
    chunks: Optional[list[dict]] = None,
    retriever_type: Optional[str] = None,
    chunks_path: Optional[Union[str, Path]] = None,
    faiss_index_path: Optional[Union[str, Path]] = None,
):
    """Factory: instantiate the configured retriever.

    Parameters
    ----------
    chunks : list[dict], optional
        Pre-loaded chunks.  Only used when calling code already has them
        loaded.  The V2 retrievers load their own chunks from disk, so
        this parameter is primarily for backward compatibility.
    retriever_type : str, optional
        Override RETRIEVER_TYPE.  One of: "bm25", "hybrid", "faiss".
        If not specified, uses the module-level RETRIEVER_TYPE
        (which can be set via env var).
    chunks_path : str or Path, optional
        Path to chunks_semantic.json.  Default: outputs/chunks_semantic.json.
    faiss_index_path : str or Path, optional
        Path to FAISS index file.  Default: outputs/faiss_index.bin.

    Returns
    -------
    object
        A retriever with .retrieve(query, top_k) method.

    GRACEFUL FALLBACK:
        If the requested retriever cannot be created (missing dependencies,
        missing index file, etc.), falls back to BM25Retriever with a warning.
        The system NEVER crashes on retriever selection.

    Examples
    --------
    >>> r = create_retriever()                                    # uses RETRIEVER_TYPE default
    >>> r = create_retriever(retriever_type="hybrid")             # explicit override
    >>> r = create_retriever(retriever_type="faiss")              # FAISS loads its own chunks
    """
    rtype = (retriever_type or RETRIEVER_TYPE).lower().strip()
    cpath = str(Path(chunks_path) if chunks_path else CHUNKS_PATH)
    ipath = str(Path(faiss_index_path) if faiss_index_path else FAISS_INDEX_PATH)

    # ── BM25 (always available, the safe fallback) ───────────────────
    if rtype == "bm25":
        from retriever import BM25Retriever
        return BM25Retriever(chunks_path=cpath)

    # ── Hybrid (BM25 + FAISS score fusion) ───────────────────────────
    if rtype == "hybrid":
        try:
            from retriever import HybridRetriever
            return HybridRetriever(
                chunks_path=cpath,
                model_name=FAISS_MODEL,
            )
        except Exception as e:
            warnings.warn(
                f"Failed to create HybridRetriever ({e}). "
                "Falling back to BM25Retriever.",
                RuntimeWarning,
                stacklevel=2,
            )
            from retriever import BM25Retriever
            return BM25Retriever(chunks_path=cpath)

    # ── FAISS (dense retrieval) ──────────────────────────────────────
    if rtype == "faiss":
        try:
            from retriever import FAISSRetriever
            return FAISSRetriever(
                chunks_path=cpath,
                index_path=ipath,
                model_name=FAISS_MODEL,
            )
        except ImportError as e:
            warnings.warn(
                f"FAISSRetriever dependencies not available ({e}). "
                "Falling back to BM25Retriever.",
                RuntimeWarning,
                stacklevel=2,
            )
            from retriever import BM25Retriever
            return BM25Retriever(chunks_path=cpath)
        except Exception as e:
            warnings.warn(
                f"Failed to create FAISSRetriever ({e}). "
                "Falling back to BM25Retriever.",
                RuntimeWarning,
                stacklevel=2,
            )
            from retriever import BM25Retriever
            return BM25Retriever(chunks_path=cpath)

    # ── Unknown type ─────────────────────────────────────────────────
    warnings.warn(
        f"Unknown RETRIEVER_TYPE='{rtype}'. "
        f"Valid options: 'bm25', 'hybrid', 'faiss'. "
        "Falling back to BM25Retriever.",
        RuntimeWarning,
        stacklevel=2,
    )
    from retriever import BM25Retriever
    return BM25Retriever(chunks_path=cpath)
