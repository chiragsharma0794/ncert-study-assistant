"""
src/retriever.py -- Sparse retrieval engines for the NCERT RAG pipeline.

Two retrievers with an identical interface:
  - BM25Retriever   : Okapi BM25 via rank_bm25
  - TFIDFRetriever  : TF-IDF + cosine similarity via scikit-learn

Both accept a list of chunk dicts and expose:
  .retrieve(query, top_k)  ->  list[dict]  (chunks with scores)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Union

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Shared tokenizer for both retrievers
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization (shared by both retrievers).

    We use a simple regex tokenizer rather than NLTK/BERT because
    BM25 and TF-IDF operate on *word-level* tokens, not subwords.
    Lowercasing ensures case-insensitive matching.
    """
    return _WORD_RE.findall(text.lower())


# =========================================================================
# BM25 Retriever
# =========================================================================
class BM25Retriever:
    """Okapi BM25 sparse retriever.

    Parameters
    ----------
    chunks : list[dict]
        Each dict must have at least a ``text`` key.
        Additional keys (chunk_id, page, type, ...) are preserved in results.

    Example
    -------
    >>> retriever = BM25Retriever(chunks)
    >>> results = retriever.retrieve("What is osmosis?", top_k=3)
    >>> results[0]["score"]
    4.237
    """

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self._corpus_tokens = [_tokenize(c["text"]) for c in chunks]
        self._index = BM25Okapi(self._corpus_tokens)
        self._n_docs = len(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top_k most relevant chunks for *query*.

        Each returned dict is a copy of the original chunk dict
        with an added ``score`` key (float, higher = more relevant).
        """
        query_tokens = _tokenize(query)
        scores = self._index.get_scores(query_tokens)

        # Rank by score descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            entry = dict(self.chunks[idx])      # shallow copy
            entry["score"] = float(scores[idx])
            results.append(entry)

        return results

    def save_index(self, path: Union[str, Path]) -> None:
        """Save chunk metadata to JSON (BM25 index is rebuilt on load)."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

    def __repr__(self) -> str:
        return f"BM25Retriever(n_docs={self._n_docs})"


# =========================================================================
# TF-IDF Retriever
# =========================================================================
class TFIDFRetriever:
    """TF-IDF + cosine similarity sparse retriever.

    Parameters
    ----------
    chunks : list[dict]
        Each dict must have at least a ``text`` key.

    Example
    -------
    >>> retriever = TFIDFRetriever(chunks)
    >>> results = retriever.retrieve("What is a cell wall?", top_k=3)
    """

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self._vectorizer = TfidfVectorizer(
            tokenizer=_tokenize,
            token_pattern=None,      # use our custom tokenizer
            lowercase=False,         # _tokenize already lowercases
        )
        texts = [c["text"] for c in chunks]
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)
        self._n_docs = len(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top_k most relevant chunks for *query*.

        Each returned dict is a copy of the original chunk dict
        with an added ``score`` key (cosine similarity, 0..1).
        """
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            entry = dict(self.chunks[idx])
            entry["score"] = float(scores[idx])
            results.append(entry)

        return results

    def save_index(self, path: Union[str, Path]) -> None:
        """Save chunk metadata to JSON."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

    def __repr__(self) -> str:
        return f"TFIDFRetriever(n_docs={self._n_docs})"


# =========================================================================
# Utility: load chunks
# =========================================================================
def load_chunks(path: Union[str, Path]) -> list[dict]:
    """Load a chunk JSON file and return the list of chunk dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
