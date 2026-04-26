"""
verify_setup.py  –  Setup Verification Script
=============================================
Run this after installing requirements.txt.
It imports every dependency and prints PASS / FAIL per library.

Usage:
    python verify_setup.py
"""

import importlib
import sys

# ── Map: pip package name  →  actual Python import name ──────────
DEPENDENCIES = {
    # PDF extraction
    "PyMuPDF":              "fitz",
    "pdfplumber":           "pdfplumber",
    # Text processing
    "nltk":                 "nltk",
    "regex":                "regex",
    # BM25 retrieval
    "rank-bm25":            "rank_bm25",
    # Embeddings (stretch)
    "sentence-transformers":"sentence_transformers",
    "faiss-cpu":            "faiss",
    # LLM generation
    "google-generativeai":  "google.generativeai",
    # Evaluation
    "rouge-score":          "rouge_score",
    "bert-score":           "bert_score",
    "ragas":                "ragas",
    # Data science
    "numpy":                "numpy",
    "pandas":               "pandas",
    "scikit-learn":         "sklearn",
    # Notebook & viz
    "jupyter":              "jupyter",
    "matplotlib":           "matplotlib",
    "tqdm":                 "tqdm",
    # Utilities
    "python-dotenv":        "dotenv",
}

WIDTH = 30          # column width for alignment
pass_count = 0
fail_count = 0

print("=" * 55)
print("  NCERT Study Assistant — Dependency Check")
print("=" * 55)
print(f"  Python {sys.version}")
print("-" * 55)

for pkg_name, import_name in DEPENDENCIES.items():
    try:
        importlib.import_module(import_name)
        status = "\033[92mPASS\033[0m"      # green
        pass_count += 1
    except ImportError:
        status = "\033[91mFAIL\033[0m"      # red
        fail_count += 1

    print(f"  {pkg_name:<{WIDTH}} {status}")

print("-" * 55)
print(f"  Results:  {pass_count} passed,  {fail_count} failed")
print("=" * 55)

if fail_count > 0:
    print("\n  ⚠  Some dependencies are missing.")
    print("  Run:  pip install -r requirements.txt")
    sys.exit(1)
else:
    print("\n  ✅  All dependencies installed. You're good to go!")
    sys.exit(0)
