# Evaluation Directory

## Final Results

- **`results_v2.json`** — Authoritative V2 evaluation results.
  Live Gemini API answers scored with V1 keyword-overlap + V2 RAGAS-style metrics.
  Combined score: **0.821** (20 questions, real answers, zero dry-run).

- **`results.json`** — V1 baseline (estimated scores, empty `results` array).
  Retained for historical reference only.

## Historical Dry-Run Artifacts

The `*_dryrun.json` files are retriever-comparison artifacts from development.
All contain `[DRY-RUN]` placeholder answers (no generation was performed).
They measure **retrieval quality only** (context_precision, context_recall)
and should **not** be used to report final model quality.

| File | Retriever | Purpose |
|------|-----------|---------|
| `results_bm25_retrieval_dryrun.json` | BM25 | Sparse retrieval baseline |
| `results_faiss_retrieval_dryrun.json` | FAISS | Dense retrieval comparison |
| `results_hybrid_retrieval_dryrun.json` | Hybrid | Score fusion comparison |

## Evaluation Framework

- **`questions.json`** — 20 evaluation questions (8 factual, 6 paraphrased, 6 out-of-scope).
- **`evaluator.py`** — V1 scoring functions (correctness, grounding, refusal).
- **`evaluator_v2.py`** — V2 scoring with RAGAS-style metrics (faithfulness, answer relevancy, context precision, context recall).

## Running Evaluation

```bash
python run_eval_live.py
```

Results are saved to `evaluation/results_v2.json` and cached in `outputs/response_cache.json`.
Subsequent runs replay from cache (zero API calls, identical scores).
