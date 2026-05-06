# NCERT Study Assistant V2 — RAG Pipeline with Hybrid Retrieval & Study Tools

A retrieval-augmented generation (RAG) system that answers Class 9 Science questions using **only** the NCERT textbook as its source of truth.

V2 adds hybrid retrieval (BM25 + FAISS), structured study tools (summaries, explanations, flashcards), response caching, adversarial guardrails, and a Streamlit web UI.

## How It Works

```
Student Question
       │
       ▼
┌─────────────────┐
│ Guardrail Check │──→ Block adversarial queries (2-level defense)
└────────┬────────┘
         ▼
┌─────────────────┐
│ BM25 + FAISS    │──→ Top-5 relevant chunks from the textbook
│ Hybrid Retriever│
└────────┬────────┘
         ▼
┌─────────────────┐
│ Gemini LLM      │──→ Grounded answer with [1],[2] citations
│ (temperature=0) │
└────────┬────────┘
         ▼
  "According to the textbook [1], the cell membrane
   is also called the plasma membrane..."
```

The model **refuses to answer** if the question is outside the textbook's scope:
> "I could not find this in the textbook."

No hallucination. Every answer is traceable to a specific chunk and page.

## Features

| Feature | Description |
|---------|-------------|
| **Q&A** | Grounded question-answering with chunk-level citations |
| **Summarization** | Structured topic summaries (overview + bullet points) |
| **Concept Explanation** | Feynman-style explanations with pedagogical additions flagged |
| **Flashcard Generation** | Anki-compatible cards (definition, fill-blank, true/false) |
| **Hybrid Retrieval** | BM25 (sparse) + FAISS (dense) with weighted score fusion |
| **Response Caching** | SHA-256 keyed, thread-safe, enables offline replay |
| **Adversarial Guardrails** | 2-level defense: 34-phrase blocklist + LLM safety classifier |
| **Streamlit UI** | 4-mode web interface with session tracking |
| **RAGAS-style Evaluation** | Faithfulness, answer relevancy, context precision, context recall |

## Project Structure

```
Ncert_V2/
├── app.py                              # Streamlit web UI (4 modes)
├── run_eval_live.py                    # Live evaluation runner
├── data/
│   └── iesc1*.pdf                      # NCERT Class 9 Science PDFs
├── src/
│   ├── retriever.py                    # BM25, FAISS, Hybrid retrievers
│   ├── generator.py                    # GroundedGenerator (pluggable LLM backend)
│   ├── cache.py                        # ResponseCache + CachedGenerator
│   ├── config.py                       # Retriever factory + env config
│   ├── guardrails.py                   # 2-level adversarial defense
│   ├── summarizer.py                   # GroundedSummarizer
│   ├── explainer.py                    # ConceptExplainer (Feynman-style)
│   ├── flashcard_generator.py          # FlashcardGenerator
│   ├── query_processor.py             # Query classification + routing
│   ├── retrieval_sufficiency.py       # Context adequacy assessment
│   └── corpus_manager.py             # Multi-chapter PDF processing
├── evaluation/
│   ├── questions.json                  # 20 test questions (8 factual, 6 paraphrased, 6 OOS)
│   ├── evaluator.py                    # V1 scoring (correctness, grounding, refusal)
│   ├── evaluator_v2.py                 # V2 scoring (+ RAGAS-style metrics)
│   └── results_v2.json                 # Evaluation output (real Gemini answers)
├── outputs/
│   ├── corpus.json                     # Structured corpus entries
│   ├── chunks_semantic.json            # 85 semantic chunks (source of truth)
│   └── response_cache.json            # Cached API responses
├── notebooks/
│   ├── 01_corpus_preparation.py
│   ├── 02_tokenization_chunking.py
│   ├── 03_retrieval_engine.py
│   ├── 04_generation.py
│   └── 05_evaluation.py
├── tests/                              # Unit + integration tests
├── reflection/
│   └── reflection.md
├── .env.example
├── requirements.txt
└── README.md
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                  │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌────────────┐  │
│  │   Q&A   │ │ Explain  │ │ Summarize │ │ Flashcards │  │
│  └────┬────┘ └────┬─────┘ └─────┬─────┘ └─────┬──────┘  │
│       └───────────┴─────────────┴──────────────┘         │
│                         │                                │
│                    ┌────▼─────┐                           │
│                    │Guardrails│                           │
│                    └────┬─────┘                           │
├─────────────────────────┼────────────────────────────────┤
│  Retrieval Layer        │                                │
│    ┌────────────────────▼─────────────────────┐          │
│    │          HybridRetriever                 │          │
│    │  ┌─────────────┐  ┌──────────────────┐   │          │
│    │  │BM25Retriever│  │ FAISSRetriever   │   │          │
│    │  │ (sparse)    │  │ (dense, semantic) │   │          │
│    │  └─────────────┘  └──────────────────┘   │          │
│    └──────────────────────────────────────────┘          │
├──────────────────────────────────────────────────────────┤
│  Generation Layer                                        │
│    ┌──────────────────────────────────────────┐          │
│    │         GroundedGenerator                │          │
│    │  ┌──────────────┐  ┌─────────────────┐   │          │
│    │  │GeminiBackend │  │ MockLLMBackend  │   │          │
│    │  │ (live API)   │  │ (offline test)  │   │          │
│    │  └──────────────┘  └─────────────────┘   │          │
│    │  ┌──────────────────┐                    │          │
│    │  │ ResponseCache    │ (SHA-256 keyed)     │          │
│    │  └──────────────────┘                    │          │
│    └──────────────────────────────────────────┘          │
├──────────────────────────────────────────────────────────┤
│  Data Layer                                              │
│    chunks_semantic.json (85 chunks, Ch.2: Cell Biology)  │
│    response_cache.json  (deterministic replay)           │
└──────────────────────────────────────────────────────────┘
```

## Setup Instructions (Windows)

### Prerequisites
- Python 3.10+ (Anaconda recommended)
- A Google Gemini API key ([get one free](https://aistudio.google.com/apikey))

### Step 1: Clone and navigate
```bash
git clone https://github.com/YOUR_USERNAME/ncert-rag.git
cd ncert-rag
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

The core dependencies are:

| Package | Purpose |
|---------|---------|
| `pdfplumber` | PDF text extraction |
| `nltk` | Sentence/word tokenization |
| `rank-bm25` | BM25 sparse retrieval |
| `faiss-cpu` | FAISS dense vector index |
| `sentence-transformers` | Cross-encoder reranking |
| `google-genai` | Gemini API client |
| `streamlit` | Web UI framework |
| `filelock` | Atomic cache writes |
| `python-dotenv` | Load API keys from `.env` |

### Step 3: Set up your API key
```bash
copy .env.example .env
```
Edit `.env` and replace `your-api-key-here` with your actual Gemini API key.

> **Note**: The app works without an API key using MockLLM (offline mode). An API key is required for live Gemini answers.

### Step 4: Download NLTK data (one-time)
```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

### Step 5: Place NCERT PDFs
Ensure NCERT Class 9 Science PDFs are in the `data/` directory (named `iesc102.pdf` for Chapter 2, etc.).

## Running the App

```bash
streamlit run app.py
```

This launches the Streamlit UI with 4 study modes:
- **Q&A**: Ask questions, get grounded answers with citations
- **Explain**: Get Feynman-style concept explanations
- **Summarize**: Get structured topic summaries
- **Flashcards**: Generate study flashcards

The app works with or without a Gemini API key (falls back to MockLLM offline mode).

## Running the Pipeline (Notebooks)

Run each notebook in order to rebuild the pipeline from scratch:

```bash
# Phase 1: Extract and clean text from the PDF
python notebooks/01_corpus_preparation.py

# Phase 2: Compare tokenizers and build semantic chunks
python notebooks/02_tokenization_chunking.py

# Phase 3: Build and compare BM25 vs TF-IDF retrievers
python notebooks/03_retrieval_engine.py

# Phase 4: Wire up Gemini for grounded answer generation
python notebooks/04_generation.py

# Phase 5: Run the 20-question evaluation
python notebooks/05_evaluation.py
```

> **Note**: Phases 1-3 run without an API key. Phases 4-5 require a Gemini API key.

## Running Evaluation

```bash
python run_eval_live.py
```

This runs all 20 evaluation questions through the live Gemini API with:
- 3-second rate limiting between API calls
- Automatic response caching (subsequent runs are instant)
- V1 + V2 metrics scoring
- Built-in validation checklist

Results are saved to `evaluation/results_v2.json`.

## Evaluation Results

Evaluated on 20 questions: 8 factual, 6 paraphrased, 6 out-of-scope.

### V1 Metrics (keyword-overlap)

| Metric | Score |
|--------|-------|
| Factual correctness (in-scope) | 85.7% (12/14) |
| Grounding accuracy (in-scope) | 100.0% (14/14) |
| Out-of-scope refusal accuracy | 90.0% (18/20) |

### V2 Metrics (RAGAS-style semantic)

| Metric | Score |
|--------|-------|
| Faithfulness | 1.000 |
| Answer Relevancy | 0.628 |
| Context Precision | 0.800 |
| Context Recall | 0.780 |
| **Combined Score** | **0.821** |

### Breakdown by Question Type

| Type | N | Correctness | Grounding | Refusal | Faithfulness | Ans. Relevancy | Ctx. Precision | Ctx. Recall |
|------|---|-------------|-----------|---------|-------------|----------------|----------------|-------------|
| Factual | 8 | 75.0% | 100.0% | 75.0% | 1.000 | 0.597 | 0.725 | 0.755 |
| Paraphrased | 6 | 100.0% | 100.0% | 100.0% | 1.000 | 0.669 | 0.900 | 0.814 |
| Out-of-scope | 6 | — | — | 100.0% | — | — | — | — |

## V1 vs V2 Comparison

| Aspect | V1 | V2 |
|--------|----|----|
| **Retriever** | BM25 only | BM25 + FAISS + Hybrid (weighted score fusion) |
| **Generation** | Single Gemini backend | Pluggable LLMBackend protocol (Gemini + MockLLM) |
| **Caching** | None | SHA-256 keyed ResponseCache with atomic writes |
| **Evaluation** | Keyword-overlap only (3 metrics) | + RAGAS-style semantic metrics (7 total) |
| **Study modes** | Q&A only | Q&A + Summarize + Explain + Flashcards |
| **UI** | None (notebook-only) | Streamlit web app (4 modes) |
| **Guardrails** | None | 2-level adversarial defense (pattern + LLM) |
| **Offline mode** | Not possible | Full offline via MockLLM + cached responses |
| **Correctness** | 78.6% (estimated) | 85.7% (measured, real Gemini answers) |
| **Grounding** | 92.9% (estimated) | 100.0% (measured) |
| **Refusal** | 83.3% (estimated) | 90.0% (measured) |

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **Semantic chunking** over fixed-window | Preserves sentence boundaries; better retrieval precision |
| **BM25** as primary retriever | No training needed; strong baseline for keyword-heavy factual queries |
| **Hybrid retrieval** (V2) | Weighted BM25 + FAISS score fusion catches semantic matches that BM25 misses |
| **Temperature = 0** | Deterministic, reproducible answers for evaluation |
| **Strict grounding prompt** | Forces the LLM to cite sources and refuse out-of-scope questions |
| **Manual RAGAS metrics** | Zero dependency on RAGAS library; works in offline/mock mode |
| **Pluggable LLMBackend** | Enables MockLLM for offline testing; future backend swaps |
| **Filelock + atomic writes** | Prevents cache corruption from concurrent Streamlit reruns |

## Known Limitations

1. **Single-chapter corpus**: Currently processes only Chapter 2 (Cell Biology). Multi-chapter corpus manager exists (`corpus_manager.py`) but processing is not yet run on all 14 chapters.
2. **BM25 vocabulary gap**: Rare terms (e.g., "chromoplasts") score low because BM25 relies on term frequency. Hybrid retrieval mitigates but does not fully solve this.
3. **No Hindi support**: Queries must be in English. A multilingual embedding model would enable Hindi/Hinglish queries.
4. **Personalization is cosmetic**: `UserSession` tracks query history and adapts difficulty labels, but does not yet modulate retrieval depth or prompt complexity.
5. **Gemini free-tier quota**: The free tier has daily token limits. The caching layer ensures evaluation is reproducible even during quota exhaustion.

## License

This project is for educational purposes.
NCERT textbook content is copyrighted by NCERT, India.
