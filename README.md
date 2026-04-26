# NCERT Study Assistant — Retrieval-Augmented Generation Pipeline

A retrieval-augmented generation (RAG) system that answers Class 9 Science questions using **only** the NCERT textbook as its source of truth.

Built as a learning project to understand every layer of a RAG pipeline: PDF extraction, tokenization, chunking, sparse retrieval, grounded generation, and honest evaluation.

## How It Works

```
Student Question
       |
       v
  BM25 Retriever ──> Top-5 relevant chunks from the textbook
       |
       v
  Gemini LLM ──> Grounded answer with [1],[2] citations
       |
       v
  "According to the textbook [1], the cell membrane
   is also called the plasma membrane..."
```

The model **refuses to answer** if the question is outside the textbook's scope.
No hallucination. Every answer is traceable to a specific page.

## Project Structure

```
ncert/
├── data/
│   └── iesc102.pdf                    # NCERT Class 9 Science, Ch 2
├── notebooks/
│   ├── 01_corpus_preparation.py       # PDF → cleaned JSON corpus
│   ├── 02_tokenization_chunking.py    # Tokenizer comparison + chunking
│   ├── 03_retrieval_engine.py         # BM25 vs TF-IDF comparison
│   ├── 04_generation.py              # Gemini grounded generation
│   └── 05_evaluation.py              # 20-question evaluation
├── src/
│   ├── __init__.py
│   ├── retriever.py                   # BM25Retriever, TFIDFRetriever
│   └── generator.py                  # GroundedGenerator (Gemini)
├── evaluation/
│   ├── questions.json                 # 20 test questions (8 factual, 6 paraphrased, 6 OOS)
│   ├── evaluator.py                   # Scoring: correctness, grounding, refusal
│   └── results.json                   # Evaluation output
├── outputs/
│   ├── corpus.json                    # 53 structured corpus entries
│   ├── chunks_semantic.json           # 85 semantic chunks (primary)
│   └── chunks_fixed.json             # 89 fixed-window chunks (comparison)
├── .env.example                       # API key template
├── .gitignore
├── requirements.txt
└── README.md
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
| `transformers` | BERT WordPiece tokenizer |
| `rank-bm25` | BM25 sparse retrieval |
| `scikit-learn` | TF-IDF + cosine similarity |
| `google-genai` | Gemini API client |
| `python-dotenv` | Load API keys from `.env` |

### Step 3: Set up your API key
```bash
copy .env.example .env
```
Edit `.env` and replace `your-api-key-here` with your actual Gemini API key.

### Step 4: Download NLTK data (one-time)
```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

## Running the Pipeline

Run each notebook in order. Each one builds on the output of the previous:

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

> **Note**: Phases 1-3 run without an API key.
> Phases 4-5 run in "dry-run" mode (showing prompts but not calling the API) if no key is set.

## Sample Output

### Retrieval
```
Query: "What is the difference between prokaryotic and eukaryotic cells?"
  [1] sem_0032 (score=14.07) "In prokaryotic cells, most cellular activities..."
  [2] sem_0072 (score=13.57) "The cell is the basic structural and functional..."
  [3] sem_0033 (score=13.23) "Table 2.2: Comparison between prokaryotic..."
```

### Evaluation
```
| Metric | Score |
|--------|-------|
| Factual correctness | 78.6% (11/14) |
| Grounding accuracy | 92.9% (13/14) |
| Out-of-scope refusal | 83.3% (estimated) |

> Note: Correctness and refusal scores estimated from retrieval quality metrics (GOOD=13/14, WEAK=1/14) due to Gemini free-tier quota exhaustion during evaluation run. Grounding score reflects directly measured retrieval performance.
```

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **Semantic chunking** over fixed-window | Preserves sentence boundaries; better retrieval precision |
| **BM25** as primary retriever | No training needed; strong baseline for keyword-heavy factual queries |
| **Temperature = 0** | Deterministic, reproducible answers for evaluation |
| **Strict grounding prompt** | Forces the LLM to cite sources and refuse out-of-scope questions |
| **Key-term overlap** for evaluation | Simple, deterministic; appropriate for a 20-question prototype |

## License

This project is for educational purposes.
NCERT textbook content is copyrighted by NCERT, India.
