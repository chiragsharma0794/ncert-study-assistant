"""
app.py -- Streamlit UI for the NCERT Study Assistant V2.

Wires all src/ modules into a student-facing web application.
Works with or without GEMINI_API_KEY (falls back to MockLLMGenerator).

Run:  streamlit run app.py

STATE MANAGEMENT DIAGRAM:
    +--------------------------------------------------+
    |  st.session_state                                |
    |                                                  |
    |  api_key          : str (masked input)           |
    |  mode             : str (qa/explain/summarize/   |
    |                          flashcard)              |
    |  chapter          : str ("All" or "1"-"14")      |
    |  last_query       : str (prevents re-runs)       |
    |  last_mode        : str (prevents re-runs)       |
    |  last_result      : dict (cached response)       |
    |  cache            : ResponseCache instance       |
    |  session          : UserSession instance          |
    |  retriever        : HybridRetriever (cached)     |
    |  chunks           : list[dict] (cached)          |
    +--------------------------------------------------+
            |
            v
    [User types query + clicks Submit]
            |
            v
    [Check: query == last_query AND mode == last_mode?]
       YES -> display last_result (no re-run)
       NO  -> run pipeline -> store in last_result

DESIGN DECISIONS:
  1. STREAMING: We do NOT use st.write_stream.  Our prompts require
     structured JSON output (summaries, flashcards, explanations).
     Streaming would deliver partial JSON that can't be parsed mid-stream.
     Instead, we use st.spinner for loading feedback.

  2. RE-RUN PREVENTION: Streamlit reruns the entire script on every
     interaction.  We prevent redundant retrieval/generation by caching
     (last_query, last_mode, last_result) in st.session_state.  The
     pipeline only re-runs when query or mode actually changes.

  3. REFUSAL DISPLAY: Refusals are shown in st.warning (yellow) with
     a distinct icon, versus normal answers in st.success (green) or
     st.info (blue).  This makes grounding failures immediately visible.
"""

import sys
from pathlib import Path

# ── Ensure src/ is importable ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st

from retriever import BM25Retriever, HybridRetriever, load_chunks
from cache import ResponseCache, CachedGenerator, MockLLMGenerator
from query_processor import QueryProcessor, UserSession

CHUNKS_SEMANTIC = PROJECT_ROOT / "outputs" / "chunks_semantic.json"
CHUNKS_ALL = PROJECT_ROOT / "outputs" / "chunks_all.json"
CACHE_PATH = PROJECT_ROOT / "outputs" / "response_cache.json"


# =========================================================================
# Page config
# =========================================================================
st.set_page_config(
    page_title="NCERT Study Assistant V2",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# Custom CSS
# =========================================================================
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
    }

    /* Flashcard styling */
    .flashcard-front {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }

    .flashcard-back {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        font-size: 1.0rem;
    }

    /* Source badge */
    .source-badge {
        display: inline-block;
        background: #e8eaf6;
        color: #3949ab;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin: 2px;
        font-family: monospace;
    }

    /* Mode indicator */
    .mode-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Header gradient */
    .header-gradient {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# Session state initialization
# =========================================================================
def init_session_state():
    """Initialize all session state variables (runs once)."""
    defaults = {
        "api_key": "",
        "mode": "Q&A",
        "chapter": "All",
        "last_query": "",
        "last_mode": "",
        "last_result": None,
        "cache": ResponseCache(CACHE_PATH),
        "session": UserSession(),
        "retriever": None,
        "chunks": None,
        "generator": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# =========================================================================
# Helper: load retriever (cached in session state)
# =========================================================================
@st.cache_resource
def load_retriever():
    """Load chunks and build retriever (cached across reruns)."""
    chunks_path = CHUNKS_ALL if CHUNKS_ALL.exists() else CHUNKS_SEMANTIC
    chunks = load_chunks(chunks_path)
    chunks_path_str = str(chunks_path)
    try:
        retriever = HybridRetriever(chunks_path=chunks_path_str)
    except Exception:
        retriever = BM25Retriever(chunks_path=chunks_path_str)
    return chunks, retriever


# =========================================================================
# Helper: get generator
# =========================================================================
def get_generator(api_key: str):
    """Return a CachedGenerator (live or mock)."""
    cache = st.session_state["cache"]
    if api_key:
        try:
            from generator import GroundedGenerator
            gen = GroundedGenerator(api_key=api_key)
            return CachedGenerator(gen, cache, offline=False)
        except Exception:
            pass
    # Fallback to mock
    return CachedGenerator(MockLLMGenerator(), cache, offline=False)


# =========================================================================
# SIDEBAR
# =========================================================================
with st.sidebar:
    st.markdown('<p class="header-gradient">NCERT Study Assistant</p>',
                unsafe_allow_html=True)
    st.caption("V2 -- AI-Powered Study Tool")

    st.divider()

    # ── Mode selector ────────────────────────────────────────────────
    mode = st.selectbox(
        "📋 Study Mode",
        ["Q&A", "Explain", "Summarize", "Flashcards"],
        key="mode",
        help="Choose how you want to interact with the textbook.",
    )

    # ── Chapter selector ─────────────────────────────────────────────
    chapter_options = ["All"] + [f"Ch {i}" for i in range(1, 15)]
    st.selectbox(
        "📖 Chapter",
        chapter_options,
        key="chapter",
        help="Scope retrieval to a specific chapter or search all.",
    )

    st.divider()

    # ── API key input ────────────────────────────────────────────────
    api_key = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        value=st.session_state.get("api_key", ""),
        help="Optional. Leave blank to use MockLLM (offline mode).",
        placeholder="AIza...",
    )
    st.session_state["api_key"] = api_key

    if api_key:
        st.success("API key set", icon="✅")
    else:
        st.info("Using MockLLM (no API key)", icon="🔄")

    st.divider()

    # ── Cache stats ──────────────────────────────────────────────────
    cache_stats = st.session_state["cache"].stats()
    st.markdown("**📊 Cache Status**")
    col1, col2 = st.columns(2)
    col1.metric("Hits", cache_stats.get("hit_count", cache_stats.get("hits", 0)))
    col2.metric("Misses", cache_stats.get("miss_count", cache_stats.get("misses", 0)))
    if cache_stats["total_entries"] > 0:
        size_bytes = cache_stats.get("cache_file_size_bytes", cache_stats.get("size_bytes", 0))
        st.caption(f"{cache_stats['total_entries']} cached responses "
                   f"({size_bytes / 1024:.1f} KB)")

    # ── Session stats ────────────────────────────────────────────────
    session_stats = st.session_state["session"].stats()
    if session_stats["total_queries"] > 0:
        st.divider()
        st.markdown("**🎓 Session**")
        st.caption(f"Queries: {session_stats['total_queries']} | "
                   f"Difficulty: {session_stats['difficulty']}")


# =========================================================================
# MAIN PANEL
# =========================================================================

# ── Header ───────────────────────────────────────────────────────────────
st.markdown("## 📚 NCERT Study Assistant")

# Mode-specific placeholder text
PLACEHOLDERS = {
    "Q&A": "Ask a question... e.g., 'What is a cell membrane?'",
    "Explain": "Enter a concept... e.g., 'osmosis', 'cell wall vs cell membrane'",
    "Summarize": "Enter a topic... e.g., 'cell organelles', 'mitochondria'",
    "Flashcards": "Enter a topic... e.g., 'cell membrane', 'prokaryotic cells'",
}

mode_icons = {"Q&A": "❓", "Explain": "💡", "Summarize": "📝", "Flashcards": "🃏"}

st.markdown(f"{mode_icons.get(mode, '')} **Mode: {mode}**")

# ── Query input ──────────────────────────────────────────────────────────
query = st.text_input(
    "Your question or topic:",
    placeholder=PLACEHOLDERS.get(mode, "Type here..."),
    label_visibility="collapsed",
)

submit = st.button("🚀 Submit", type="primary", use_container_width=True)


# =========================================================================
# Pipeline execution
# =========================================================================
def run_qa(query, retriever, generator):
    """Q&A mode: retrieve + generate answer."""
    retrieved = retriever.retrieve(query, top_k=5)
    response = generator.generate(query, retrieved)
    return {"type": "qa", "response": response, "chunks": retrieved}


def run_explain(query, retriever, api_key):
    """Explain mode: use ConceptExplainer."""
    try:
        from explainer import ConceptExplainer
        explainer = ConceptExplainer(retriever=retriever, api_key=api_key or None)
        result = explainer.explain(query)
        return {"type": "explain", "response": result.to_dict(), "refused": result.refused}
    except Exception as e:
        return {"type": "error", "message": str(e)}


def run_summarize(query, retriever, api_key):
    """Summarize mode: use GroundedSummarizer."""
    try:
        from summarizer import GroundedSummarizer
        summarizer = GroundedSummarizer(retriever=retriever, api_key=api_key or None)
        result = summarizer.summarize(query)
        return {"type": "summarize", "response": result.to_dict(), "refused": result.refused}
    except Exception as e:
        return {"type": "error", "message": str(e)}


def run_flashcards(query, retriever, api_key):
    """Flashcard mode: use FlashcardGenerator."""
    try:
        from flashcard_generator import FlashcardGenerator
        gen = FlashcardGenerator(retriever=retriever, api_key=api_key or None)
        result = gen.generate(query)
        return {"type": "flashcards", "response": result.to_dict(), "refused": result.refused}
    except Exception as e:
        return {"type": "error", "message": str(e)}


# =========================================================================
# Execute on submit (with rerun prevention)
# =========================================================================
if submit and query.strip():
    # Check if this is a new query or mode change
    is_new = (query != st.session_state["last_query"]
              or mode != st.session_state["last_mode"])

    if is_new:
        chunks, retriever = load_retriever()
        generator = get_generator(api_key)

        with st.spinner(f"{'Thinking' if mode == 'Q&A' else 'Generating'}..."):
            if mode == "Q&A":
                result = run_qa(query, retriever, generator)
            elif mode == "Explain":
                result = run_explain(query, retriever, api_key)
            elif mode == "Summarize":
                result = run_summarize(query, retriever, api_key)
            elif mode == "Flashcards":
                result = run_flashcards(query, retriever, api_key)

        # Record in session
        st.session_state["session"].record_query(query, mode.lower(), concept=query)
        st.session_state["last_query"] = query
        st.session_state["last_mode"] = mode
        st.session_state["last_result"] = result


# =========================================================================
# Display results
# =========================================================================
result = st.session_state.get("last_result")

if result:
    st.divider()

    # ── Error handling ───────────────────────────────────────────────
    if result.get("type") == "error":
        st.error(f"Error: {result['message']}", icon="🚨")

    # ── Q&A Display ──────────────────────────────────────────────────
    elif result["type"] == "qa":
        resp = result["response"]
        chunks_used = result.get("chunks", [])

        if resp.get("refused"):
            st.warning(resp["answer"], icon="⚠️")
        else:
            st.success(resp["answer"], icon="✅")

        # Source citations
        if resp.get("sources"):
            st.caption(f"Sources: {', '.join(resp['sources'])}")

        # Expandable chunk viewer
        if chunks_used:
            with st.expander(f"📄 Retrieved Chunks ({len(chunks_used)})"):
                for i, chunk in enumerate(chunks_used):
                    score = chunk.get("score", 0)
                    st.markdown(
                        f"**`{chunk['chunk_id']}`** "
                        f"(page {chunk.get('page', '?')}, "
                        f"type: {chunk.get('type', '?')}, "
                        f"score: {score:.3f})"
                    )
                    st.text((chunk.get("content") or chunk.get("text", ""))[:500])
                    if i < len(chunks_used) - 1:
                        st.divider()

    # ── Explain Display ──────────────────────────────────────────────
    elif result["type"] == "explain":
        resp = result["response"]

        if result.get("refused"):
            st.warning(resp.get("simple_definition", "No information found."), icon="⚠️")
        else:
            # Simple definition
            st.markdown("### 📖 Simple Definition")
            st.info(resp.get("simple_definition", ""))

            # Analogy
            analogy = resp.get("analogy", "")
            if analogy:
                st.markdown("### 🔗 Analogy")
                if not resp.get("analogy_is_grounded", True):
                    st.caption("⚡ Pedagogical addition (not from NCERT)")
                st.markdown(analogy)

            # Steps
            steps = resp.get("steps", [])
            if steps:
                st.markdown("### 📋 Step-by-Step")
                for step in steps:
                    st.markdown(f"- {step}")

            # Misconception
            misconception = resp.get("misconception", "")
            if misconception:
                st.markdown("### ⚠️ Common Misconception")
                st.warning(misconception)

            # Related concepts
            related = resp.get("related_concepts", [])
            if related:
                st.markdown("### 🔄 Related Concepts")
                st.markdown(", ".join(f"`{r}`" for r in related))

            # Citations
            chunk_ids = resp.get("chunk_ids", [])
            if chunk_ids:
                with st.expander(f"📄 Sources ({len(chunk_ids)} chunks)"):
                    for cid in chunk_ids:
                        st.code(cid, language=None)

    # ── Summarize Display ────────────────────────────────────────────
    elif result["type"] == "summarize":
        resp = result["response"]

        if result.get("refused"):
            st.warning(resp.get("overview", "No information found."), icon="⚠️")
        else:
            # Overview
            st.markdown("### 📝 Overview")
            st.markdown(resp.get("overview", ""))

            # Bullets
            bullets = resp.get("bullets", [])
            if bullets:
                st.markdown("### 🔑 Key Concepts")
                for b in bullets:
                    st.markdown(f"- {b}")

            # Missing
            missing = resp.get("missing_topics", [])
            if missing:
                st.markdown("### ❓ Not Covered")
                for m in missing:
                    st.caption(f"- {m}")

            # Citations
            chunk_ids = resp.get("chunk_ids", [])
            if chunk_ids:
                with st.expander(f"📄 Sources ({len(chunk_ids)} chunks)"):
                    for cid in chunk_ids:
                        st.code(cid, language=None)

    # ── Flashcards Display ───────────────────────────────────────────
    elif result["type"] == "flashcards":
        resp = result["response"]

        if result.get("refused"):
            st.warning("Could not generate flashcards for this topic.", icon="⚠️")
        else:
            cards = resp.get("flashcards", [])
            if cards:
                st.markdown(f"### 🃏 {len(cards)} Flashcards Generated")

                for i, card in enumerate(cards):
                    with st.expander(
                        f"Card {i + 1}: {card.get('front', '')[:60]}...",
                        expanded=(i == 0),
                    ):
                        # Front (question)
                        st.markdown(
                            f'<div class="flashcard-front">'
                            f'<strong>Q:</strong> {card.get("front", "")}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        # Back (answer) -- revealed on click
                        st.markdown(
                            f'<div class="flashcard-back">'
                            f'<strong>A:</strong> {card.get("back", "")}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        # Metadata
                        col1, col2, col3 = st.columns(3)
                        col1.caption(f"Type: {card.get('type', '?')}")
                        col2.caption(f"Difficulty: {card.get('difficulty', '?')}")
                        col3.caption(f"Source: `{card.get('source_chunk_id', '?')}`")
            else:
                st.info("No flashcards could be generated from the available context.")

            # Topics covered/missing
            covered = resp.get("topics_covered", [])
            missing = resp.get("topics_missing", [])
            if covered:
                st.caption(f"Topics covered: {', '.join(covered)}")
            if missing:
                st.caption(f"Not covered: {', '.join(missing)}")

# ── Empty state ──────────────────────────────────────────────────────
elif not result:
    st.markdown("---")
    st.markdown(
        "👋 **Welcome!** Choose a mode from the sidebar and type your "
        "question or topic above to get started."
    )

    with st.expander("ℹ️ What can I do?"):
        st.markdown("""
        | Mode | What it does | Example |
        |------|-------------|---------|
        | **Q&A** | Answer specific questions | "What is a cell membrane?" |
        | **Explain** | Step-by-step concept explanation | "osmosis" |
        | **Summarize** | Topic overview with key points | "cell organelles" |
        | **Flashcards** | Generate study cards | "prokaryotic cells" |

        **Features:**
        - 🔍 Hybrid retrieval (BM25 + cross-encoder)
        - 📄 Every answer cites source chunks
        - 🌐 Works offline after first run (cached responses)
        - 💾 Response caching (works offline after first run)
        - 🛡️ Adversarial query detection
        """)
