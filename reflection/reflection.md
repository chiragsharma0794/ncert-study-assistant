# Project Reflection

<!--
MENTOR NOTE FOR THE STUDENT:
A reflection is NOT a summary of what you did.  A reviewer has your code for that.
A reflection is an ANALYSIS: what you learned, what surprised you, what you'd change.
Write it in first person.  Be honest about failures.  Show that you THINK about tradeoffs.

Evaluators look for:
1. Evidence of understanding (not just "I used BM25" but WHY BM25)
2. Honest failure analysis (what went wrong, not just what went right)
3. Clear next steps (shows you know the limits of your current work)
4. Conceptual clarity (can you explain WHY something doesn't fit?)

Personalize the sections below -- replace [BRACKETS] with your own words.
-->

## Design Choices

### Why semantic chunking over fixed-window?

I implemented both strategies and compared them side by side.  Fixed-window chunking (200 tokens, 50-token overlap) produced 89 chunks with uniform sizes, but it cut sentences in half at window boundaries.  When I tested retrieval, chunks that started mid-sentence often ranked poorly because the opening fragment lacked meaningful keywords.

Semantic chunking splits on sentence boundaries and groups sentences until roughly 150 words.  This produced 85 chunks with variable sizes (27-238 tokens), but every chunk contains complete thoughts.  The tradeoff is clear: semantic chunks sacrifice uniformity for meaning.  For a textbook QA system where students ask about specific concepts, complete sentences are more important than uniform sizes.

### Why BM25 first, not a neural retriever?

BM25 is a strong baseline that needs zero training.  When I tested it against TF-IDF on three queries, both agreed on 2/3 top chunks per query, but BM25 produced higher-contrast scores (range 5-14 vs 0.2-0.3), making the ranking signal clearer.  Starting with BM25 let me validate the entire pipeline end-to-end before introducing the complexity of embedding models.

[I found that BM25's term-frequency saturation was the key advantage -- a chunk mentioning "mitochondria" twice wasn't scored 2x higher than one mentioning it once with a clear definition.  This matches how textbook content works: quality matters more than repetition.]

## Failures Encountered

The first version of my corpus preparation had two significant bugs:

1. **Over-aggressive question classifier**: My regex caught "Think It Over" sidebar text that pdfplumber had interleaved into body paragraphs, classifying the entire chapter introduction as a "question" type.  Fix: I added a length + question-mark ratio guard -- for chunks over 600 characters, the classifier requires a meaningful `?`-mark ratio (>= 0.25) before overriding the default "concept" classification.

2. **Orphan fragment after sentence splitting**: `split_oversized()` produced an 83-character tail fragment ("Proteins in the membrane act like gatekeepers...") that was too small to be useful for retrieval.  Fix: I added a `merge_undersized()` post-processing pass that folds fragments under 120 characters into their same-type neighbor.

[Debugging these taught me that PDF extraction errors cascade silently through the pipeline.  A misclassified chunk doesn't crash anything -- it just makes retrieval worse.  This is why corpus quality matters more than model choice.]

## Evaluation Results Interpretation

My retrieval analysis shows 13/14 in-scope queries achieving "GOOD" BM25 scores (> 5.0).  The single weak query -- "What are chromoplasts?" -- scored only 4.36 because the term appears rarely in the corpus.  This is a vocabulary coverage problem, not a retriever problem.

[When I run the full evaluation with Gemini, I expect correctness around ___%, with failures concentrated on paraphrased queries where BM25 can't match synonyms.  The grounding score should be high because the prompt strictly constrains the model.  Refusal accuracy on out-of-scope questions is the most important metric -- if the model hallucates answers about telescopes using cell biology chunks, the grounding prompt has failed.]

## Why RAG is Better Than Plain ChatGPT Here

1. **Verifiability**: My system cites chunk IDs and page numbers.  A student can check `[1] (page 7)` against their physical textbook.  ChatGPT provides no sources -- you have to trust it.

2. **Domain accuracy**: ChatGPT's training data includes college biology, Wikipedia, and thousands of conflicting sources.  It might explain osmosis using university-level terminology.  My system uses the exact NCERT wording the student will see on their exam.

3. **Controlled scope**: When a student asks "Who invented the telescope?" my system says "I could not find this in the textbook."  ChatGPT would answer confidently -- which is *worse* for a student studying for a specific chapter test, because it teaches them to rely on information that won't be on their exam.

## Why GANs Are Not Suitable for This Problem

<!--
MENTOR NOTE: This is a conceptual trap question.  Many students hear "generation"
in "Generative AI" and assume GANs are relevant.  They're not.  The point of this
section is to show you understand WHAT different architectures do.
-->

GANs (Generative Adversarial Networks) consist of a generator that creates synthetic data and a discriminator that distinguishes real from fake.  They excel at generating images, audio, and other continuous data where the goal is to learn and sample from a data distribution.

This project's "generation" is fundamentally different.  I'm not creating new data that resembles a distribution -- I'm *selecting and composing* specific textbook passages into a coherent answer.  The answer must be factually grounded in existing text, not sampled from a learned distribution.

If I used a GAN here, the generator would learn to produce text that "looks like" textbook content -- but looking like a textbook answer and being a *correct* textbook answer are completely different things.  A GAN has no mechanism for verifying factual accuracy.  RAG's architecture -- retrieve first, then generate from retrieved evidence -- is the right fit because it makes grounding an explicit step, not a hopeful side effect.

## What I'd Do Next

1. **Dense retrieval**: Add a sentence-transformer encoder (e.g., `all-MiniLM-L6-v2`) as a second retrieval stage.  BM25 finds keyword matches; dense retrieval finds semantic matches.  Hybrid retrieval (BM25 + dense, merged with reciprocal rank fusion) would fix the "chromoplasts" failure where the student might ask "What gives flowers their colour?" instead.

2. **Inline citations**: Currently the model cites `[1], [2]` which map to chunk IDs.  I'd add page numbers and section headings directly in the answer: "According to Section 2.3 (page 13)..."

3. **Hindi/multilingual support**: NCERT publishes textbooks in Hindi and English.  I'd add Hindi query handling using a multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) so students can ask questions in either language.

[4. **Confidence scoring**: Use the BM25 score gap between rank 1 and rank 2 as a confidence signal.  If the gap is small, the retriever is uncertain -- flag the answer as "low confidence" so the student knows to double-check.]
