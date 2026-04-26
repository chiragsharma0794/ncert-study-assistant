"""
src/generator.py -- Grounded answer generation for the NCERT RAG pipeline.

Uses Google Gemini to generate answers that are STRICTLY grounded in
retrieved context chunks.  If the context doesn't contain the answer,
the model is instructed to refuse politely.

Usage:
    from generator import GroundedGenerator
    gen = GroundedGenerator(api_key="...", model_name="gemini-2.0-flash")
    result = gen.generate(query, retrieved_chunks)
    print(result["answer"])
"""

from __future__ import annotations

import os
import re
from typing import Optional

from google import genai
from google.genai import types


# =========================================================================
# System prompt — the grounding contract
# =========================================================================
SYSTEM_PROMPT = """\
You are a helpful study assistant for NCERT Class 9 Science.

RULES — follow these STRICTLY:
1. Answer the student's question using ONLY the context provided below.
2. If the context does NOT contain enough information to answer, respond
   EXACTLY with: "I could not find this in the textbook."
3. Do NOT use any outside knowledge, even if you know the answer.
4. Cite your sources using the chunk labels [1], [2], etc.
5. Keep answers clear, concise, and at a Class 9 reading level.
6. If the question is ambiguous, answer the most likely interpretation
   and note the ambiguity.
"""

# =========================================================================
# Prompt template for user turn
# =========================================================================
USER_PROMPT_TEMPLATE = """\
CONTEXT (retrieved from the textbook):
{context_block}

QUESTION: {query}

Answer the question using ONLY the context above. Cite sources as [1], [2], etc.
If the answer is not in the context, say: "I could not find this in the textbook."
"""


# =========================================================================
# GroundedGenerator
# =========================================================================
class GroundedGenerator:
    """Generate grounded answers using Gemini + retrieved context.

    Parameters
    ----------
    api_key : str or None
        Gemini API key.  If None, reads from GEMINI_API_KEY env var.
    model_name : str
        Gemini model to use (e.g. "gemini-2.0-flash", "gemini-1.5-flash").
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "No API key provided.  Set GEMINI_API_KEY in .env or "
                "pass api_key= to GroundedGenerator()."
            )
        self.model_name = model_name
        self._client = genai.Client(api_key=self.api_key)

    # -----------------------------------------------------------------
    # Prompt construction
    # -----------------------------------------------------------------
    def build_prompt(self, query: str, chunks: list[dict]) -> str:
        """Build the full user prompt with labeled context chunks.

        Parameters
        ----------
        query : str
            The student's question.
        chunks : list[dict]
            Retrieved chunks, each with at least "text" and "chunk_id".

        Returns
        -------
        str
            The formatted prompt string.
        """
        context_lines = []
        for i, chunk in enumerate(chunks, 1):
            page_info = f" (page {chunk['page']})" if "page" in chunk else ""
            ctype = f" [{chunk['type']}]" if "type" in chunk else ""
            context_lines.append(
                f"[{i}]{ctype}{page_info}:\n{chunk['text']}"
            )

        context_block = "\n\n".join(context_lines)

        return USER_PROMPT_TEMPLATE.format(
            context_block=context_block,
            query=query,
        )

    # -----------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------
    def generate(self, query: str, chunks: list[dict]) -> dict:
        """Generate a grounded answer from query + retrieved chunks.

        Parameters
        ----------
        query : str
            The student's question.
        chunks : list[dict]
            Retrieved chunks (typically top_k from retriever).

        Returns
        -------
        dict
            {
                "answer":  str,         # the generated text
                "sources": list[str],   # chunk_ids cited in the answer
                "refused": bool,        # True if model refused to answer
                "model":   str,         # model name used
            }
        """
        user_prompt = self.build_prompt(query, chunks)

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=1024,
            ),
        )

        answer_text = response.text.strip() if response.text else ""

        # --- Detect refusal ---
        refused = "could not find" in answer_text.lower()

        # --- Extract cited chunk IDs ---
        # Match [1], [2], etc. in the answer and map back to chunk_ids
        cited_indices = set(int(m) for m in re.findall(r"\[(\d+)\]", answer_text))
        sources = []
        for idx in sorted(cited_indices):
            if 1 <= idx <= len(chunks):
                sources.append(chunks[idx - 1]["chunk_id"])

        return {
            "answer":  answer_text,
            "sources": sources,
            "refused": refused,
            "model":   self.model_name,
        }

    def __repr__(self) -> str:
        return f"GroundedGenerator(model={self.model_name!r})"
