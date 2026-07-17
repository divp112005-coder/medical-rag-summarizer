# =============================================================================
# main.py  —  Phase 2 (Complete): FastAPI RAG Orchestration + LLM Integration
# =============================================================================
# What this file does (plain English):
#
#   1. STARTUP  — Loads the ChromaDB vector store and embedding model into
#                 memory ONCE at boot so every request is fast.
#
#   2. GET /    — Health-check: confirms the server is alive and shows how
#                 many chunks are indexed.
#
#   3. POST /api/query  — Full RAG pipeline:
#       a) Retrieve top-K semantically matching text chunks from ChromaDB.
#       b) Build a structured prompt with strict citation and refusal rules.
#       c) Call the HuggingFace Inference API (zephyr-7b-beta) for the answer.
#       d) Return the LLM answer + retrieved chunks + mandatory disclaimer.
#
# How to run:
#   uvicorn main:app --reload
#
# Interactive Swagger docs:
#   http://127.0.0.1:8000/docs
# =============================================================================

import os                                    # To read environment variables (API key)
import requests                              # For calling the HuggingFace Inference API
from contextlib import asynccontextmanager   # Modern FastAPI lifespan hook
from typing import Any                       # Generic type hint for flexible dicts

from fastapi import FastAPI, HTTPException   # FastAPI core + structured error responses
from pydantic import BaseModel, Field        # Request/response validation

# Phase 1 modules
from vector_store import init_vector_store, query_vector_store


# =============================================================================
# CONSTANTS
# =============================================================================

# The HuggingFace Inference API endpoint for zephyr-7b-beta.
# Zephyr is a fine-tuned, instruction-following model based on Mistral-7B.
# It follows system prompts reliably, making it ideal for our citation and
# refusal rules.  It is free to use via the HuggingFace Inference API
# (rate-limited; no credit card required).
HF_MODEL_ID   = "HuggingFaceH4/zephyr-7b-beta"
HF_API_URL    = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# Read the HuggingFace token from an environment variable so we never
# hard-code secrets in source code.
# To set it:  set HF_TOKEN=hf_your_token_here   (Windows cmd)
#             $env:HF_TOKEN="hf_your_token_here" (PowerShell)
# Get a free token at: https://huggingface.co/settings/tokens
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Timeout (seconds) for the external LLM API call.
# If the API doesn't respond within this time we catch the error gracefully.
LLM_TIMEOUT_SECONDS = 30

# The mandatory disclaimer appended to every response that contains an
# LLM-generated summary.  This is a hard-coded, non-negotiable constant —
# it MUST appear regardless of what the LLM returns.
MEDICAL_DISCLAIMER = (
    "DISCLAIMER: This tool is an AI-powered summary for educational/informational "
    "purposes and does not provide professional medical advice. Always consult a "
    "qualified healthcare professional for medical decisions."
)


# =============================================================================
# APPLICATION STATE  (module-level cache populated at startup)
# =============================================================================

chroma_collection = None   # ChromaDB collection object
embedding_model   = None   # SentenceTransformer model


# =============================================================================
# LIFESPAN CONTEXT MANAGER
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs startup logic before the server begins accepting requests,
    and shutdown logic when the server is stopped.
    """
    global chroma_collection, embedding_model

    print("\n[STARTUP]  Initializing ChromaDB + embedding model ...")
    chroma_collection, embedding_model = init_vector_store()
    doc_count = chroma_collection.count()
    print(f"[STARTUP]  Ready — {doc_count} chunk(s) indexed.\n")

    yield  # <-- server is live and handling requests here

    print("\n[SHUTDOWN]  Server shutting down gracefully.")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Medical Report Summarizer API",
    description=(
        "A Retrieval-Augmented Generation (RAG) API. "
        "Upload medical PDFs, then ask questions — the API retrieves relevant "
        "context from the vector store and uses an LLM to generate a cited, "
        "grounded answer with a mandatory medical disclaimer."
    ),
    version="0.3.0",
    lifespan=lifespan,
)


# =============================================================================
# PYDANTIC MODELS  (Request + Response schemas)
# =============================================================================

class QueryRequest(BaseModel):
    """
    JSON body for POST /api/query.

    Example:
        { "question": "What are the admission deadlines?", "top_k": 3 }
    """
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The natural-language question to answer from the document.",
        examples=["What graduate programs are available in Biomedical Engineering?"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of top matching chunks to retrieve (1-10).",
    )


class RetrievedChunk(BaseModel):
    """One text chunk returned from the vector store."""
    rank:        int   = Field(description="1 = most relevant.")
    score:       float = Field(description="Cosine distance — lower means more similar.")
    chunk_id:    str   = Field(description="Unique chunk identifier.")
    source_file: str   = Field(description="Source PDF filename.")
    page_number: int   = Field(description="Page this chunk was extracted from.")
    word_count:  int   = Field(description="Number of words in this chunk.")
    text:        str   = Field(description="The raw chunk text.")


class QueryResponse(BaseModel):
    """
    Full structured response from POST /api/query.

    Contains:
    - The original question
    - The retrieved evidence chunks (for transparency / debugging)
    - The LLM-generated answer with explicit citations
    - A mandatory medical/legal disclaimer
    """
    question:               str                  = Field(description="The question that was asked.")
    total_chunks_retrieved: int                  = Field(description="Number of chunks returned.")
    retrieved_chunks:       list[RetrievedChunk] = Field(description="Ranked evidence chunks.")
    llm_summary:            str | None           = Field(
        default=None,
        description=(
            "LLM-generated answer grounded in the retrieved chunks. "
            "Contains explicit chunk_id / page citations. "
            "Set to None if the LLM call failed."
        ),
    )
    disclaimer: str = Field(
        default=MEDICAL_DISCLAIMER,
        description="Mandatory legal/medical disclaimer — always present.",
    )


# =============================================================================
# LLM HELPER — build prompt + call HuggingFace Inference API
# =============================================================================

def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    Assemble the full chat prompt that will be sent to zephyr-7b-beta.

    Zephyr uses the ChatML / <|system|> / <|user|> / <|assistant|> format.
    We use that structure here for maximum reliability.

    Prompt engineering decisions
    ----------------------------
    1. SYSTEM prompt:
       - Establishes the model as a grounded, citation-enforcing assistant.
       - Explicitly forbids hallucination ("ONLY use the provided context").
       - Mandates the refusal phrase when context is insufficient.
       - Mandates citing chunk_id AND page_number for every factual claim.

    2. USER message:
       - Provides the numbered context chunks with metadata clearly labelled.
       - Ends with the user's question.

    Parameters
    ----------
    question : str
        The user's natural-language question.
    chunks : list[RetrievedChunk]
        The retrieved evidence chunks from the vector store.

    Returns
    -------
    str
        The fully formatted prompt string.
    """

    # ------------------------------------------------------------------
    # SYSTEM PROMPT — strict rules the model must follow
    # ------------------------------------------------------------------
    system_prompt = (
        "You are a precise, grounded medical document assistant. "
        "Your ONLY job is to answer questions using the provided context chunks. "
        "\n\n"
        "STRICT RULES YOU MUST FOLLOW:\n"
        "1. ONLY use information explicitly present in the provided context. "
        "   Do NOT infer, assume, or add any knowledge from outside the context.\n"
        "2. For EVERY factual statement you make, you MUST cite the source using "
        "   this exact format: [chunk_id: <id>, page: <number>]. "
        "   Example: 'The deadline is June 15 [chunk_id: doc_p1_c0, page: 1].'\n"
        "3. If the provided context chunks do NOT contain enough information to "
        "   answer the question, you MUST respond with exactly this phrase and "
        "   nothing else: "
        "'I cannot find the answer in the provided document.'\n"
        "4. Do NOT make up chunk IDs or page numbers. Only cite ones from the context.\n"
        "5. Keep your answer concise and factual."
    )

    # ------------------------------------------------------------------
    # Numbered context block — each chunk is clearly labelled with its
    # metadata so the model can cite it accurately
    # ------------------------------------------------------------------
    context_lines = []
    for chunk in chunks:
        context_lines.append(
            f"--- Context chunk {chunk.rank} ---\n"
            f"chunk_id  : {chunk.chunk_id}\n"
            f"page      : {chunk.page_number}\n"
            f"source    : {chunk.source_file}\n"
            f"text      : {chunk.text}"
        )
    context_block = "\n\n".join(context_lines)

    # ------------------------------------------------------------------
    # Assemble in Zephyr's ChatML format
    # <|system|>, <|user|>, <|assistant|> are special tokens the model
    # was fine-tuned on — using them gives much better instruction-following
    # than plain text.
    # ------------------------------------------------------------------
    prompt = (
        f"<|system|>\n{system_prompt}</s>\n"
        f"<|user|>\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}</s>\n"
        f"<|assistant|>\n"
    )

    return prompt


def call_llm(prompt: str) -> str | None:
    """
    Send the prompt to the HuggingFace Inference API and return the
    generated text, or None if the call fails for any reason.

    The function is intentionally defensive — it catches ALL exceptions
    so a flaky external API never causes a 500 crash in our endpoint.

    Parameters
    ----------
    prompt : str
        The fully assembled prompt from build_prompt().

    Returns
    -------
    str | None
        The LLM's generated text, or None on any failure.
    """

    # ------------------------------------------------------------------
    # Build the request headers
    # ------------------------------------------------------------------
    headers = {"Content-Type": "application/json"}

    # Attach the Bearer token if one is configured.
    # Without a token, HuggingFace still works but at a lower rate limit.
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    else:
        print("[LLM WARN]  HF_TOKEN not set — using unauthenticated (rate-limited) access.")

    # ------------------------------------------------------------------
    # HuggingFace Inference API payload
    # ------------------------------------------------------------------
    payload = {
        "inputs": prompt,
        "parameters": {
            # max_new_tokens: how many tokens the model can generate.
            # 512 is enough for a concise cited answer.
            "max_new_tokens": 512,

            # temperature: controls randomness.
            # 0.1 = very deterministic/factual (good for grounded RAG).
            # Increase toward 1.0 for more creative, varied responses.
            "temperature": 0.1,

            # return_full_text=False means we only get the assistant's
            # new output, NOT the entire prompt echoed back.
            "return_full_text": False,

            # stop sequences — tell the model to stop generating when it
            # hits the next turn boundary (prevents run-on output).
            "stop": ["</s>", "<|user|>", "<|system|>"],
        },
    }

    # ------------------------------------------------------------------
    # Make the HTTP request with a timeout guard
    # ------------------------------------------------------------------
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        )

        # Raise an exception for 4xx / 5xx HTTP status codes
        response.raise_for_status()

        # The API returns a list; the first element has our generated text
        result = response.json()

        if isinstance(result, list) and len(result) > 0:
            generated = result[0].get("generated_text", "").strip()
            return generated if generated else None

        # Unexpected response shape
        print(f"[LLM WARN]  Unexpected API response shape: {result}")
        return None

    except requests.exceptions.Timeout:
        # The API took longer than LLM_TIMEOUT_SECONDS
        print(f"[LLM ERROR]  HuggingFace API timed out after {LLM_TIMEOUT_SECONDS}s.")
        return None

    except requests.exceptions.ConnectionError as e:
        # No internet / DNS failure / API unreachable
        print(f"[LLM ERROR]  Connection error: {e}")
        return None

    except requests.exceptions.HTTPError as e:
        # 4xx / 5xx from the API (e.g. 503 model loading, 429 rate limit)
        status = e.response.status_code if e.response is not None else "unknown"
        body   = e.response.text[:200] if e.response is not None else ""
        print(f"[LLM ERROR]  HTTP {status} from HuggingFace API: {body}")
        return None

    except Exception as e:
        # Catch-all: JSON parse errors, unexpected SDK exceptions, etc.
        print(f"[LLM ERROR]  Unexpected error during LLM call: {type(e).__name__}: {e}")
        return None


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", summary="Health Check", tags=["Utility"])
def read_root() -> dict[str, Any]:
    """
    Health-check endpoint.
    Returns server status, version, and live vector store stats.
    """
    doc_count = chroma_collection.count() if chroma_collection else 0
    return {
        "status":  "ok",
        "message": "Medical Report Summarizer API is running.",
        "version": "0.3.0",
        "vector_store": {
            "status":         "loaded" if chroma_collection else "not loaded",
            "indexed_chunks": doc_count,
        },
        "llm_model": HF_MODEL_ID,
        "hf_token_configured": bool(HF_TOKEN),
    }


@app.post(
    "/api/query",
    response_model=QueryResponse,
    summary="Query the Medical Knowledge Base",
    tags=["RAG Pipeline"],
)
def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    """
    **Phase 2 Complete — Full RAG Pipeline.**

    ```
    User Question
         |
         v
    [ Embed with all-MiniLM-L6-v2 ]
         |
         v
    [ ChromaDB cosine similarity search ]
         |
         v
    [ Top-K chunks retrieved ]
         |
         v
    [ Build grounded prompt with strict citation rules ]
         |
         v
    [ HuggingFace Inference API: zephyr-7b-beta ]
         |
         v
    [ Structured JSON: chunks + cited summary + disclaimer ]
    ```
    """

    # ------------------------------------------------------------------
    # GUARD 1 — Startup check
    # ------------------------------------------------------------------
    if chroma_collection is None or embedding_model is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store is not initialised. Check server logs.",
        )

    # ------------------------------------------------------------------
    # GUARD 2 — Empty database check
    # ------------------------------------------------------------------
    if chroma_collection.count() == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                "The vector store is empty. "
                "Run `python vector_store.py` to ingest PDFs first."
            ),
        )

    # ------------------------------------------------------------------
    # STEP 1 — Retrieve top-K semantically similar chunks from ChromaDB
    # ------------------------------------------------------------------
    # The embedding model converts the question into a 384-dim vector and
    # ChromaDB returns the nearest stored vectors (cosine distance).
    raw_results: list[dict] = query_vector_store(
        collection=chroma_collection,
        model=embedding_model,
        query_text=request.question,
        n_results=request.top_k,
    )

    # ------------------------------------------------------------------
    # STEP 2 — Shape raw dicts into validated Pydantic RetrievedChunk objects
    # ------------------------------------------------------------------
    retrieved_chunks: list[RetrievedChunk] = [
        RetrievedChunk(
            rank=r["rank"],
            score=r["score"],
            chunk_id=r["chunk_id"],
            source_file=r["source_file"],
            page_number=r["page_number"],
            word_count=r["word_count"],
            text=r["text"],
        )
        for r in raw_results
    ]

    # ------------------------------------------------------------------
    # STEP 3 — Build the grounded, citation-enforcing prompt
    # ------------------------------------------------------------------
    # build_prompt() assembles:
    #   - A SYSTEM block with strict rules (no hallucination, cite everything,
    #     use the refusal phrase if context is insufficient)
    #   - A USER block containing the numbered context chunks (with their
    #     chunk_id and page_number clearly labelled) + the question
    #   - The <|assistant|> tag so the model knows it's its turn to respond
    prompt = build_prompt(
        question=request.question,
        chunks=retrieved_chunks,
    )

    # ------------------------------------------------------------------
    # STEP 4 — Call the LLM (HuggingFace Inference API)
    # ------------------------------------------------------------------
    # call_llm() is fully defensive — it catches every possible failure
    # (timeout, connection error, HTTP error, JSON parse error) and returns
    # None instead of raising, so our endpoint never crashes with a 500.
    llm_summary: str | None = call_llm(prompt)

    # If the LLM call failed for any reason, we surface a clear fallback
    # message instead of returning None silently.
    if llm_summary is None:
        llm_summary = (
            "[LLM unavailable] The language model did not return a response. "
            "The retrieved context chunks above are still valid and can be "
            "reviewed directly. Common causes: HuggingFace API is loading the "
            "model (~20s), rate limit hit, or no internet connection. "
            "Set the HF_TOKEN environment variable for higher rate limits."
        )

    # ------------------------------------------------------------------
    # STEP 5 — Return the complete structured response
    # ------------------------------------------------------------------
    # The `disclaimer` field is populated automatically by Pydantic from
    # the MEDICAL_DISCLAIMER constant defined at the top of this file.
    # It is always present — callers cannot remove or override it.
    return QueryResponse(
        question=request.question,
        total_chunks_retrieved=len(retrieved_chunks),
        retrieved_chunks=retrieved_chunks,
        llm_summary=llm_summary,
        disclaimer=MEDICAL_DISCLAIMER,
    )
