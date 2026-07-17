# =============================================================================
# main.py  —  Phase 2 + 3 (Complete): FastAPI RAG Orchestration + Groq LLM
# =============================================================================
# What this file does (plain English):
#
#   1. STARTUP  — Loads the ChromaDB vector store and embedding model into
#                 memory ONCE at boot so every request is fast.
#
#   2. GET /    — Health-check: confirms the server is alive and shows how
#                 many chunks are indexed.
#
#   3. POST /api/query  — Full async RAG pipeline:
#       a) Retrieve top-K semantically matching text chunks from ChromaDB.
#       b) Build a structured healthcare messages array with strict citation
#          rules and a mandatory "I cannot find the answer" refusal clause.
#       c) Asynchronously call the Groq Inference API (llama-3.3-70b-versatile).
#       d) Extract the answer from response.choices[0].message.content.
#       e) Return the LLM answer + retrieved chunks + mandatory disclaimer.
#          Falls back gracefully to a descriptive message on any error.
#
# How to run:
#   uvicorn main:app --reload
#
# Set your Groq API key (free at console.groq.com):
#   Add GROQ_API_KEY=gsk_... to a .env file  — OR —
#   PowerShell:  $env:GROQ_API_KEY="gsk_your_key_here"
#   CMD:         set GROQ_API_KEY=gsk_your_key_here
#
# Interactive Swagger docs:
#   http://127.0.0.1:8000/docs
# =============================================================================

import os                                    # Read environment variables (API key)
import asyncio                               # Offload CPU-bound embedding to thread pool
from contextlib import asynccontextmanager   # Modern FastAPI lifespan hook
from typing import Any                       # Generic type hint for flexible dicts

from dotenv import load_dotenv               # Load GROQ_API_KEY from .env file
from groq import AsyncGroq                   # Official Groq async Python client
from fastapi import FastAPI, HTTPException   # FastAPI core + structured error responses
from pydantic import BaseModel, Field        # Request/response validation

# Load .env before os.getenv() calls so GROQ_API_KEY is available immediately
load_dotenv()

# Phase 1 modules
from vector_store import init_vector_store, query_vector_store



# =============================================================================
# CONSTANTS
# =============================================================================

# Groq model ID — llama-3.3-70b-versatile is confirmed active on the Groq
# Inference API (verified via check_models.py).  It offers:
#   • 128 k-token context window
#   • Extremely fast inference (~280 tokens/s on Groq hardware)
#   • Strong instruction-following for RAG citation tasks
GROQ_MODEL_ID = "llama-3.3-70b-versatile"

# Read the Groq API key from the environment (populated from .env by load_dotenv
# above, or set directly as an OS environment variable).
# Get a free key at: https://console.groq.com
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Maximum new tokens the model may generate per call.
# 1024 tokens ≈ 768 words — plenty for a concise, fully-cited RAG answer.
# Groq's ultra-fast inference makes this cost-free in terms of latency.
LLM_MAX_NEW_TOKENS = 1024

# The mandatory disclaimer appended to EVERY response.
# This is a hard-coded, non-negotiable constant — it MUST appear regardless
# of what the LLM returns or whether the LLM call itself succeeds.
MEDICAL_DISCLAIMER = (
    "DISCLAIMER: This tool is an AI-powered summary for educational/informational "
    "purposes and does not provide professional medical advice. "
    "Always consult a qualified healthcare professional for medical decisions."
)

# Human-readable label inserted into fallback llm_summary when the LLM is
# unavailable so API consumers can programmatically detect a fallback response.
LLM_UNAVAILABLE_PREFIX = "[LLM unavailable]"

# Cosine-distance threshold for the confidence guardrail.
#
# ChromaDB stores cosine *distance* (not similarity): 0 = identical vectors,
# 2 = perfectly opposite.  all-MiniLM-L6-v2 typical score ranges:
#
#   0.00 – 0.30  →  strong match    (question clearly answered by this chunk)
#   0.30 – 0.55  →  moderate match  (partial or tangential relevance)
#   0.55 – 0.80  →  weak match      (loosely related topic)
#   0.80+        →  poor match      (likely off-topic; document may not contain
#                                    the answer at all)
#
# We flag low confidence when the BEST retrieved chunk (rank 1, lowest score)
# exceeds this threshold, signalling the corpus probably cannot answer the
# question reliably.  0.55 is the boundary between "moderate" and "weak".
CONFIDENCE_DISTANCE_THRESHOLD: float = 0.55

# Canonical prefix of the LLM strict-refusal string (RULE 3 in the system
# prompt).  We detect this in llm_summary to auto-raise the confidence flag
# even when the top chunk score looked acceptable.
REFUSAL_PREFIX = "I cannot find the answer in the provided document"


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
        "context from the vector store and uses the Groq LLM to generate a "
        "cited, grounded answer with a mandatory medical disclaimer.\n\n"
        f"**Model**: `{GROQ_MODEL_ID}` via Groq Inference API  \n"
        "**Citation rule**: Every factual claim must include `[chunk_id: ..., page: ...]`.  \n"
        "**Refusal rule**: Returns *'I cannot find the answer in the provided document'* "
        "when the context is insufficient."
    ),
    version="0.5.0",
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

    Fields
    ------
    question               : The original question that was asked.
    total_chunks_retrieved : How many chunks were returned by the vector store.
    retrieved_chunks       : The ranked evidence chunks (for transparency).
    llm_summary            : The Groq LLM-generated answer, grounded in the
                             retrieved chunks. Every factual claim is cited
                             with [chunk_id, page]. Set to a descriptive
                             fallback string (prefixed '[LLM unavailable]')
                             if the Groq API call failed.
    llm_model              : Groq model ID used for generation.
    low_confidence_warning : True when the answer may be unreliable because
                             the best-matching chunk exceeds the cosine-distance
                             threshold (CONFIDENCE_DISTANCE_THRESHOLD) OR the
                             LLM triggered its strict refusal string.  Callers
                             should surface this flag prominently in any UI.
    disclaimer             : Mandatory legal/medical safety disclaimer — ALWAYS
                             present regardless of LLM success or failure.
    """
    question:               str                  = Field(description="The question that was asked.")
    total_chunks_retrieved: int                  = Field(description="Number of chunks returned.")
    retrieved_chunks:       list[RetrievedChunk] = Field(description="Ranked evidence chunks.")
    llm_summary:            str | None           = Field(
        default=None,
        description=(
            "Groq LLM-generated answer grounded in the retrieved chunks. "
            "Contains explicit [chunk_id, page] citations per factual claim. "
            "Returns 'I cannot find the answer in the provided document' when "
            "context is insufficient. Prefixed '[LLM unavailable]' if the "
            "Groq API call itself failed."
        ),
    )
    llm_model:  str = Field(
        default=GROQ_MODEL_ID,
        description="Groq model ID used for generation.",
    )
    low_confidence_warning: bool = Field(
        default=False,
        description=(
            "True when the answer reliability is questionable. Triggered by either: "
            "(1) the top retrieved chunk's cosine distance exceeds "
            f"{CONFIDENCE_DISTANCE_THRESHOLD} (all-MiniLM-L6-v2 scale: 0=identical, "
            "2=opposite), indicating the document likely does not contain the "
            "answer; or (2) the LLM returned its strict refusal string. "
            "Callers should display a prominent warning to the user when True."
        ),
    )
    disclaimer: str = Field(
        default=MEDICAL_DISCLAIMER,
        description="Mandatory legal/medical safety disclaimer — always present.",
    )


# =============================================================================
# *** LLM INTEGRATION POINT (Phase 3) ***
# LLM HELPERS — build_messages() + async call_llm_async()
# =============================================================================

def build_messages(
    question: str,
    chunks: list[RetrievedChunk],
) -> list[dict[str, str]]:
    """
    Build the OpenAI-compatible ``messages`` array sent to the Groq API.

    The Groq SDK uses the same chat-completion contract as OpenAI:
    a list of {"role": ..., "content": ...} dicts.  We produce exactly two
    messages — a ``system`` message that encodes all strict RAG rules, and a
    ``user`` message that provides the retrieved context chunks and the
    question.

    Prompt engineering decisions
    ----------------------------
    System message
      • Establishes the model as a grounded, citation-enforcing medical
        assistant — its ONLY knowledge source is the provided context.
      • 5 hard rules (hallucination ban, mandatory citation format,
        verbatim refusal phrase, conciseness, no medical advice).

    User message
      • Starts with an explicit chunk-id quick-reference index so the model
        can locate valid IDs without parsing every chunk body.
      • Numbered context chunks, each with chunk_id, page, source, word_count,
        and full text clearly labelled.
      • Ends with the user's question.

    Parameters
    ----------
    question : str
        The user's natural-language question.
    chunks : list[RetrievedChunk]
        The retrieved evidence chunks from the vector store.

    Returns
    -------
    list[dict[str, str]]
        A two-element list: [{"role": "system", ...}, {"role": "user", ...}]
        ready to pass directly to ``client.chat.completions.create()``.
    """

    # ------------------------------------------------------------------
    # SYSTEM MESSAGE — strict healthcare rules the model MUST obey
    # ------------------------------------------------------------------
    system_content = (
        "You are a precise, grounded medical document assistant operating under "
        "strict clinical information safety rules.\n\n"

        "YOUR SOLE JOB: Answer the user's question using ONLY the text chunks "
        "provided in the user message. You have no other source of knowledge "
        "for this task.\n\n"

        "=" * 46 + "\n"
        "MANDATORY RULES — violating any rule is a critical failure:\n"
        "=" * 46 + "\n\n"

        "RULE 1 — GROUNDING (No Hallucination)\n"
        "  • Use ONLY information that is explicitly present in the provided "
        "context chunks. Do NOT infer, assume, extrapolate, or add knowledge "
        "from outside the provided text, even if you believe it to be true.\n\n"

        "RULE 2 — MANDATORY CITATION\n"
        "  • For EVERY factual statement you make, you MUST append a citation "
        "immediately after the fact using this EXACT format:\n"
        "      [chunk_id: <id>, page: <number>]\n"
        "  • Example: 'The patient was prescribed metformin 500 mg daily "
        "[chunk_id: report_p2_c1, page: 2].'\n"
        "  • Only cite chunk_ids and page numbers that appear in the context. "
        "Never invent or guess an id.\n\n"

        "RULE 3 — STRICT REFUSAL\n"
        "  • If the provided context chunks do NOT contain sufficient information "
        "to answer the question, you MUST respond with EXACTLY this phrase and "
        "NOTHING ELSE:\n"
        "      I cannot find the answer in the provided document.\n"
        "  • Do not apologise, do not elaborate, do not suggest alternatives. "
        "Output that single sentence verbatim.\n\n"

        "RULE 4 — CONCISENESS\n"
        "  • Keep your answer factual and concise. Do not pad with filler text.\n\n"

        "RULE 5 — NO MEDICAL ADVICE\n"
        "  • You are summarising a document. You are NOT a physician and must "
        "NOT give personal medical recommendations or treatment decisions."
    )

    # ------------------------------------------------------------------
    # USER MESSAGE — chunk-id index + numbered context chunks + question
    # ------------------------------------------------------------------
    chunk_index_lines: list[str] = []
    context_lines:     list[str] = []

    for chunk in chunks:
        # Quick-reference index line for reliable citation lookup
        chunk_index_lines.append(
            f"  • chunk_id={chunk.chunk_id}  |  page={chunk.page_number}  "
            f"|  source={chunk.source_file}"
        )
        # Full chunk body with all metadata clearly labelled
        context_lines.append(
            f"--- Context chunk #{chunk.rank} ---\n"
            f"chunk_id   : {chunk.chunk_id}\n"
            f"page       : {chunk.page_number}\n"
            f"source     : {chunk.source_file}\n"
            f"word_count : {chunk.word_count}\n"
            f"\nTEXT:\n{chunk.text}\n"
            f"--- end chunk #{chunk.rank} ---"
        )

    chunk_index_block = "\n".join(chunk_index_lines)
    context_block     = "\n\n".join(context_lines)

    user_content = (
        f"AVAILABLE CHUNK IDs (use ONLY these for citations):\n"
        f"{chunk_index_block}\n\n"
        f"CONTEXT CHUNKS:\n"
        f"{context_block}\n\n"
        f"QUESTION: {question}"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]


async def call_llm_async(
    messages: list[dict[str, str]],
) -> str | None:
    """
    Asynchronously call the Groq Inference API via ``AsyncGroq`` and return
    the generated text string, or None if the call fails for any reason.

    Uses the official ``groq`` Python SDK's ``AsyncGroq`` client, which
    provides a native async interface backed by ``httpx`` — the FastAPI
    event loop is never blocked.

    Response extraction
    -------------------
    Groq returns an OpenAI-compatible ``ChatCompletion`` object.  The
    generated text is at::

        response.choices[0].message.content

    That string is stripped and mapped directly to ``llm_summary`` in the
    final ``QueryResponse``.

    Fallback contract
    -----------------
    This function NEVER raises — it catches all exceptions and returns None
    so a flaky external service never causes a 500 crash in our endpoint.

    Parameters
    ----------
    messages : list[dict[str, str]]
        The messages array from ``build_messages()``, containing the system
        prompt and the user message with context chunks + question.

    Returns
    -------
    str | None
        The model's generated text (stripped), or None on any failure.
    """

    # ------------------------------------------------------------------
    # Guard — fail fast with a clear log if the API key is missing
    # ------------------------------------------------------------------
    if not GROQ_API_KEY:
        print(
            "[LLM ERROR]  GROQ_API_KEY is not set. "
            "Add it to your .env file or set the environment variable. "
            "Get a free key at https://console.groq.com"
        )
        return None

    # ------------------------------------------------------------------
    # Call the Groq Inference API
    # AsyncGroq is initialised per-call — it is lightweight and the SDK
    # manages connection pooling internally.
    # ------------------------------------------------------------------
    try:
        print(f"[LLM]  Sending request to Groq ({GROQ_MODEL_ID}) ...")

        async with AsyncGroq(api_key=GROQ_API_KEY) as client:
            response = await client.chat.completions.create(
                model=GROQ_MODEL_ID,
                messages=messages,                # type: ignore[arg-type]
                max_tokens=LLM_MAX_NEW_TOKENS,

                # temperature=0.1 → very deterministic/factual output.
                # Ideal for grounded RAG where reproducible citations matter.
                temperature=0.1,

                # top_p nucleus sampling — paired with low temperature for
                # tight, factual responses.
                top_p=0.9,

                # stream=False — wait for the full response before returning.
                # Keeps the downstream mapping to llm_summary simple.
                stream=False,
            )

        # ------------------------------------------------------------------
        # Extract generated text from the OpenAI-compatible response object
        # response.choices[0].message.content  →  the assistant's reply string
        # ------------------------------------------------------------------
        if response.choices and response.choices[0].message.content:
            generated: str = response.choices[0].message.content.strip()
            if generated:
                word_count = len(generated.split())
                print(f"[LLM]  Generation complete ({word_count} words).")
                # Map the real model generation string to llm_summary
                return generated
            else:
                print("[LLM WARN]  Groq returned an empty content string.")
                return None

        print("[LLM WARN]  Groq response had no choices or empty message.")
        return None

    # ------------------------------------------------------------------
    # Granular exception handlers — each logs a clear diagnostic message.
    # We import Groq exception types inline to avoid a hard top-level
    # dependency crash if the package is somehow unavailable at boot.
    # ------------------------------------------------------------------
    except Exception as exc:
        # Covers groq.APIConnectionError, groq.RateLimitError,
        # groq.APIStatusError, groq.APITimeoutError, and any other errors.
        exc_type = type(exc).__name__
        exc_msg  = str(exc)

        if "AuthenticationError" in exc_type or "401" in exc_msg:
            print(
                "[LLM ERROR]  Groq authentication failed — check that "
                "GROQ_API_KEY in your .env is valid."
            )
        elif "RateLimitError" in exc_type or "429" in exc_msg:
            print(
                "[LLM ERROR]  Groq rate limit exceeded. "
                "Wait a moment and retry, or upgrade your Groq plan."
            )
        elif "Timeout" in exc_type or "timeout" in exc_msg.lower():
            print(
                f"[LLM ERROR]  Groq API call timed out: {exc_msg}"
            )
        elif "Connection" in exc_type or "connect" in exc_msg.lower():
            print(
                f"[LLM ERROR]  Could not connect to Groq API: {exc_msg}"
            )
        else:
            print(
                f"[LLM ERROR]  Unexpected Groq error — "
                f"{exc_type}: {exc_msg}"
            )

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
        "version": "0.5.0",
        "vector_store": {
            "status":         "loaded" if chroma_collection else "not loaded",
            "indexed_chunks": doc_count,
        },
        "llm_provider":         "Groq",
        "llm_model":            GROQ_MODEL_ID,
        "groq_api_key_configured": bool(GROQ_API_KEY),
        "disclaimer":           MEDICAL_DISCLAIMER,
    }


@app.post(
    "/api/query",
    response_model=QueryResponse,
    summary="Query the Medical Knowledge Base",
    tags=["RAG Pipeline"],
)
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    """
    **Phase 2 + 3 Complete — Full Async RAG Pipeline (Groq Edition).**

    ```
    User Question
         |
         v
    [ Embed with all-MiniLM-L6-v2  (asyncio.to_thread — non-blocking) ]
         |
         v
    [ ChromaDB cosine similarity search ]
         |
         v
    [ Top-K chunks retrieved ]
         |
         v
    [ STEP 2.5 — Confidence Guardrail: top chunk score vs threshold ]
         |
         v
    [ build_messages(): system (5 RAG rules) + user (chunk index + context + question) ]
         |
         v
    [ call_llm_async(): AsyncGroq → client.chat.completions.create() ]
         |              model: llama-3.3-70b-versatile  (Groq Inference API)
         |              extract: response.choices[0].message.content
         v
    [ JSON: retrieved_chunks + llm_summary + low_confidence_warning + disclaimer ]
    ```

    **Confidence flag**: `low_confidence_warning=True` when the best-matching
    chunk's cosine distance exceeds `CONFIDENCE_DISTANCE_THRESHOLD` (0.55) OR
    when the LLM returns its strict refusal string.

    **Citation rule**: Every factual claim in `llm_summary` carries
    `[chunk_id: <id>, page: <n>]` immediately after the fact.

    **Refusal rule**: Returns *"I cannot find the answer in the provided
    document."* verbatim when retrieved chunks lack the answer.

    **Disclaimer**: The `disclaimer` field is hard-coded and always present —
    it cannot be removed or overridden by callers.
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
    # model.encode() is CPU-bound; offload to a thread pool via
    # asyncio.to_thread() so the async event loop stays free.
    raw_results: list[dict] = await asyncio.to_thread(
        query_vector_store,
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
    # STEP 2.5 — Confidence Guardrail (Phase 2 safety feature)
    # ------------------------------------------------------------------
    # Evaluate the top-ranked chunk's cosine distance (rank=1, lowest score).
    # ChromaDB cosine distance: 0 = identical, 2 = opposite.
    # all-MiniLM-L6-v2 empirical bands:
    #   0.00–0.30  strong match  |  0.30–0.55  moderate  |
    #   0.55–0.80  weak          |  0.80+       poor / off-topic
    #
    # We pre-emptively set the flag here based on retrieval quality alone.
    # It may also be set again after the LLM responds (refusal detection below).
    low_confidence_warning: bool = False

    if retrieved_chunks:
        top_score: float = retrieved_chunks[0].score   # rank-1 chunk, lowest distance
        if top_score > CONFIDENCE_DISTANCE_THRESHOLD:
            low_confidence_warning = True
            print(
                f"[CONFIDENCE]  Low-confidence flag raised: top chunk distance "
                f"{top_score:.4f} > threshold {CONFIDENCE_DISTANCE_THRESHOLD}. "
                f"Chunk: {retrieved_chunks[0].chunk_id}"
            )
        else:
            print(
                f"[CONFIDENCE]  Top chunk distance {top_score:.4f} ≤ "
                f"{CONFIDENCE_DISTANCE_THRESHOLD} — confidence OK."
            )

    # ------------------------------------------------------------------
    # STEP 3 — Build the Groq messages array
    # ------------------------------------------------------------------
    # build_messages() returns a list[dict] with two entries:
    #   [{"role": "system", "content": <5 strict RAG rules>},
    #    {"role": "user",   "content": <chunk-id index + context + question>}]
    # This is passed verbatim to client.chat.completions.create(messages=...).
    messages = build_messages(
        question=request.question,
        chunks=retrieved_chunks,
    )

    # ------------------------------------------------------------------
    # STEP 4 — Asynchronously call the Groq LLM
    #          *** LLM INTEGRATION POINT (Phase 3) ***
    # ------------------------------------------------------------------
    # call_llm_async() is fully defensive:
    #   • AsyncGroq — native async, non-blocking, event-loop friendly.
    #   • Extracts response.choices[0].message.content as the answer string.
    #   • Catches all Groq exceptions (auth, rate-limit, timeout, network).
    #   • Returns str | None — NEVER raises, NEVER crashes the endpoint.
    llm_summary: str | None = await call_llm_async(messages)

    # ------------------------------------------------------------------
    # STEP 5 — Populate llm_summary with a descriptive fallback message
    #          if the Groq API call failed for any reason
    # ------------------------------------------------------------------
    # Prefixed with LLM_UNAVAILABLE_PREFIX so consumers can detect it
    # programmatically.  The retrieved chunks are still valid and returned.
    if llm_summary is None:
        llm_summary = (
            f"{LLM_UNAVAILABLE_PREFIX} The Groq language model did not return "
            "a response. The retrieved context chunks above are still valid and "
            "can be reviewed directly.\n\n"
            "Common causes:\n"
            "  • GROQ_API_KEY is missing or invalid — check your .env file.\n"
            "  • Groq rate limit hit — wait a moment and retry.\n"
            "  • No internet connection or DNS failure.\n"
            "  • The selected model is temporarily unavailable.\n\n"
            "Fix: verify GROQ_API_KEY is set correctly "
            "(https://console.groq.com) and that the model "
            f"'{GROQ_MODEL_ID}' is listed as active."
        )

    # ------------------------------------------------------------------
    # STEP 5.5 — Refusal-string detection → auto-raise confidence flag
    # ------------------------------------------------------------------
    # Even if the retrieval score looked acceptable, the LLM may still
    # determine the context is insufficient and output the strict refusal
    # phrase (RULE 3 of the system prompt).  We detect that here and
    # ensure low_confidence_warning is True in that case as well.
    if llm_summary and llm_summary.startswith(REFUSAL_PREFIX):
        if not low_confidence_warning:
            low_confidence_warning = True
            print(
                "[CONFIDENCE]  Low-confidence flag raised: LLM returned "
                "strict refusal string despite retrieval score passing threshold."
            )

    # ------------------------------------------------------------------
    # STEP 6 — Return the complete structured JSON response
    # ------------------------------------------------------------------
    # Schema highlights:
    #   llm_summary            — response.choices[0].message.content mapped here
    #                            (or the descriptive fallback message above)
    #   llm_model              — GROQ_MODEL_ID for auditability / reproducibility
    #   low_confidence_warning — True if retrieval was weak OR LLM refused
    #   disclaimer             — MEDICAL_DISCLAIMER constant, hard-coded and
    #                            always present; callers cannot remove it
    return QueryResponse(
        question=request.question,
        total_chunks_retrieved=len(retrieved_chunks),
        retrieved_chunks=retrieved_chunks,
        llm_summary=llm_summary,                    # ← response.choices[0].message.content
        llm_model=GROQ_MODEL_ID,
        low_confidence_warning=low_confidence_warning,  # ← confidence guardrail flag
        disclaimer=MEDICAL_DISCLAIMER,              # ← mandatory legal/medical safety disclaimer
    )
