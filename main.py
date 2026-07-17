# =============================================================================
# main.py  —  Phase 2: FastAPI Orchestration Layer
# =============================================================================
# What this file does (plain English):
#
#   1. STARTUP  — When the server boots, it loads the ChromaDB vector store
#                 and the sentence-embedding model into memory ONCE.
#                 Keeping them in memory avoids reloading on every request,
#                 which would be very slow (models take 1-2 seconds to load).
#
#   2. ROOT     — GET /  → health-check endpoint, confirms the server is alive.
#
#   3. QUERY    — POST /api/query
#                 Accepts a JSON body with a "question" string, searches the
#                 vector store for the 3 most relevant chunks, and returns them
#                 as structured JSON.
#                 *** This is where we will add the LLM call in Phase 3 ***
#
# How to run:
#   uvicorn main:app --reload
#
# Then test with curl or the built-in Swagger UI at:
#   http://127.0.0.1:8000/docs
# =============================================================================

from contextlib import asynccontextmanager   # For the lifespan startup/shutdown hook
from typing import Any                       # Generic type hint for flexible dicts

from fastapi import FastAPI, HTTPException   # FastAPI core + error responses
from pydantic import BaseModel, Field        # Data validation for request bodies

# Our Phase 1 modules
from vector_store import init_vector_store, query_vector_store


# =============================================================================
# APPLICATION STATE
# =============================================================================
# We store the ChromaDB collection and embedding model here as module-level
# variables so every request handler can access them without re-initializing.
#
# Think of this as a "global cache" that lives for the lifetime of the server
# process.  It is populated during startup (see the lifespan function below)
# and read by the /api/query endpoint on each request.
# =============================================================================

# Will hold the ChromaDB collection object after startup
chroma_collection = None

# Will hold the loaded SentenceTransformer model after startup
embedding_model = None


# =============================================================================
# LIFESPAN CONTEXT MANAGER  (replaces the older @app.on_event("startup"))
# =============================================================================
# The lifespan approach is the modern FastAPI pattern (v0.93+).
# Code BEFORE `yield` runs at server startup.
# Code AFTER  `yield` runs at server shutdown (useful for cleanup).
#
# Using `global` here lets us write to the module-level variables above
# from inside this function.
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler.

    Startup:  Initialize ChromaDB + load the embedding model into RAM.
    Shutdown: (Nothing needed — Python's garbage collector handles cleanup.)
    """
    global chroma_collection, embedding_model

    # ------------------------------------------------------------------
    # STARTUP — runs once when `uvicorn main:app` is executed
    # ------------------------------------------------------------------
    print("\n[STARTUP]  Initializing vector store and embedding model ...")

    # init_vector_store() opens (or creates) the local chroma_db/ database
    # and loads 'all-MiniLM-L6-v2' into memory.  Both objects are returned
    # and cached in the module-level variables defined above.
    chroma_collection, embedding_model = init_vector_store()

    # Confirm how many documents are already indexed so we know the DB is live
    doc_count = chroma_collection.count()
    print(f"[STARTUP]  Ready.  Vector store contains {doc_count} indexed chunk(s).\n")

    # ------------------------------------------------------------------
    # Hand control back to FastAPI — the server starts accepting requests
    # ------------------------------------------------------------------
    yield

    # ------------------------------------------------------------------
    # SHUTDOWN — runs when the server is stopped (Ctrl+C / SIGTERM)
    # ------------------------------------------------------------------
    print("\n[SHUTDOWN]  Cleaning up resources ...")
    # No explicit cleanup needed for ChromaDB or SentenceTransformer,
    # but this is where you would close DB connections, flush logs, etc.


# =============================================================================
# FASTAPI APPLICATION INSTANCE
# =============================================================================

app = FastAPI(
    title="Medical Report Summarizer API",
    description=(
        "A Retrieval-Augmented Generation (RAG) API that ingests medical PDFs, "
        "stores them in a local vector database, and answers clinical questions "
        "by retrieving the most relevant text chunks."
    ),
    version="0.2.0",
    lifespan=lifespan,   # <-- wire in the startup/shutdown logic above
)


# =============================================================================
# PYDANTIC REQUEST / RESPONSE MODELS
# =============================================================================
# Pydantic models do two things for us automatically:
#   1. VALIDATION  — If the incoming JSON is missing a field or has the wrong
#                    type, FastAPI returns a helpful 422 error automatically.
#   2. DOCUMENTATION — FastAPI reads these models to generate the interactive
#                      Swagger UI at /docs with no extra work from us.
# =============================================================================

class QueryRequest(BaseModel):
    """
    JSON body expected by POST /api/query.

    Example payload:
        {
            "question": "What are the admission deadlines for graduate programs?"
        }
    """
    question: str = Field(
        ...,                              # `...` means this field is REQUIRED
        min_length=3,                     # guard against empty / trivial queries
        max_length=500,                   # prevent absurdly long inputs
        description="The clinical or academic question to search for.",
        examples=["What graduate programs are available?"],
    )
    top_k: int = Field(
        default=3,                        # return 3 results unless caller overrides
        ge=1,                             # must be at least 1
        le=10,                            # cap at 10 to prevent huge responses
        description="Number of top matching chunks to return (1-10).",
    )


class RetrievedChunk(BaseModel):
    """
    A single retrieved text chunk returned inside the query response.
    """
    rank: int           = Field(description="1 = most relevant, 2 = second-most, etc.")
    score: float        = Field(description="Cosine distance (lower = more similar to query).")
    chunk_id: str       = Field(description="Unique identifier for this chunk.")
    source_file: str    = Field(description="Original PDF filename this chunk came from.")
    page_number: int    = Field(description="Page in the source PDF.")
    word_count: int     = Field(description="Number of words in this chunk.")
    text: str           = Field(description="The raw text of the retrieved chunk.")


class QueryResponse(BaseModel):
    """
    Structured JSON response returned by POST /api/query.
    """
    question: str               = Field(description="The original question that was asked.")
    total_chunks_retrieved: int = Field(description="How many chunks were returned.")
    retrieved_chunks: list[RetrievedChunk]  = Field(
        description="Ordered list of the most relevant text chunks."
    )
    # ------------------------------------------------------------------
    # PHASE 3 PLACEHOLDER — LLM Summary
    # ------------------------------------------------------------------
    # In the next phase, this field will hold the LLM-generated answer
    # synthesised from the retrieved chunks above.
    # For now it carries a clear note so reviewers know it is intentional.
    # ------------------------------------------------------------------
    llm_summary: str | None = Field(
        default=None,
        description=(
            "[Phase 3] LLM-generated summary of the retrieved chunks. "
            "Currently None — will be populated after the LLM integration step."
        ),
    )


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get(
    "/",
    summary="Health Check",
    tags=["Utility"],
)
def read_root() -> dict[str, Any]:
    """
    Simple health-check endpoint.

    Returns a welcome message plus a live count of indexed documents so you
    can confirm the vector store loaded correctly on startup.
    """
    doc_count = chroma_collection.count() if chroma_collection else 0
    return {
        "status":  "ok",
        "message": "Welcome to the Medical Report Summarizer API",
        "version": "0.2.0",
        "vector_store": {
            "status":          "loaded" if chroma_collection else "not loaded",
            "indexed_chunks":  doc_count,
        },
    }


@app.post(
    "/api/query",
    response_model=QueryResponse,
    summary="Query the Medical Knowledge Base",
    tags=["RAG Pipeline"],
)
def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    """
    **Phase 2 — Retrieval endpoint.**

    Accepts a natural-language question, searches the vector store for the
    most semantically relevant text chunks, and returns them as structured JSON.

    **RAG Pipeline so far:**
    ```
    User Question
         │
         ▼
    [ Embed question with all-MiniLM-L6-v2 ]
         │
         ▼
    [ ChromaDB nearest-neighbour search ]
         │
         ▼
    Top-K chunks returned ← YOU ARE HERE
         │
         ▼
    [ *** LLM summarisation — Phase 3 *** ]
         │
         ▼
    Structured JSON response
    ```
    """

    # ------------------------------------------------------------------
    # GUARD — Make sure the startup hook ran successfully
    # ------------------------------------------------------------------
    if chroma_collection is None or embedding_model is None:
        # This should never happen in normal operation, but it could occur
        # if the server was started without the lifespan hook (e.g. in tests).
        raise HTTPException(
            status_code=503,
            detail=(
                "Vector store is not initialised yet. "
                "Please wait a moment and retry, or check server logs."
            ),
        )

    # ------------------------------------------------------------------
    # GUARD — Warn if the collection is empty (no PDFs have been ingested)
    # ------------------------------------------------------------------
    if chroma_collection.count() == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                "The vector store is empty. "
                "Run `python vector_store.py` to ingest PDFs before querying."
            ),
        )

    # ------------------------------------------------------------------
    # STEP 1 — Retrieve the top-K most relevant chunks from ChromaDB
    # ------------------------------------------------------------------
    # query_vector_store() converts the question into a vector, performs a
    # cosine similarity search, and returns a ranked list of chunk dicts.
    raw_results: list[dict] = query_vector_store(
        collection=chroma_collection,
        model=embedding_model,
        query_text=request.question,
        n_results=request.top_k,
    )

    # ------------------------------------------------------------------
    # STEP 2 — Shape the raw results into our validated Pydantic model
    # ------------------------------------------------------------------
    # Each raw_result dict has: rank, score, chunk_id, source_file,
    # page_number, word_count, text  (set in vector_store.py)
    retrieved_chunks = [
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
    # STEP 3 — *** LLM INTEGRATION POINT (Phase 3) ***
    # ------------------------------------------------------------------
    # In the next phase, THIS is where we will:
    #
    #   a) Concatenate the retrieved chunk texts into a single context string:
    #        context = "\n\n".join(chunk.text for chunk in retrieved_chunks)
    #
    #   b) Build a prompt with a medical-safe system instruction, e.g.:
    #        prompt = f"""
    #        You are a clinical assistant. Using ONLY the context below,
    #        answer the question. Do not infer beyond the provided text.
    #        Add a disclaimer that this is not professional medical advice.
    #
    #        Context:
    #        {context}
    #
    #        Question: {request.question}
    #        """
    #
    #   c) Call the LLM API (e.g. Google Gemini / OpenAI GPT-4o):
    #        import google.generativeai as genai
    #        llm_response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
    #        summary = llm_response.text
    #
    #   d) Populate the `llm_summary` field in the response below.
    #
    # For now, llm_summary is None — the comment block above makes this
    # intentional and visible to anyone reading the code.
    # ------------------------------------------------------------------
    llm_summary = None   # <-- REPLACE with LLM call in Phase 3

    # ------------------------------------------------------------------
    # STEP 4 — Build and return the final structured response
    # ------------------------------------------------------------------
    return QueryResponse(
        question=request.question,
        total_chunks_retrieved=len(retrieved_chunks),
        retrieved_chunks=retrieved_chunks,
        llm_summary=llm_summary,
    )
