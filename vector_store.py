# =============================================================================
# vector_store.py  —  Phase 1: Vector Storage
# =============================================================================
# What this file does (plain English):
#
#   1. INIT    — Creates (or reopens) a local ChromaDB database stored on disk.
#                It also loads the 'all-MiniLM-L6-v2' sentence-embedding model,
#                which converts text into a list of 384 numbers (a "vector")
#                that captures the *meaning* of the text.
#
#   2. INGEST  — Calls parser.py to get all text chunks, then:
#                  a) Converts each chunk's text into a 384-dimensional vector.
#                  b) Saves the vector + the original text + metadata (source
#                     file, page number, chunk ID) into the Chroma collection.
#
#   3. QUERY   — Converts your search question into a vector and asks Chroma
#                "which stored vectors are closest to this?" — the closest ones
#                are the most semantically relevant chunks.
#
# Why vectors?
# ------------
# Instead of matching keywords, vectors capture *meaning*.  Searching for
# "campus location" can return results that mention "Chicago", "main campus",
# or "situated in the city" — even if those exact words weren't in the query.
#
# Dependencies:
#   pip install chromadb sentence-transformers pymupdf
# =============================================================================

import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Import our chunking pipeline from the parser we built in Phase 1
from parser import process_all_pdfs

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# The embedding model we use.
# 'all-MiniLM-L6-v2' is a great balance of speed and quality:
#   - Size  : ~80 MB  (downloads once, then cached locally)
#   - Speed : very fast on CPU
#   - Output: 384-dimensional vectors
# For a production medical app you might upgrade to 'all-mpnet-base-v2'
# (768 dims, slower) or a fine-tuned clinical model.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Where ChromaDB will persist its database files on disk.
# Using a subfolder keeps the project root tidy.
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"

# The "collection" is ChromaDB's equivalent of a database table.
# All our medical report chunks live in this one collection.
COLLECTION_NAME = "medical_reports"


# ---------------------------------------------------------------------------
# STEP 1 — Initialize ChromaDB Client and Embedding Model
# ---------------------------------------------------------------------------

def init_vector_store():
    """
    Set up the ChromaDB client (persistent, on-disk) and load the embedding
    model into memory.  Call this once at the start of your programme.

    Returns
    -------
    collection : chromadb.Collection
        The Chroma collection object where vectors are stored/queried.
    model : SentenceTransformer
        The loaded embedding model ready to encode text into vectors.
    """

    # --- 1a. Create the persistent ChromaDB client ---
    # PersistentClient stores data to disk so it survives between runs.
    # The first call creates the folder; subsequent calls just reopen it.
    print("[INIT]  Starting ChromaDB (persistent) ...")
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # --- 1b. Get (or create) the collection ---
    # get_or_create_collection is idempotent: safe to call many times.
    # We deliberately do NOT pass an embedding_function here because we
    # want to generate embeddings ourselves with SentenceTransformer — that
    # gives us more control (batching, custom models, etc.).
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        # cosine distance measures the angle between two vectors.
        # It's the best default for semantic text similarity.
        metadata={"hnsw:space": "cosine"},
    )
    print(f"[INIT]  Collection '{COLLECTION_NAME}' ready  "
          f"({collection.count()} documents already stored)")

    # --- 1c. Load the SentenceTransformer embedding model ---
    # The first run downloads the model from Hugging Face (~80 MB).
    # After that it's cached in your home directory and loads in ~1 second.
    print(f"[INIT]  Loading embedding model '{EMBEDDING_MODEL_NAME}' ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("[INIT]  Model loaded.\n")

    return collection, model


# ---------------------------------------------------------------------------
# STEP 2 — Ingest Chunks: Embed and Store in Chroma
# ---------------------------------------------------------------------------

def ingest_chunks(collection, model, force_reingest: bool = False):
    """
    Run the full ingestion pipeline:
      1. Call process_all_pdfs() from parser.py to get text chunks.
      2. Generate a vector embedding for each chunk's text.
      3. Upsert (insert or update) every chunk into the Chroma collection
         together with its metadata.

    Parameters
    ----------
    collection : chromadb.Collection
        The Chroma collection returned by init_vector_store().
    model : SentenceTransformer
        The embedding model returned by init_vector_store().
    force_reingest : bool
        If True, clears the collection first and re-embeds everything.
        Useful if you've added new PDFs or re-chunked existing ones.
        Default is False — skip ingestion if documents already exist.

    Returns
    -------
    int
        Number of chunks successfully stored.
    """

    # --- 2a. Guard: skip if already populated (unless forced) ---
    existing_count = collection.count()
    if existing_count > 0 and not force_reingest:
        print(f"[INGEST]  Skipping — collection already has {existing_count} "
              f"documents.  Pass force_reingest=True to re-embed.\n")
        return existing_count

    if force_reingest and existing_count > 0:
        print(f"[INGEST]  force_reingest=True — clearing {existing_count} "
              f"existing documents ...")
        # ChromaDB has no 'drop'; we delete all IDs instead.
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
        print("[INGEST]  Collection cleared.\n")

    # --- 2b. Get all chunks from parser.py ---
    print("[INGEST]  Extracting and chunking PDFs via parser.py ...")
    chunks = process_all_pdfs()

    if not chunks:
        print("[WARN]  No chunks returned from parser.  "
              "Make sure the data/ folder contains PDF files.")
        return 0

    print(f"[INGEST]  Got {len(chunks)} chunks.  Generating embeddings ...\n")

    # --- 2c. Prepare the four parallel lists ChromaDB expects ---
    # Chroma's add() / upsert() method takes:
    #   ids        — a unique string ID per document
    #   embeddings — the vector for each document
    #   documents  — the raw text (stored for retrieval)
    #   metadatas  — a dict of extra info per document

    ids        = []
    embeddings = []
    documents  = []
    metadatas  = []

    # Encode all chunk texts at once — SentenceTransformer handles batching
    # internally and is much faster than encoding one-by-one in a loop.
    texts = [chunk["text"] for chunk in chunks]

    # show_progress_bar=True gives a nice tqdm bar during encoding
    # model.encode() returns a numpy array of shape (n_chunks, 384).
    # We call .tolist() to convert it to plain Python lists, which is the
    # format ChromaDB's upsert() expects for the embeddings argument.
    # NOTE: convert_to_list was removed in sentence-transformers v5+.
    vectors = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,          # process 32 chunks at a time — tweak if needed
    ).tolist()

    for chunk, vector in zip(chunks, vectors):
        ids.append(chunk["chunk_id"])
        embeddings.append(vector)
        documents.append(chunk["text"])
        metadatas.append({
            # Store every useful field so we can filter/display them later
            "source_file": chunk["source_file"],
            "page_number": chunk["page_number"],   # int is fine for Chroma
            "chunk_index": chunk["chunk_index"],
            "word_count":  chunk["word_count"],
        })

    # --- 2d. Upsert into Chroma ---
    # upsert = insert if new, update if the same ID already exists.
    # This is safer than add() which would error on duplicate IDs.
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    final_count = collection.count()
    print(f"\n[INGEST]  Done.  {final_count} chunks now stored in "
          f"'{COLLECTION_NAME}'.\n")
    return final_count


# ---------------------------------------------------------------------------
# STEP 3 — Query: Semantic Similarity Search
# ---------------------------------------------------------------------------

def query_vector_store(
    collection,
    model,
    query_text: str,
    n_results: int = 3,
) -> list[dict]:
    """
    Search the vector store for chunks most semantically similar to the query.

    How it works:
      1. Encode `query_text` into a 384-dim vector using the same model.
      2. Ask Chroma to find the `n_results` stored vectors with the smallest
         cosine distance to the query vector.
      3. Return those chunks with their text and metadata.

    Parameters
    ----------
    collection : chromadb.Collection
        The Chroma collection to search.
    model : SentenceTransformer
        The same embedding model used during ingestion.
    query_text : str
        The question or phrase to search for.
    n_results : int
        How many top-matching chunks to return (default 3).

    Returns
    -------
    list[dict]
        Each dict:
        {
            "rank":        int,   # 1 = most relevant
            "score":       float, # cosine distance (lower = more similar)
            "chunk_id":    str,
            "source_file": str,
            "page_number": int,
            "text":        str,   # the retrieved chunk text
        }
    """
    if collection.count() == 0:
        print("[QUERY]  The collection is empty — run ingest_chunks() first.")
        return []

    # --- 3a. Embed the query using the exact same model ---
    # It's critical to use the same model for queries and documents,
    # otherwise the vector spaces won't align and results will be garbage.
    # Encode returns a numpy array; .tolist() converts it to a plain list
    # that ChromaDB's query() method can accept.
    query_vector = model.encode(query_text).tolist()

    # --- 3b. Ask Chroma for the nearest neighbours ---
    raw_results = collection.query(
        query_embeddings=[query_vector],  # Chroma expects a list-of-lists
        n_results=min(n_results, collection.count()),  # can't exceed total
        include=["documents", "metadatas", "distances"],
    )

    # --- 3c. Unpack Chroma's response into friendlier dicts ---
    # Chroma returns nested lists because you can send multiple queries at once.
    # We only sent one query, so everything is at index [0].
    results = []
    for rank, (doc, meta, dist, cid) in enumerate(zip(
        raw_results["documents"][0],
        raw_results["metadatas"][0],
        raw_results["distances"][0],
        raw_results["ids"][0],
    ), start=1):
        results.append({
            "rank":        rank,
            "score":       round(dist, 4),   # cosine distance: 0=identical, 2=opposite
            "chunk_id":    cid,
            "source_file": meta.get("source_file", "unknown"),
            "page_number": meta.get("page_number", -1),
            "word_count":  meta.get("word_count", 0),
            "text":        doc,
        })

    return results


# ---------------------------------------------------------------------------
# Helper — Pretty-print query results to the terminal
# ---------------------------------------------------------------------------

def print_query_results(query_text: str, results: list[dict], preview_chars: int = 400):
    """
    Print the retrieved chunks in a readable format for manual verification.

    Parameters
    ----------
    query_text : str
        The original search query (printed as a header).
    results : list[dict]
        Output of query_vector_store().
    preview_chars : int
        How many characters of each chunk's text to display.
    """
    print("\n" + "=" * 65)
    print(f"  QUERY : \"{query_text}\"")
    print(f"  HITS  : {len(results)} result(s)")
    print("=" * 65 + "\n")

    if not results:
        print("  [!]  No results found.  Is the collection populated?\n")
        return

    for r in results:
        print(f"+-- Rank #{r['rank']}  |  Score (cosine dist): {r['score']}")
        print(f"|   Chunk ID   : {r['chunk_id']}")
        print(f"|   Source     : {r['source_file']}  (page {r['page_number']})")
        print(f"|   Word count : {r['word_count']}")
        print("|")
        preview = r["text"][:preview_chars]
        if len(r["text"]) > preview_chars:
            preview += " ..."
        for line in preview.splitlines():
            print(f"|   {line}")
        print("+" + "-" * 63 + "\n")


# ---------------------------------------------------------------------------
# LOCAL EXECUTION BLOCK
# Run:  python vector_store.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "[Medical Report Summarizer] Phase 1: Vector Storage".center(65))
    print("=" * 65)

    # ------------------------------------------------------------------
    # Step A: Initialize ChromaDB + load embedding model
    # ------------------------------------------------------------------
    collection, model = init_vector_store()

    # ------------------------------------------------------------------
    # Step B: Ingest all PDFs from data/ into the vector store.
    # Set force_reingest=True if you update a PDF or add a new one.
    # ------------------------------------------------------------------
    ingest_chunks(collection, model, force_reingest=True)

    # ------------------------------------------------------------------
    # Step C: Run two test queries to verify retrieval works correctly.
    # ------------------------------------------------------------------

    # --- Test Query 1 ---
    q1 = "graduate programs"
    results1 = query_vector_store(collection, model, query_text=q1, n_results=2)
    print_query_results(q1, results1)

    # --- Test Query 2 ---
    q2 = "campus location"
    results2 = query_vector_store(collection, model, query_text=q2, n_results=2)
    print_query_results(q2, results2)

    print("[DONE]  Vector store test complete.")
    print("        The chroma_db/ folder now holds your persistent database.")
    print("        Import init_vector_store() + query_vector_store() in main.py"
          " to use this in the API.\n")
