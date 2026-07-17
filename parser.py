# =============================================================================
# parser.py — Phase 1: Document Ingestion & Chunking
# =============================================================================
# What this file does (in plain English):
#   1. Scans the `data/` folder for PDF files.
#   2. Opens each PDF with PyMuPDF and reads the text page-by-page.
#   3. Splits the full text into overlapping chunks (~300-500 tokens each)
#      using a paragraph-aware strategy so that medical context is never
#      cut in the middle of a sentence.
#   4. Returns a list of chunk dictionaries ready to be embedded / stored.
#
# Dependencies:
#   pip install pymupdf
# =============================================================================

import os           # For walking the file system
import re           # For cleaning whitespace
from pathlib import Path  # A nicer, cross-platform way to work with paths

import fitz         # PyMuPDF — "fitz" is the historical name of the library

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Where all PDF (and future DOCX/TXT) files live
DATA_DIR = Path(__file__).parent / "data"

# Approximate target size for each chunk (measured in whitespace-delimited
# "words", which is close enough to tokens for our purposes at this stage).
CHUNK_TARGET_WORDS = 400   # aim for the middle of the 300-500 range

# How many words from the END of the previous chunk to repeat at the START of
# the next chunk.  This overlap ensures that a medical concept that straddles
# two chunks appears in full in at least one of them.
CHUNK_OVERLAP_WORDS = 50


# ---------------------------------------------------------------------------
# STEP 1 — PDF Text Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Open a PDF file and return the text of every page as a list of dicts.

    Each dict has the shape:
        {
            "page_number": int,   # 1-based page number (human-friendly)
            "text":        str,   # raw text extracted from that page
        }

    Parameters
    ----------
    pdf_path : str or Path
        Absolute or relative path to the .pdf file.

    Returns
    -------
    list[dict]
        One entry per page.  Pages with no extractable text are skipped.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages_data = []

    # fitz.open() loads the PDF into memory.
    # We use a context manager so the file is closed automatically.
    with fitz.open(str(pdf_path)) as doc:
        print(f"  [PDF]  Opened '{pdf_path.name}'  ({len(doc)} pages)")

        for page_index, page in enumerate(doc):
            # get_text("text") extracts plain text; "blocks" or "html" give
            # more structure but plain text is simplest for chunking.
            raw_text = page.get_text("text")

            # Skip entirely blank pages (scanned images without an OCR layer)
            if not raw_text.strip():
                continue

            pages_data.append({
                "page_number": page_index + 1,   # convert 0-based → 1-based
                "text": raw_text,
            })

    return pages_data


# ---------------------------------------------------------------------------
# STEP 2 — Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Lightly normalise extracted PDF text.

    PDF extraction often produces:
      • Multiple consecutive blank lines  → collapse to one
      • Hyphenated line-breaks            → rejoin the word
      • Trailing spaces on every line     → strip them

    We deliberately keep paragraph breaks (double newlines) intact because
    the chunker below uses them as natural split points.

    Parameters
    ----------
    text : str
        Raw string from get_text().

    Returns
    -------
    str
        Cleaned string.
    """
    # Re-join words split across lines with a hyphen  (e.g. "med-\nical")
    text = re.sub(r"-\n", "", text)

    # Remove trailing whitespace from every line
    text = "\n".join(line.rstrip() for line in text.splitlines())

    # Collapse 3+ consecutive blank lines into exactly 2 (one paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# STEP 3 — Paragraph-Aware Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    source_file: str,
    page_number: int,
    target_words: int = CHUNK_TARGET_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[dict]:
    """
    Split a page's text into overlapping chunks using paragraph boundaries.

    Strategy
    --------
    1. Split the page text on blank lines → list of paragraphs.
    2. Accumulate paragraphs into a "current chunk" until adding the next
       paragraph would exceed `target_words`.
    3. When the limit is reached, save the current chunk, then seed the next
       chunk with the last `overlap_words` words of the saved chunk.
    4. Repeat until all paragraphs are consumed.

    Why paragraphs?
    ---------------
    Medical reports are structured in paragraphs: Findings, Impressions,
    History, etc.  Splitting at paragraph boundaries avoids cutting a clinical
    observation in the middle of a sentence.

    Parameters
    ----------
    text : str
        Cleaned text for a single page (or the full document).
    source_file : str
        Name of the source PDF — stored in metadata for traceability.
    page_number : int
        1-based page number — also stored in metadata.
    target_words : int
        Soft upper limit on words per chunk.
    overlap_words : int
        Number of words to repeat at the start of the next chunk.

    Returns
    -------
    list[dict]
        Each dict:
        {
            "chunk_id":    str,   # unique identifier  "filename_p1_c0"
            "source_file": str,
            "page_number": int,
            "chunk_index": int,   # 0-based index within this page
            "word_count":  int,
            "text":        str,   # the actual chunk text
        }
    """
    # --- 3a. Split into paragraphs ---
    # A "paragraph" is a block of text separated by at least one blank line.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []          # finished chunks go here
    current_words = []   # words accumulated so far in the current chunk
    chunk_index = 0      # 0-based counter for chunks on this page

    # Build a clean stem for chunk IDs (remove extension, spaces → underscores)
    file_stem = Path(source_file).stem.replace(" ", "_")

    for paragraph in paragraphs:
        # Turn the paragraph into a flat list of words
        para_words = paragraph.split()

        # --- 3b. Check if adding this paragraph would overflow ---
        if current_words and (len(current_words) + len(para_words) > target_words):
            # --- 3c. Save the current chunk ---
            chunk_text_str = " ".join(current_words)
            chunks.append({
                "chunk_id":    f"{file_stem}_p{page_number}_c{chunk_index}",
                "source_file": source_file,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "word_count":  len(current_words),
                "text":        chunk_text_str,
            })
            chunk_index += 1

            # --- 3d. Seed the next chunk with the overlap tail ---
            # Take the LAST `overlap_words` words of the chunk we just saved.
            current_words = current_words[-overlap_words:]

        # Add the paragraph's words to the current accumulator
        current_words.extend(para_words)

    # --- 3e. Don't forget the final (possibly under-sized) chunk ---
    if current_words:
        chunk_text_str = " ".join(current_words)
        chunks.append({
            "chunk_id":    f"{file_stem}_p{page_number}_c{chunk_index}",
            "source_file": source_file,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "word_count":  len(current_words),
            "text":        chunk_text_str,
        })

    return chunks


# ---------------------------------------------------------------------------
# STEP 4 — High-Level Pipeline: Process All PDFs in data/
# ---------------------------------------------------------------------------

def process_all_pdfs(data_dir: str | Path = DATA_DIR) -> list[dict]:
    """
    Scan `data_dir` for PDF files, extract and chunk every page, and return
    a flat list of all chunk dicts ready for downstream embedding.

    Parameters
    ----------
    data_dir : str or Path
        Directory to scan.  Defaults to the `data/` folder next to this file.

    Returns
    -------
    list[dict]
        All chunks from all pages of all PDFs, in order.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"data/ directory not found: {data_dir}")

    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"[WARN]  No PDF files found in {data_dir}")
        return []

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path.name}")
        print(f"{'='*60}")

        # Extract page-by-page text
        pages = extract_text_from_pdf(pdf_path)
        print(f"  [OK]  Extracted text from {len(pages)} pages")

        file_chunks = []

        for page_info in pages:
            # Clean the raw text
            cleaned = clean_text(page_info["text"])

            # Chunk this page's text
            page_chunks = chunk_text(
                text=cleaned,
                source_file=pdf_path.name,
                page_number=page_info["page_number"],
            )
            file_chunks.extend(page_chunks)

        print(f"  [CHUNKS]  Created {len(file_chunks)} chunks from '{pdf_path.name}'")
        all_chunks.extend(file_chunks)

    print(f"\n[DONE]  Total chunks across all PDFs: {len(all_chunks)}")
    return all_chunks


# ---------------------------------------------------------------------------
# STEP 5 — Process a SINGLE PDF (helper used in the test block below)
# ---------------------------------------------------------------------------

def process_single_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Convenience wrapper: extract + chunk one specific PDF file.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF file.

    Returns
    -------
    list[dict]
        All chunks from this PDF.
    """
    pdf_path = Path(pdf_path)

    print(f"\n{'='*60}")
    print(f"Processing single file: {pdf_path.name}")
    print(f"{'='*60}")

    pages = extract_text_from_pdf(pdf_path)
    print(f"  [OK]  Extracted text from {len(pages)} pages")

    all_chunks = []

    for page_info in pages:
        cleaned = clean_text(page_info["text"])
        page_chunks = chunk_text(
            text=cleaned,
            source_file=pdf_path.name,
            page_number=page_info["page_number"],
        )
        all_chunks.extend(page_chunks)

    print(f"  [CHUNKS]  Created {len(all_chunks)} chunks\n")
    return all_chunks


# ---------------------------------------------------------------------------
# LOCAL EXECUTION BLOCK
# Run:  python parser.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Path to the specific test PDF mentioned in the project brief
    TEST_PDF = DATA_DIR / "Illinois Tech - Chicago, USA - Graduate Fast Facts_2026 (1).pdf"

    print("\n" + "[Medical Report Summarizer] Phase 1: Document Ingestion".center(60))
    print("=" * 60)

    # Process the test PDF
    chunks = process_single_pdf(TEST_PDF)

    # -----------------------------------------------------------------------
    # Print a formatted preview of every chunk so we can visually verify that:
    #   • Each chunk is ~300-500 words
    #   • The overlap is working (the start of chunk N+1 echoes the end of N)
    #   • Medical context is not cut mid-sentence
    # -----------------------------------------------------------------------
    PREVIEW_CHARS = 300   # How many characters of each chunk to display

    print(f"\n" + "-" * 60)
    print(f"  CHUNK PREVIEW  ({len(chunks)} total chunks)")
    print("-" * 60 + "\n")

    for i, chunk in enumerate(chunks):
        print(f"+-- Chunk {i+1:>3}  |  ID: {chunk['chunk_id']}")
        print(f"|   Page: {chunk['page_number']}  |  Words: {chunk['word_count']}")
        print("|")

        # Show the first PREVIEW_CHARS characters of the chunk text
        preview = chunk["text"][:PREVIEW_CHARS]
        if len(chunk["text"]) > PREVIEW_CHARS:
            preview += " ..."

        # Indent each line of the preview for readability
        for line in preview.splitlines():
            print(f"|   {line}")

        print("+" + "-" * 58 + "\n")

    print(f"\n[DONE]  {len(chunks)} chunks ready for embedding.")
    print(f"   Next step: pass `chunks` to your embedding model (Phase 2).\n")
