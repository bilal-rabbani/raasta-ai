import json
import pickle
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
import pdfplumber
from sentence_transformers import SentenceTransformer

from clean_sources import clean_text  # reuse your existing text-cleaning logic
from chunk_sources import create_chunks  # reuse your existing chunking logic


# =========================
# CONFIGURATION
# =========================

# Where the NEW, not-yet-processed sources live (separate from data/raw)
NEW_SOURCES_DIR = Path("data/new_sources")
NEW_SOURCES_META = NEW_SOURCES_DIR / "sources.json"
NEW_RAW_DIR = NEW_SOURCES_DIR / "raw"

# Existing pipeline outputs — these get APPENDED to, never overwritten wholesale
MASTER_SOURCES_FILE = Path("sources.json")
CLEANED_DIR = Path("data/cleaned")
CHUNKS_FILE = Path("data/chunks.json")
KNOWLEDGE_BASE_FILE = Path("data/knowledge_base.pkl")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MODEL_NAME = "all-MiniLM-L6-v2"


# =========================
# LOAD NEW SOURCE METADATA
# =========================

def load_new_sources_metadata():
    """
    Reads data/new_sources/sources.json. If it's missing, falls back to
    inferring minimal metadata from filenames in data/new_sources/raw/
    so the script still runs, but you should add a proper sources.json
    for correct institution/title/url/jurisdiction fields.
    """
    if NEW_SOURCES_META.exists():
        with open(NEW_SOURCES_META, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"⚠️ {NEW_SOURCES_META} not found — inferring minimal metadata from filenames.")
    inferred = []
    for file_path in sorted(NEW_RAW_DIR.iterdir()):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        source_type = "pdf" if suffix == ".pdf" else ("md" if suffix == ".md" else "html")
        inferred.append({
            "source_id": file_path.stem,
            "institution": "Unknown",
            "title": file_path.stem,
            "url": "",
            "jurisdiction": "",
            "source_type": source_type,
        })
    return inferred


# =========================
# HTML / PDF EXTRACTION
# (mirrors clean_sources.py's logic)
# =========================

def extract_html(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "svg", "iframe", "nav", "footer", "header"]):
        tag.decompose()

    main_content = soup.find("main") or soup.find("article") or soup.find("body")
    if main_content is None:
        return ""

    text = main_content.get_text(separator="\n", strip=True)
    return clean_text(text)


def extract_md(file_path):
    """
    Markdown / plain-text files need no HTML stripping — just read and
    run through the same whitespace/control-character cleanup.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return clean_text(text)


def extract_pdf(file_path):
    pages_text = []
    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text()
                if text:
                    pages_text.append(f"[PAGE {page_number}]\n{text}")
            except Exception as e:
                print(f"⚠️ Could not extract page {page_number} of {file_path.name}: {e}")
    return clean_text("\n\n".join(pages_text))


# =========================
# STEP 1: CLEAN NEW RAW FILES → data/cleaned/
# =========================

def clean_new_sources(new_sources):
    """
    Cleans each new raw file and writes it into the SAME data/cleaned/
    folder your existing pipeline uses, so future full rebuilds stay
    consistent. Skips a file if its cleaned .txt already exists (i.e.
    this source was already processed in a previous incremental run).
    """
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    newly_cleaned_ids = []

    for source in new_sources:
        source_id = source["source_id"]
        source_type = source.get("source_type", "html").lower()
        if source_type == "pdf":
            extension = ".pdf"
        elif source_type == "md":
            extension = ".md"
        else:
            extension = ".html"
        raw_file = NEW_RAW_DIR / f"{source_id}{extension}"
        cleaned_file = CLEANED_DIR / f"{source_id}.txt"

        if cleaned_file.exists():
            print(f"⏭️ Already cleaned, skipping: {source_id}")
            continue

        if not raw_file.exists():
            print(f"❌ Raw file not found for source_id={source_id}: {raw_file}")
            continue

        print(f"Cleaning: {raw_file.name}")
        if extension == ".pdf":
            text = extract_pdf(raw_file)
        elif extension == ".md":
            text = extract_md(raw_file)
        else:
            text = extract_html(raw_file)

        with open(cleaned_file, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✅ Saved: {cleaned_file} ({len(text):,} characters)")
        newly_cleaned_ids.append(source_id)

    return newly_cleaned_ids


# =========================
# STEP 2: CHUNK ONLY THE NEW CLEANED FILES
# =========================

def chunk_new_sources(newly_cleaned_ids, source_lookup):
    """
    Builds chunk records only for the sources that were just cleaned,
    using the same create_chunks() logic as chunk_sources.py.
    """
    new_chunks = []

    for source_id in newly_cleaned_ids:
        cleaned_file = CLEANED_DIR / f"{source_id}.txt"
        with open(cleaned_file, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = create_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
        source = source_lookup.get(source_id, {})

        for index, chunk in enumerate(chunks):
            new_chunks.append({
                "chunk_id": f"{source_id}_chunk_{index:04d}",
                "source_id": source_id,
                "institution": source.get("institution", "Unknown"),
                "title": source.get("title", ""),
                "url": source.get("url", ""),
                "jurisdiction": source.get("jurisdiction", ""),
                "source_type": source.get("source_type", ""),
                "chunk_index": index,
                "text": chunk,
            })

        print(f"Created {len(chunks)} chunks for {source_id}")

    return new_chunks


# =========================
# STEP 3: APPEND TO chunks.json
# =========================

def append_to_chunks_file(new_chunks):
    if CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)
    else:
        existing_chunks = []

    existing_ids = {c["chunk_id"] for c in existing_chunks}
    to_add = [c for c in new_chunks if c["chunk_id"] not in existing_ids]
    skipped = len(new_chunks) - len(to_add)
    if skipped:
        print(f"⏭️ Skipped {skipped} chunk(s) already present in {CHUNKS_FILE}")

    merged = existing_chunks + to_add

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"✅ Appended {len(to_add)} new chunk(s) → {CHUNKS_FILE} (total now {len(merged)})")
    return to_add


# =========================
# STEP 4: EMBED ONLY THE NEW CHUNKS, APPEND TO knowledge_base.pkl
# =========================

def embed_and_append(new_chunks_to_add):
    if not new_chunks_to_add:
        print("No new chunks to embed.")
        return

    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in new_chunks_to_add]
    print(f"Creating embeddings for {len(texts)} new chunk(s)...")
    new_embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    if KNOWLEDGE_BASE_FILE.exists():
        with open(KNOWLEDGE_BASE_FILE, "rb") as f:
            kb = pickle.load(f)

        if kb.get("model_name") != MODEL_NAME:
            raise ValueError(
                f"Existing knowledge base was built with model '{kb.get('model_name')}', "
                f"but this script uses '{MODEL_NAME}'. Re-embedding everything is required "
                f"if you want to change models — mixing embedding spaces will break search."
            )

        existing_ids = {c["chunk_id"] for c in kb["chunks"]}
        keep_mask = [c["chunk_id"] not in existing_ids for c in new_chunks_to_add]

        filtered_chunks = [c for c, keep in zip(new_chunks_to_add, keep_mask) if keep]
        filtered_embeddings = new_embeddings[np.array(keep_mask)]

        kb["chunks"] = kb["chunks"] + filtered_chunks
        kb["embeddings"] = np.vstack([kb["embeddings"], filtered_embeddings]) if len(filtered_embeddings) else kb["embeddings"]
    else:
        kb = {
            "model_name": MODEL_NAME,
            "embedding_dimension": int(new_embeddings.shape[1]),
            "chunks": new_chunks_to_add,
            "embeddings": new_embeddings,
        }

    KNOWLEDGE_BASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(KNOWLEDGE_BASE_FILE, "wb") as f:
        pickle.dump(kb, f)

    print(f"✅ Knowledge base now has {len(kb['chunks'])} total chunks → {KNOWLEDGE_BASE_FILE}")


# =========================
# STEP 5 (optional): MERGE NEW METADATA INTO MASTER sources.json
# =========================

def merge_into_master_sources(new_sources):
    if MASTER_SOURCES_FILE.exists():
        with open(MASTER_SOURCES_FILE, "r", encoding="utf-8") as f:
            master = json.load(f)
    else:
        master = []

    existing_ids = {s["source_id"] for s in master}
    to_add = [s for s in new_sources if s["source_id"] not in existing_ids]

    if not to_add:
        print("No new entries to merge into master sources.json.")
        return

    merged = master + to_add
    with open(MASTER_SOURCES_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"✅ Merged {len(to_add)} new source(s) into {MASTER_SOURCES_FILE}")


# =========================
# MAIN
# =========================

def main():
    print("=" * 70)
    print("INCREMENTAL: PROCESSING NEW SOURCES FOLDER")
    print("=" * 70)

    if not NEW_RAW_DIR.exists():
        print(f"❌ {NEW_RAW_DIR} does not exist. Nothing to do.")
        return

    new_sources = load_new_sources_metadata()
    source_lookup = {s["source_id"]: s for s in new_sources}
    print(f"Found {len(new_sources)} new source(s) to consider.")

    newly_cleaned_ids = clean_new_sources(new_sources)
    if not newly_cleaned_ids:
        print("\nNo new files were cleaned (all already processed, or none found). Done.")
        return

    new_chunks = chunk_new_sources(newly_cleaned_ids, source_lookup)
    chunks_added = append_to_chunks_file(new_chunks)
    embed_and_append(chunks_added)
    merge_into_master_sources(new_sources)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"New sources cleaned : {len(newly_cleaned_ids)}")
    print(f"New chunks added    : {len(chunks_added)}")


if __name__ == "__main__":
    main()
