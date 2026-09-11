
import json
import re
from pathlib import Path


# =========================
# CONFIGURATION
# =========================

SOURCES_FILE = Path("sources.json")
CLEANED_DIR = Path("data/cleaned")
OUTPUT_FILE = Path("data/chunks.json")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# =========================
# LOAD SOURCE METADATA
# =========================

with open(SOURCES_FILE, "r", encoding="utf-8") as f:
    sources = json.load(f)


source_lookup = {
    source["source_id"]: source
    for source in sources
}


# =========================
# TEXT NORMALIZATION
# =========================

def normalize_text(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# =========================
# CHUNK FUNCTION
# =========================

def create_chunks(text, chunk_size=1000, overlap=150):

    text = normalize_text(text)

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        # If paragraph fits
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:

            if current_chunk:
                current_chunk += "\n\n"

            current_chunk += paragraph

        else:

            if current_chunk:
                chunks.append(current_chunk)

            # Start next chunk
            if len(paragraph) <= chunk_size:
                current_chunk = paragraph

            else:
                # Handle very long paragraphs
                start = 0

                while start < len(paragraph):

                    end = start + chunk_size

                    piece = paragraph[start:end]

                    chunks.append(piece)

                    start = end - overlap

                current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# =========================
# PROCESS DOCUMENTS
# =========================

all_chunks = []

files = sorted(CLEANED_DIR.glob("*.txt"))

print(f"Found {len(files)} cleaned documents.")


for file_path in files:

    print("\n" + "=" * 70)
    print(f"Processing: {file_path.name}")

    source_id = file_path.stem

    # Read text
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    chunks = create_chunks(
        text,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )

    print(f"Created {len(chunks)} chunks.")

    # Get metadata
    source = source_lookup.get(source_id, {})

    for index, chunk in enumerate(chunks):

        chunk_id = f"{source_id}_chunk_{index:04d}"

        all_chunks.append({
            "chunk_id": chunk_id,

            "source_id": source_id,

            "institution": source.get(
                "institution",
                "Unknown"
            ),

            "title": source.get(
                "title",
                ""
            ),

            "url": source.get(
                "url",
                ""
            ),

            "jurisdiction": source.get(
                "jurisdiction",
                ""
            ),

            "source_type": source.get(
                "source_type",
                ""
            ),

            "chunk_index": index,

            "text": chunk
        })


# =========================
# SAVE CHUNKS
# =========================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_chunks,
        f,
        indent=2,
        ensure_ascii=False
    )


# =========================
# SUMMARY
# =========================

print("\n" + "=" * 70)
print("CHUNKING SUMMARY")
print("=" * 70)

print(f"Documents processed : {len(files)}")
print(f"Total chunks        : {len(all_chunks)}")
print(f"Chunk size          : {CHUNK_SIZE}")
print(f"Overlap             : {CHUNK_OVERLAP}")

print(f"\nSaved to:")
print(OUTPUT_FILE)
