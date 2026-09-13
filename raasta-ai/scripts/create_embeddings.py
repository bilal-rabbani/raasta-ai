
import json
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer


# =========================
# CONFIGURATION
# =========================

CHUNKS_FILE = Path("data/chunks.json")
OUTPUT_FILE = Path("data/knowledge_base.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"


# =========================
# LOAD CHUNKS
# =========================

print("Loading chunks...")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks.")


# =========================
# LOAD EMBEDDING MODEL
# =========================

print("\nLoading embedding model:")
print(MODEL_NAME)

model = SentenceTransformer(MODEL_NAME)

print("✅ Model loaded.")


# =========================
# CREATE EMBEDDINGS
# =========================

texts = [
    chunk["text"]
    for chunk in chunks
]

print("\nCreating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("✅ Embeddings created.")

print(
    "Embedding shape:",
    embeddings.shape
)


# =========================
# CREATE KNOWLEDGE BASE
# =========================

knowledge_base = {
    "model_name": MODEL_NAME,

    "embedding_dimension": int(
        embeddings.shape[1]
    ),

    "chunks": chunks,

    "embeddings": embeddings
}


# =========================
# SAVE
# =========================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "wb"
) as f:

    pickle.dump(
        knowledge_base,
        f
    )


# =========================
# SUMMARY
# =========================

file_size_mb = (
    OUTPUT_FILE.stat().st_size /
    (1024 * 1024)
)

print("\n" + "=" * 70)
print("EMBEDDING SUMMARY")
print("=" * 70)

print(f"Chunks              : {len(chunks)}")
print(f"Embedding dimension : {embeddings.shape[1]}")
print(f"Model               : {MODEL_NAME}")
print(f"Output file         : {OUTPUT_FILE}")
print(f"File size           : {file_size_mb:.2f} MB")

print("\n✅ Knowledge base created successfully.")
