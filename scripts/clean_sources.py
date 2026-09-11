
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
import pdfplumber


# =========================
# CONFIGURATION
# =========================

RAW_DIR = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")
LOG_FILE = Path("data/cleaning_log.json")

CLEANED_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# TEXT CLEANING
# =========================

def clean_text(text):
    """
    Basic cleanup:
    - Normalize whitespace
    - Remove repeated blank lines
    - Remove obvious control characters
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null/control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Clean spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


# =========================
# HTML EXTRACTION
# =========================

def extract_html(file_path):

    print(f"Reading HTML: {file_path.name}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "lxml")

    # Remove elements that normally contain navigation/boilerplate
    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "iframe",
        "nav",
        "footer",
        "header"
    ]):
        tag.decompose()

    # Try to identify the main content
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find("body")
    )

    if main_content is None:
        return ""

    text = main_content.get_text(
        separator="\n",
        strip=True
    )

    return clean_text(text)


# =========================
# PDF EXTRACTION
# =========================

def extract_pdf(file_path):

    print(f"Reading PDF: {file_path.name}")

    pages_text = []

    with pdfplumber.open(file_path) as pdf:

        print(f"Pages: {len(pdf.pages)}")

        for page_number, page in enumerate(pdf.pages, start=1):

            try:
                text = page.extract_text()

                if text:
                    pages_text.append(
                        f"[PAGE {page_number}]\n{text}"
                    )

            except Exception as e:

                print(
                    f"⚠️ Could not extract page "
                    f"{page_number}: {e}"
                )

    return clean_text("\n\n".join(pages_text))


# =========================
# PROCESS FILES
# =========================

results = []

files = sorted(RAW_DIR.iterdir())

print(f"Found {len(files)} files in {RAW_DIR}")

for file_path in files:

    if not file_path.is_file():
        continue

    extension = file_path.suffix.lower()

    try:

        if extension == ".html":

            text = extract_html(file_path)

        elif extension == ".pdf":

            text = extract_pdf(file_path)

        else:

            print(
                f"⏭️ Skipping unsupported file: "
                f"{file_path.name}"
            )

            continue

        # Output filename
        output_file = (
            CLEANED_DIR /
            f"{file_path.stem}.txt"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(text)

        character_count = len(text)
        word_count = len(text.split())

        print(
            f"✅ Saved: {output_file} "
            f"({character_count:,} characters, "
            f"{word_count:,} words)"
        )

        results.append({
            "source_file": str(file_path),
            "cleaned_file": str(output_file),
            "status": "success",
            "characters": character_count,
            "words": word_count
        })

    except Exception as e:

        print(
            f"❌ Failed: {file_path.name}"
        )
        print(f"   Error: {e}")

        results.append({
            "source_file": str(file_path),
            "status": "failed",
            "error": str(e)
        })


# =========================
# SAVE LOG
# =========================

with open(
    LOG_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


# =========================
# SUMMARY
# =========================

successful = sum(
    r["status"] == "success"
    for r in results
)

failed = sum(
    r["status"] == "failed"
    for r in results
)

print("\n" + "=" * 70)
print("CLEANING SUMMARY")
print("=" * 70)

print(f"Files found : {len(results)}")
print(f"Successful  : {successful}")
print(f"Failed      : {failed}")

print(f"\nCleaned files saved in: {CLEANED_DIR}")
print(f"Log saved to: {LOG_FILE}")
