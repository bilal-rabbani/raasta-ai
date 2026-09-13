
import json
import time
from pathlib import Path

import requests


# =========================
# CONFIGURATION
# =========================

SOURCES_FILE = Path("sources.json")
RAW_DIR = Path("data/raw")
LOG_FILE = Path("data/download_log.json")

REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


# =========================
# SETUP
# =========================

RAW_DIR.mkdir(parents=True, exist_ok=True)

with open(SOURCES_FILE, "r", encoding="utf-8") as f:
    sources = json.load(f)

print(f"Loaded {len(sources)} sources from {SOURCES_FILE}")


# =========================
# DOWNLOAD FUNCTION
# =========================

def download_source(source):

    source_id = source["source_id"]
    url = source["url"]
    source_type = source.get("source_type", "html")

    print("\n" + "=" * 70)
    print(f"Source: {source_id}")
    print(f"Institution: {source.get('institution', 'Unknown')}")
    print(f"Type: {source_type}")
    print(f"URL: {url}")

    # Skip unverified sources
    if source.get("source_status") == "unverified":
        print("⚠️ Source marked UNVERIFIED. Skipping.")
        return {
            "source_id": source_id,
            "status": "unverified_skipped",
            "url": url,
        }

    # Determine extension
    if source_type.lower() == "pdf" or url.lower().split("?")[0].endswith(".pdf"):
        extension = ".pdf"
    else:
        extension = ".html"

    output_file = RAW_DIR / f"{source_id}{extension}"

    # Skip existing files
    if output_file.exists() and output_file.stat().st_size > 0:
        print(
            f"⏭️ Already exists: {output_file} "
            f"({output_file.stat().st_size:,} bytes)"
        )

        return {
            "source_id": source_id,
            "status": "already_exists",
            "url": url,
            "file": str(output_file),
        }

    # Try multiple times
    for attempt in range(1, MAX_RETRIES + 1):

        try:
            print(f"Attempt {attempt}/{MAX_RETRIES}...")

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            )

            response.raise_for_status()

            # Save PDF
            if extension == ".pdf":

                with open(output_file, "wb") as f:
                    f.write(response.content)

            # Save HTML
            else:

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(response.text)

            file_size = output_file.stat().st_size

            print(f"✅ HTTP {response.status_code}")
            print(
                f"Saved: {output_file} "
                f"({file_size:,} bytes)"
            )

            return {
                "source_id": source_id,
                "status": "downloaded",
                "url": url,
                "http_status": response.status_code,
                "file": str(output_file),
                "size_bytes": file_size,
            }

        except requests.exceptions.SSLError as e:

            print(f"❌ SSL certificate error: {e}")

            return {
                "source_id": source_id,
                "status": "ssl_error",
                "url": url,
                "error": str(e),
            }

        except requests.exceptions.Timeout as e:

            print(f"⏱️ Timeout on attempt {attempt}")

            if attempt < MAX_RETRIES:
                print(f"Waiting {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                return {
                    "source_id": source_id,
                    "status": "timeout",
                    "url": url,
                    "error": str(e),
                }

        except requests.exceptions.HTTPError as e:

            print(f"❌ HTTP error: {e}")

            return {
                "source_id": source_id,
                "status": "http_error",
                "url": url,
                "http_status": (
                    response.status_code
                    if "response" in locals()
                    else None
                ),
                "error": str(e),
            }

        except requests.exceptions.RequestException as e:

            print(f"❌ Request error: {e}")

            if attempt < MAX_RETRIES:
                print(f"Waiting {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                return {
                    "source_id": source_id,
                    "status": "request_error",
                    "url": url,
                    "error": str(e),
                }

        except Exception as e:

            print(f"❌ Unexpected error: {e}")

            return {
                "source_id": source_id,
                "status": "unexpected_error",
                "url": url,
                "error": str(e),
            }


# =========================
# RUN SCRAPER
# =========================

results = []

for index, source in enumerate(sources, start=1):

    print(f"\n[{index}/{len(sources)}]")

    result = download_source(source)

    results.append(result)


# =========================
# SAVE LOG
# =========================

with open(LOG_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)


# =========================
# SUMMARY
# =========================

downloaded = sum(
    r["status"] == "downloaded"
    for r in results
)

already_exists = sum(
    r["status"] == "already_exists"
    for r in results
)

unverified = sum(
    r["status"] == "unverified_skipped"
    for r in results
)

failed = len(results) - downloaded - already_exists - unverified

print("\n" + "=" * 70)
print("SCRAPER SUMMARY")
print("=" * 70)

print(f"Total sources       : {len(results)}")
print(f"Downloaded          : {downloaded}")
print(f"Already existed     : {already_exists}")
print(f"Unverified skipped  : {unverified}")
print(f"Failed              : {failed}")

print(f"\nLog saved to: {LOG_FILE}")
