from apify import Actor
import re
import json
import urllib.request
from typing import List, Any

SCRAPER_TEXT_FIELDS = [
    "text", "content", "body", "markdown", "html", "page_content",
    "pageContent", "description", "fullText", "full_text", "rawText",
    "raw_text", "extractedText", "extracted_text", "article", "post",
    "message", "review", "comment", "snippet", "summary", "details",
    "productDescription", "product_description", "jobDescription",
    "job_description", "about", "overview", "readme", "notes"
]

def fetch_dataset_items(dataset_id: str, token: str) -> List[dict]:
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?token={token}&clean=true&format=json"
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())
    if isinstance(data, list):
        return data
    return []

async def main() -> None:
    async with Actor:
        actor_input = await Actor.get_input() or {}
        chunk_size = actor_input.get("chunk_size", 1000)
        overlap    = actor_input.get("overlap", 100)
        items      = []

        # === CHAINING MODE: accept datasetId from previous actor ===
        dataset_id = actor_input.get("datasetId") or actor_input.get("dataset_id")
        if dataset_id and isinstance(dataset_id, str) and len(dataset_id) > 5:
            Actor.log.info(f"Chaining mode: loading dataset {dataset_id}...")
            token = actor_input.get("token") or Actor.get_env().get("token") or ""
            try:
                items = fetch_dataset_items(dataset_id, token)
                Actor.log.info(f"Loaded {len(items)} items from dataset.")
            except Exception as e:
                Actor.log.warning(f"Failed to fetch dataset: {e}")

        # === Also check resource chaining (Apify trigger format) ===
        if not items:
            resource = actor_input.get("resource") or {}
            res_dataset_id = resource.get("defaultDatasetId")
            if res_dataset_id:
                Actor.log.info(f"Resource chaining: loading dataset {res_dataset_id}...")
                token = actor_input.get("token") or Actor.get_env().get("token") or ""
                try:
                    items = fetch_dataset_items(res_dataset_id, token)
                    Actor.log.info(f"Loaded {len(items)} items from resource dataset.")
                except Exception as e:
                    Actor.log.warning(f"Failed to fetch resource dataset: {e}")

        # === DIRECT MODE: accept inline data array ===
        if not items:
            items = actor_input.get("data") or actor_input.get("items") or []
            if not isinstance(items, list):
                items = [items] if items else []

        # === FALLBACK: treat entire input as one item ===
        if not items and actor_input:
            items = [actor_input]

        Actor.log.info(f"Processing {len(items)} items...")
        processed_count = 0

        for idx, item in enumerate(items):
            raw_text = None
            for key in SCRAPER_TEXT_FIELDS:
                val = item.get(key)
                if val and isinstance(val, str) and len(val) > 0:
                    raw_text = val
                    break

            if not raw_text:
                raw_text = json.dumps(item, ensure_ascii=False, default=str)

            clean = clean_text_function(raw_text)
            chunks = chunk_text(clean, chunk_size=chunk_size, overlap=overlap)

            source_url = item.get("url") or item.get("sourceUrl") or item.get("link") or ""
            source_id  = str(item.get("id") or item.get("url") or f"item_{idx}")

            for i, chunk in enumerate(chunks):
                await Actor.push_data({
                    "original_id":        source_id,
                    "source_url":         source_url,
                    "chunk_index":        i,
                    "total_chunks":       len(chunks),
                    "chunk_text":         chunk,
                    "chunk_length_chars": len(chunk),
                    "cleaned_at":         str(Actor.get_env()["started_at"]),
                })
                processed_count += 1

        Actor.log.info(f"Done! Pushed {processed_count} chunks.")
        await Actor.set_status_message(f"Processed {processed_count} chunks from {len(items)} items")


def clean_text_function(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start  = 0
    while start < len(text):
        end   = start + chunk_size
        chunk = text[start:end]

        if end < len(text):
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.6:
                end = start + last_period + 1

        chunks.append(text[start:end].strip())
        start = end - overlap if end - overlap > start else end

    return [c for c in chunks if c]


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
