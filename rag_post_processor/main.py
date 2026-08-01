from apify import Actor
import math
import re
import json
import urllib.request
from typing import List, Any


class ChunkCountExceeded(Exception):
    """Raised when the chunker produces far more chunks than a sane bound
    for the input length and chunk_size allow. Defense in depth against
    runaway output/compute for a single item (e.g. an infinite loop from
    chunk_size<=0), independent of input-schema validation (which never
    sees values arriving via the datasetId chaining path). Billing is no
    longer tied to chunk count - see charge_for_input - so this guards
    compute and output size, not cost."""

    def __init__(self, actual: int, expected: int, chunk_size: int, overlap: int):
        self.actual = actual
        self.expected = expected
        self.chunk_size = chunk_size
        self.overlap = overlap
        super().__init__(
            f"chunk count {actual} exceeded expected bound {expected} "
            f"(chunk_size={chunk_size}, overlap={overlap})"
        )


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


INPUT_KB_EVENT = "input-kb-processed"


def compute_input_bytes(items: List[Any]) -> int:
    """Raw size of the input as received, in bytes, before any cleaning or
    chunking. Sums each item's own JSON-serialized byte length, so inline
    data/items and items fetched via the datasetId chaining path are
    measured identically - whichever path populated `items`, this is the
    same computation over the same list."""
    return sum(
        len(json.dumps(item, ensure_ascii=False, default=str).encode("utf-8"))
        for item in items
    )


def compute_billed_units(total_bytes: int) -> int:
    """units = max(1, ceil(bytes / 1024)); 0 bytes -> 0 units (no charge)."""
    if total_bytes <= 0:
        return 0
    return max(1, math.ceil(total_bytes / 1024))


async def charge_for_input(items: List[Any]):
    """The single place input-kb-processed is charged, based purely on the
    raw byte size of `items` as received - not on anything the chunker
    later produces. Must run before any cleaning/chunking/compute. Returns
    (units_charged, charge_result); charge_result is None when there was
    nothing to charge for (empty input)."""
    units = compute_billed_units(compute_input_bytes(items))
    if units == 0:
        return units, None
    result = await Actor.charge(event_name=INPUT_KB_EVENT, count=units)
    return units, result

def resolve_chunk_params(actor_input: dict) -> tuple[int, int, int]:
    chunk_size      = actor_input.get("chunk_size", 1000)
    overlap         = actor_input.get("overlap", 100)
    min_chunk_chars = actor_input.get("min_chunk_chars", 50)

    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        Actor.log.warning(
            f"Invalid chunk_size {chunk_size!r} (expected a positive integer); "
            f"falling back to default 1000."
        )
        chunk_size = 1000

    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        Actor.log.warning(
            f"Invalid overlap {overlap!r} (expected a non-negative integer); "
            f"falling back to default 100."
        )
        overlap = 100

    if not isinstance(min_chunk_chars, int) or isinstance(min_chunk_chars, bool) or min_chunk_chars < 0:
        Actor.log.warning(
            f"Invalid min_chunk_chars {min_chunk_chars!r} (expected a non-negative "
            f"integer); falling back to default 50."
        )
        min_chunk_chars = 50

    max_overlap = chunk_size // 2
    if overlap > max_overlap:
        Actor.log.warning(
            f"overlap ({overlap}) exceeds 50% of chunk_size ({chunk_size}); "
            f"a large overlap causes excessive near-duplicate chunks. "
            f"Requested overlap={overlap}, applying clamped overlap={max_overlap} "
            f"(chunk_size // 2)."
        )
        overlap = max_overlap

    return chunk_size, overlap, min_chunk_chars


async def main() -> None:
    async with Actor:
        actor_input = await Actor.get_input() or {}
        chunk_size, overlap, min_chunk_chars = resolve_chunk_params(actor_input)
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

        # Charge exactly once, before any cleaning/chunking, based only on
        # the raw bytes received - not on chunk count. Respect the run's
        # spend cap before doing any billable work.
        units_charged, charge_result = await charge_for_input(items)
        if charge_result is not None and getattr(charge_result, "event_charge_limit_reached", False):
            Actor.log.warning(
                f"This input requires {units_charged} unit(s) of '{INPUT_KB_EVENT}', "
                f"which exceeds this run's configured max cost per run. Emitting "
                f"nothing and exiting without processing any items."
            )
            await Actor.set_status_message(
                f"Stopped: input requires {units_charged} unit(s), exceeding this run's cost cap."
            )
            return

        Actor.log.info(f"Processing {len(items)} items ({units_charged} unit(s) charged for input size)...")
        processed_count = 0

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                Actor.log.warning(
                    f"Item {idx} is a {type(item).__name__}, not an object; "
                    f"expected a dict with a text field (e.g. 'text', 'content', 'html'). Skipping."
                )
                continue

            try:
                raw_text = None
                for key in SCRAPER_TEXT_FIELDS:
                    val = item.get(key)
                    if val and isinstance(val, str) and len(val) > 0:
                        raw_text = val
                        break

                if not raw_text:
                    Actor.log.warning(
                        f"Item {idx} has none of the recognized text fields "
                        f"({', '.join(SCRAPER_TEXT_FIELDS[:5])}, ...); "
                        f"falling back to processing its raw JSON representation."
                    )
                    raw_text = json.dumps(item, ensure_ascii=False, default=str)

                clean = clean_text_function(raw_text)
                chunks = chunk_text(
                    clean, chunk_size=chunk_size, overlap=overlap, min_chunk_chars=min_chunk_chars
                )

                source_url = item.get("url") or item.get("sourceUrl") or item.get("link") or ""
                source_id  = str(item.get("id") or item.get("url") or f"item_{idx}")

                started_at = Actor.get_env()["started_at"]
                cleaned_at = started_at.isoformat() if hasattr(started_at, "isoformat") else str(started_at)

                for i, chunk in enumerate(chunks):
                    dataset_item = {
                        "original_id":        source_id,
                        "chunk_index":        i,
                        "total_chunks":       len(chunks),
                        "chunk_text":         chunk,
                        "chunk_length_chars": len(chunk),
                        "cleaned_at":         cleaned_at,
                    }
                    if source_url:
                        # Omitted (not emitted as "") when absent so the dataset
                        # schema's "link" format doesn't try to render an empty
                        # string as a URL.
                        dataset_item["source_url"] = source_url
                    await Actor.push_data(dataset_item)
                    processed_count += 1
            except ChunkCountExceeded as e:
                Actor.log.error(
                    f"Item {idx}: chunker produced {e.actual} chunks, exceeding the "
                    f"expected bound of {e.expected} for chunk_size={e.chunk_size}, "
                    f"overlap={e.overlap}. Aborting this item to avoid runaway output; "
                    f"no chunks were pushed for it. (Input was already charged in full "
                    f"under input-kb-processed, independent of this item's chunk count.)"
                )
                continue
            except Exception as e:
                Actor.log.warning(f"Skipping item {idx} due to unexpected error: {e}")
                continue

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


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100, min_chunk_chars: int = 50) -> List[str]:
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # Output-count invariant: a hard bound on how many chunks/loop iterations
    # a single input item may produce, independent of whatever chunk_size/
    # overlap were passed in. This is the last line of defense against
    # runaway output or an infinite loop (e.g. chunk_size<=0) for values
    # that never passed through input-schema validation - notably items
    # arriving via the datasetId chaining path at runtime. Chunk count no
    # longer affects billing (see charge_for_input), so this guards compute
    # and dataset size, not cost.
    max_expected = math.ceil(len(text) / max(1, chunk_size // 2)) + 2

    chunks = []
    starts = []
    start  = 0
    attempts = 0
    while start < len(text):
        attempts += 1
        if attempts > max_expected:
            raise ChunkCountExceeded(attempts, max_expected, chunk_size, overlap)

        end   = start + chunk_size
        chunk = text[start:end]

        if end < len(text):
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.6:
                end = start + last_period + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
            starts.append(start)
        start = end - overlap if end - overlap > start else end

    # Merge (never drop) a too-short trailing chunk into the previous one,
    # so no text ever becomes unsearchable in the customer's vector index.
    if len(chunks) > 1 and len(chunks[-1]) < min_chunk_chars:
        merge_from = starts[-2]
        chunks[-2] = text[merge_from:].strip()
        chunks.pop()
        starts.pop()

    return chunks


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
