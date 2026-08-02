"""RAG Post Processor - clean and chunk scraped text for retrieval pipelines.

Billing note: `input_kb_processed` is charged once per run, before any cleaning
or chunking, on the bytes of text this Actor actually reads. It is deliberately
not tied to chunk count, so no chunking setting can inflate a bill.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from apify import Actor

from .chunking import Chunk, chunk_document
from .cleaning import clean_text
from .fetching import fetch_dataset_items, redact
from .tokens import (
    DEFAULT_ENCODING,
    MAX_EMBEDDING_TOKENS,
    SUPPORTED_ENCODINGS,
    get_encoder,
    is_exact,
    load_errors,
    method_label,
)

INPUT_KB_EVENT = "input_kb_processed"

# Field names a scraper might put the document text in, in priority order.
# `markdown` first: it carries the structure this Actor is built to preserve.
# The SEO-ish fields (`description`, `summary`, `snippet`) are last, so an item
# that has both a 155-character meta description and a full body no longer
# silently indexes the meta description.
SCRAPER_TEXT_FIELDS = [
    "markdown", "text", "content", "body", "page_content", "pageContent",
    "html", "fullText", "full_text", "rawText", "raw_text", "extractedText",
    "extracted_text", "article", "post", "message", "review", "comment",
    "readme", "details", "productDescription", "product_description",
    "jobDescription", "job_description", "about", "overview", "notes",
    "description", "summary", "snippet",
]

URL_FIELDS = ["url", "sourceUrl", "source_url", "link", "loadedUrl", "canonicalUrl", "href"]
ID_FIELDS = ["id", "_id", "uuid", "guid", "objectId"]

DEFAULTS = {
    "max_tokens": 512,
    "overlap_tokens": 64,
    "min_tokens": 24,
    "encoding": DEFAULT_ENCODING,
    "split_on_heading_level": 0,
    "include_heading_context": True,
    "preserve_links": True,
    "drop_nav": True,
}


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

def _as_int(value: Any, default: int, *, minimum: int, maximum: int, label: str) -> int:
    # isinstance(True, int) is True in Python; reject bools explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        if value is not None:
            Actor.log.warning(
                f"Invalid {label} {value!r} (expected an integer); using default {default}."
            )
        return default
    if value < minimum:
        Actor.log.warning(f"{label}={value} is below the minimum {minimum}; using {minimum}.")
        return minimum
    if value > maximum:
        Actor.log.warning(f"{label}={value} is above the maximum {maximum}; using {maximum}.")
        return maximum
    return value


def _as_bool(value: Any, default: bool, label: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    Actor.log.warning(f"Invalid {label} {value!r} (expected true/false); using {default}.")
    return default


# v0.x sized chunks in characters. Anything already chained to this Actor, or
# any saved task, still sends those keys. Silently ignoring them would apply
# defaults and change behaviour without saying so, so they are translated and
# the translation is logged. Measured chars/token on cl100k_base ranges 0.9
# (Japanese) to 6.1 (English prose); 4.0 is the conventional English estimate
# and is what the old defaults were implicitly tuned against.
LEGACY_CHARS_PER_TOKEN = 4.0
LEGACY_KEYS = {
    "chunk_size": "max_tokens",
    "overlap": "overlap_tokens",
    "min_chunk_chars": "min_tokens",
}


def apply_legacy_input(actor_input: Dict[str, Any]) -> Dict[str, Any]:
    """Translate v0.x character-based keys into their token equivalents.

    An explicit new-style key always wins; the legacy key is only used when its
    modern counterpart is absent.
    """
    translated = dict(actor_input)
    for old_key, new_key in LEGACY_KEYS.items():
        value = actor_input.get(old_key)
        if value is None or new_key in actor_input:
            if value is not None:
                Actor.log.warning(
                    f"Both '{old_key}' (removed in v1.0) and '{new_key}' were provided; "
                    f"using '{new_key}' and ignoring '{old_key}'."
                )
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            Actor.log.warning(f"Ignoring invalid legacy '{old_key}' value {value!r}.")
            continue
        # Half-up, not Python's banker's rounding: round(12.5) is 12, which is
        # a surprising number to read in a log line.
        converted = max(0, int(value / LEGACY_CHARS_PER_TOKEN + 0.5))
        Actor.log.warning(
            f"'{old_key}' was removed in v1.0 - chunk sizes are now measured in tokens, "
            f"not characters. Translating {old_key}={value} characters to "
            f"{new_key}={converted} tokens (at ~{LEGACY_CHARS_PER_TOKEN:g} chars/token). "
            f"Set '{new_key}' directly to control this exactly."
        )
        translated[new_key] = converted
    return translated


def resolve_settings(actor_input: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalise every knob, logging any substitution made."""
    actor_input = apply_legacy_input(actor_input)
    max_tokens = _as_int(
        actor_input.get("max_tokens", DEFAULTS["max_tokens"]), DEFAULTS["max_tokens"],
        minimum=16, maximum=MAX_EMBEDDING_TOKENS, label="max_tokens",
    )
    overlap = _as_int(
        actor_input.get("overlap_tokens", DEFAULTS["overlap_tokens"]), DEFAULTS["overlap_tokens"],
        minimum=0, maximum=MAX_EMBEDDING_TOKENS, label="overlap_tokens",
    )
    # Clamped, not rejected. Unlike the previous implementation, the maximum
    # legal overlap is a fully supported configuration and never discards the
    # item.
    ceiling = max_tokens // 2
    if overlap > ceiling:
        Actor.log.warning(
            f"overlap_tokens ({overlap}) exceeds 50% of max_tokens ({max_tokens}); "
            f"clamping to {ceiling}. Above 50%, near-duplicate chunks dominate the index."
        )
        overlap = ceiling

    min_tokens = _as_int(
        actor_input.get("min_tokens", DEFAULTS["min_tokens"]), DEFAULTS["min_tokens"],
        minimum=0, maximum=max(0, max_tokens - 1), label="min_tokens",
    )

    encoding = actor_input.get("encoding") or DEFAULTS["encoding"]
    if encoding not in SUPPORTED_ENCODINGS:
        Actor.log.warning(
            f"Unknown encoding {encoding!r}; supported: {', '.join(SUPPORTED_ENCODINGS)}. "
            f"Using {DEFAULT_ENCODING}."
        )
        encoding = DEFAULT_ENCODING

    split_level = _as_int(
        actor_input.get("split_on_heading_level", DEFAULTS["split_on_heading_level"]),
        DEFAULTS["split_on_heading_level"], minimum=0, maximum=6,
        label="split_on_heading_level",
    )

    return {
        "max_tokens": max_tokens,
        "overlap_tokens": overlap,
        "min_tokens": min_tokens,
        "encoding": encoding,
        "split_on_heading_level": split_level,
        "include_heading_context": _as_bool(
            actor_input.get("include_heading_context"),
            DEFAULTS["include_heading_context"], "include_heading_context"),
        "preserve_links": _as_bool(
            actor_input.get("preserve_links"), DEFAULTS["preserve_links"], "preserve_links"),
        "drop_nav": _as_bool(actor_input.get("drop_nav"), DEFAULTS["drop_nav"], "drop_nav"),
        "text_field": actor_input.get("text_field") or None,
    }


def extract_text(item: Dict[str, Any], preferred: Optional[str]) -> Tuple[str, str]:
    """Return (text, field_name_used).

    Honours an explicit `text_field` first, then the priority list. The field
    actually used is emitted on every row, so a user can see whether the Actor
    read `markdown` when they meant `text`.
    """
    if preferred:
        val = item.get(preferred)
        if isinstance(val, str) and val.strip():
            return val, preferred
        Actor.log.warning(
            f"text_field={preferred!r} is missing or empty on this item; "
            f"falling back to automatic field detection."
        )
    for key in SCRAPER_TEXT_FIELDS:
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val, key
    return "", ""


def first_of(item: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


# ---------------------------------------------------------------------------
# Billing
# ---------------------------------------------------------------------------

def compute_billed_units(total_bytes: int) -> int:
    """units = ceil(bytes / 1024); 0 bytes -> 0 units (no charge)."""
    if total_bytes <= 0:
        return 0
    return max(1, math.ceil(total_bytes / 1024))


def compute_billable_bytes(texts: List[str]) -> int:
    """Bytes of text this Actor actually reads.

    Previously this summed each item's *entire* JSON, including fields the
    processing loop never touched. On a real Website Content Crawler item
    (text + metadata + screenshotUrl + debug + html) that billed 3395 bytes to
    process 572 - 5.9x. Charging on the extracted text is what the store
    listing already says ("per KB of input processed") and is a number the
    customer can verify from their own data.
    """
    return sum(len(t.encode("utf-8")) for t in texts)


async def charge_for_input(total_bytes: int):
    """The single place `input_kb_processed` is charged. Runs before any
    cleaning or chunking, and is independent of how many chunks result."""
    units = compute_billed_units(total_bytes)
    if units == 0:
        return units, None
    result = await Actor.charge(event_name=INPUT_KB_EVENT, count=units)
    return units, result


# ---------------------------------------------------------------------------
# Input gathering
# ---------------------------------------------------------------------------

async def gather_items(actor_input: Dict[str, Any]) -> Tuple[List[Any], Optional[str]]:
    """Resolve the four supported input paths. Returns (items, fatal_error)."""
    token = actor_input.get("token") or Actor.get_env().get("token") or ""

    for source, dataset_id in (
        ("input", actor_input.get("datasetId") or actor_input.get("dataset_id")),
        ("resource", (actor_input.get("resource") or {}).get("defaultDatasetId")),
    ):
        if isinstance(dataset_id, str) and len(dataset_id) > 5:
            Actor.log.info(f"Chaining mode ({source}): loading dataset {dataset_id}...")
            result = fetch_dataset_items(dataset_id, token, log=Actor.log.warning)
            if result.error:
                # A failed fetch is fatal, not something to fall through from.
                # The old code fell through to chunking the Actor's own input,
                # API token included, and pushed it to the output dataset.
                return [], (
                    f"Could not read dataset {dataset_id}: {result.error}. "
                    f"Nothing was charged and nothing was processed."
                )
            if result.truncated:
                Actor.log.warning(
                    f"Dataset reports {result.reported_total} items but only "
                    f"{len(result.items)} were read. Output is INCOMPLETE."
                )
            Actor.log.info(
                f"Loaded {len(result.items)} item(s) across {result.pages} page(s)"
                + (f"; dataset reports {result.reported_total} total."
                   if result.reported_total is not None else ".")
            )
            if result.items:
                return result.items, None

    items = actor_input.get("data") or actor_input.get("items") or []
    if not isinstance(items, list):
        items = [items] if items else []
    if items:
        return items, None

    # Whole-input fallback, redacted so credentials present in the input can
    # never be chunked into the output dataset.
    reserved = {"token", "datasetId", "dataset_id", "resource", "data", "items",
                "text_field", *DEFAULTS.keys(), *LEGACY_KEYS.keys()}
    leftover = {k: v for k, v in actor_input.items() if k not in reserved}
    if leftover:
        Actor.log.warning(
            "No datasetId, data or items provided; treating the remaining input "
            "fields as a single document."
        )
        return [redact(leftover)], None
    return [], None


def row_for_chunk(
    chunk: Chunk, index: int, total: int, *, source_id: str, source_url: str,
    field_used: str, encoder_label: str, cleaned_at: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "chunk_id": f"{chunk.content_hash[:16]}-{index}",
        "original_id": source_id,
        "chunk_index": index,
        "total_chunks": total,
        "chunk_text": chunk.text,
        "token_count": chunk.tokens,
        "token_count_method": encoder_label,
        "chunk_length_chars": len(chunk.text),
        "content_hash": chunk.content_hash,
        "overlap_tokens": chunk.overlap_tokens,
        "block_kinds": chunk.block_kinds,
        "cleaned_at": cleaned_at,
    }
    if chunk.heading_path:
        row["heading_path"] = chunk.heading_path
        row["section"] = " > ".join(chunk.heading_path)
    if source_url:
        row["source_url"] = source_url
    if field_used:
        row["source_field"] = field_used
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    async with Actor:
        actor_input = await Actor.get_input() or {}
        settings = resolve_settings(actor_input)

        encoder = get_encoder(settings["encoding"])
        if not is_exact(encoder):
            Actor.log.warning(
                "tiktoken encoding unavailable (" + "; ".join(load_errors()) + "). "
                "Token counts are ESTIMATED from character length; every row is "
                "marked token_count_method=estimated:chars so you can tell."
            )

        items, fatal = await gather_items(actor_input)
        if fatal:
            Actor.log.error(fatal)
            await Actor.set_status_message(fatal)
            return
        if not items:
            Actor.log.warning("No input items. Nothing to process, nothing charged.")
            await Actor.set_status_message("No input items; nothing charged.")
            return

        # Extract before charging, so the bill matches what is actually read.
        extracted: List[Tuple[int, Dict[str, Any], str, str]] = []
        skipped_non_dict = 0
        skipped_no_text = 0
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                skipped_non_dict += 1
                continue
            text, field_used = extract_text(item, settings["text_field"])
            if not text:
                skipped_no_text += 1
                continue
            extracted.append((idx, item, text, field_used))

        if skipped_non_dict:
            Actor.log.warning(
                f"{skipped_non_dict} item(s) were not objects and were skipped (not charged)."
            )
        if skipped_no_text:
            Actor.log.warning(
                f"{skipped_no_text} item(s) had none of the recognized text fields and "
                f"were skipped (not charged). Set `text_field` to name the field "
                f"explicitly if your scraper uses a custom one."
            )
        if not extracted:
            Actor.log.warning("No readable text in any item. Nothing charged.")
            await Actor.set_status_message("No readable text found; nothing charged.")
            return

        billable = compute_billable_bytes([t for _, _, t, _ in extracted])
        units, charge_result = await charge_for_input(billable)
        if charge_result is not None and getattr(charge_result, "event_charge_limit_reached", False):
            msg = (f"Stopped: this input needs {units} unit(s) of '{INPUT_KB_EVENT}', "
                   f"which exceeds this run's max cost per run. Nothing was processed.")
            Actor.log.warning(msg)
            await Actor.set_status_message(msg)
            return

        Actor.log.info(
            f"Processing {len(extracted)} document(s), {billable} bytes of text "
            f"({units} KB unit(s) charged). max_tokens={settings['max_tokens']}, "
            f"overlap_tokens={settings['overlap_tokens']}, encoding={settings['encoding']}."
        )

        started_at = Actor.get_env().get("started_at")
        cleaned_at = (started_at.isoformat() if hasattr(started_at, "isoformat")
                      else str(started_at))
        label = method_label(encoder)

        total_chunks = 0
        total_out_tokens = 0
        total_in_tokens = 0
        failed_items = 0

        for idx, item, raw_text, field_used in extracted:
            try:
                cleaned = clean_text(
                    raw_text,
                    preserve_links=settings["preserve_links"],
                    preserve_structure=True,
                    drop_nav=settings["drop_nav"],
                )
                if not cleaned:
                    Actor.log.warning(f"Item {idx}: nothing left after cleaning; skipped.")
                    continue

                chunks = chunk_document(
                    cleaned, encoder,
                    max_tokens=settings["max_tokens"],
                    overlap_tokens=settings["overlap_tokens"],
                    min_tokens=settings["min_tokens"],
                    split_on_heading_level=settings["split_on_heading_level"],
                    include_heading_context=settings["include_heading_context"],
                )
                if not chunks:
                    continue

                total_in_tokens += encoder.count(cleaned)
                source_url = first_of(item, URL_FIELDS)
                source_id = first_of(item, ID_FIELDS) or source_url or f"item_{idx}"

                for i, chunk in enumerate(chunks):
                    await Actor.push_data(row_for_chunk(
                        chunk, i, len(chunks),
                        source_id=source_id, source_url=source_url,
                        field_used=field_used, encoder_label=label,
                        cleaned_at=cleaned_at,
                    ))
                    total_out_tokens += chunk.tokens
                total_chunks += len(chunks)

            except Exception as exc:
                failed_items += 1
                Actor.log.warning(
                    f"Item {idx} failed and was skipped: {type(exc).__name__}: {exc}"
                )
                continue

        amplification = (total_out_tokens / total_in_tokens) if total_in_tokens else 1.0
        summary = (
            f"{total_chunks} chunks from {len(extracted)} document(s). "
            f"{total_out_tokens} output tokens vs {total_in_tokens} input tokens "
            f"({amplification:.2f}x from overlap) - that multiplier is what your "
            f"embedding provider will bill you for."
        )
        if failed_items:
            summary += f" {failed_items} item(s) failed."
        Actor.log.info(summary)
        await Actor.set_status_message(
            f"{total_chunks} chunks from {len(extracted)} document(s) "
            f"({amplification:.2f}x token amplification)"
        )
