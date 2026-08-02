# RAG Post Processor — token-aware, structure-preserving chunker

Turn scraped pages into chunks you can embed without surprises: **sized in
tokens**, **split on document structure**, and **carrying the metadata a
retrieval pipeline actually needs**.

Drop it after any scraper, or feed it items directly.

---

## Why tokens, not characters

Character budgets are the default in most chunkers, and they don't mean
anything to an embedding model. Measured with `cl100k_base`, one
1000-character chunk is:

| Content | chars/token | tokens in 1000 chars |
|---|---|---|
| English prose | 6.08 | ~160 |
| Python source | 4.47 | ~220 |
| JSON | 2.41 | ~415 |
| German | 3.27 | ~300 |
| Japanese | 0.89 | **~1080** |

A single character setting gives a 7× spread. `max_tokens` here is a hard
ceiling under the encoding you select — no emitted chunk ever exceeds it.

> `text-embedding-3-small` and `text-embedding-3-large` use **`cl100k_base`**,
> not `o200k_base`, despite being newer models. That's the default here.

---

## Why structure

Chunking by slicing a character stream cuts through the things that make a
passage interpretable. This Actor parses the document into blocks first —
headings, paragraphs, fenced code, tables, lists, quotes — and packs whole
blocks into chunks.

- **Code that must be split keeps its fence and language on every part**, so
  every chunk is still valid, highlightable code.
- **A table that must be split repeats its header row on every part**, so no
  chunk is an anonymous wall of pipe-separated values.
- **Lists split between items, never inside one.**
- **Headings lead their content**, never trail at the end of the previous chunk.
- **Every chunk carries its heading path**, both inline (optional breadcrumb)
  and as structured `heading_path` / `section` fields.
- **Chunks begin and end on whole words and whole sentences.**

## Why the metadata

Every row carries what an ingestion pipeline would otherwise have to recompute:

- `token_count` — exact, under your chosen encoding, so you can batch embedding
  calls without re-tokenizing
- `content_hash` + `chunk_id` — stable across runs, so you can upsert unchanged
  chunks instead of paying to re-embed them
- `heading_path` / `section` — for metadata filtering and for showing users
  where an answer came from
- `block_kinds` — filter code out of prose retrieval, or the reverse
- `source_field` — which field of the input item was actually read
- `overlap_tokens` — how much of this chunk is repeated context

The run log also reports **token amplification** (output tokens ÷ input
tokens), which is the multiplier your embedding provider will bill you for.

---

## Input

Four input paths, tried in order. Chaining reads **every page** of the source
dataset, not just the first.

| Field | Type | Default | Description |
|---|---|---|---|
| `datasetId` | string | — | Dataset ID from a previous Actor run. Chain after any scraper. |
| `data` / `items` | array | — | Items passed inline. |
| `text_field` | string | auto | Read this field instead of auto-detecting. |
| `max_tokens` | integer | `512` | Hard token ceiling per chunk (16–8191). |
| `overlap_tokens` | integer | `64` | Whole-sentence overlap, clamped to 50% of `max_tokens`. |
| `min_tokens` | integer | `24` | Below this, merge into the previous chunk. Text is never dropped. |
| `encoding` | enum | `cl100k_base` | `cl100k_base`, `o200k_base`, `p50k_base`, `r50k_base`. |
| `split_on_heading_level` | integer | `0` | `0` packs sections; `1`–`6` forces a break at that heading level. |
| `include_heading_context` | boolean | `true` | Prefix each chunk with its heading breadcrumb. |
| `preserve_links` | boolean | `true` | Keep URLs; convert anchors to markdown links. |
| `drop_nav` | boolean | `true` | Drop `<nav>`/`<menu>`. Script/style/comments are always dropped. |

Text is auto-detected from `markdown`, `text`, `content`, `body`,
`page_content`, `html` and ~25 other common scraper field names. The field
chosen is reported on every row as `source_field`.

Chunk sizes moved from characters to tokens in v1.0. The old `chunk_size`,
`overlap` and `min_chunk_chars` inputs are still accepted: they are converted
to their token equivalents at ~4 characters per token, and the conversion is
written to the run log.

### Example input

```json
{
  "datasetId": "YOUR_SCRAPER_DATASET_ID",
  "max_tokens": 512,
  "overlap_tokens": 64,
  "encoding": "cl100k_base",
  "split_on_heading_level": 2
}
```

---

## Output

```json
{
  "chunk_id": "9f2c1a77b4e3d018-1",
  "original_id": "https://docs.example.com/sdk/install",
  "source_url": "https://docs.example.com/sdk/install",
  "source_field": "markdown",
  "chunk_index": 1,
  "total_chunks": 4,
  "chunk_text": "Widget SDK > Quick start\n\n```python\nfrom widget import Client\n\nclient = Client(api_key=\"sk-test\")\n```",
  "token_count": 67,
  "token_count_method": "tiktoken:cl100k_base",
  "chunk_length_chars": 231,
  "heading_path": ["Widget SDK", "Quick start"],
  "section": "Widget SDK > Quick start",
  "block_kinds": ["code", "heading"],
  "content_hash": "9f2c1a77b4e3d018…",
  "overlap_tokens": 0,
  "cleaned_at": "2026-08-02T04:31:00+00:00"
}
```

---

## Cleaning

The cleaner is HTML-aware (stdlib parser, not regex) and removes the contents —
not just the tags — of `<script>`, `<style>`, `<noscript>`, `<template>`,
`<svg>` and HTML comments, so tracking snippets and CSS rules never reach your
index.

It **decodes** HTML entities rather than deleting them (`&pound;99` stays
`£99`, `&#8220;` becomes `“`), handles numeric and named forms, and **keeps
URLs**, converting `<a href>` into markdown links so retrieved passages remain
citable. HTML headings, lists, tables and `<pre>` blocks are converted to their
markdown equivalents so the chunker can see the structure.

---

## Pricing

**$0.0005 per KB of text processed** (rounded up), charged once per run,
computed from the bytes of text this Actor actually reads — not from whole-item
JSON, and not from how many chunks the text turns into. No chunking setting can
change your bill.

Items skipped because they had no readable text are not charged for.

---

## Chaining with other actors

Works after **Website Content Crawler**, **Cheerio Scraper**, or any Actor
emitting a text-like field. Pass its dataset ID as `datasetId`, or use it as an
Actor-to-Actor trigger target — the `resource.defaultDatasetId` chaining format
is handled too.

Common destinations for the output: Pinecone, Qdrant, Weaviate, pgvector,
Chroma, Milvus, or any LangChain / LlamaIndex ingestion step.

---

## Reliability notes

- The tokenizer's BPE tables are baked into the Docker image at build time, so
  no run depends on a network fetch to tokenize. If a tokenizer is somehow
  unavailable anyway, the run continues with a conservative character estimate
  and **every row is marked** `token_count_method: "estimated:chars"` so you can
  tell the difference.
- Dataset reads are paginated, retried with backoff on `429`/`5xx`, and
  time-limited. `X-Apify-Pagination-Total` is compared against what was read,
  and an incomplete read is reported in the log rather than passing silently.
- A failed dataset read stops the run before anything is charged.
- Credentials present in the Actor input are redacted before any fallback path
  can put them into the output dataset.
