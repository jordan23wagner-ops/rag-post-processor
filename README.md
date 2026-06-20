# RAG Post Processor - Text Cleaner & Chunker for LLM Pipelines

Clean and chunk raw scraped text for RAG and LLM pipelines. Drop it after any scraper actor and get embedding-ready chunks in seconds.

## What it does

- Strips HTML tags and boilerplate
- Collapses whitespace and normalizes line breaks
- Splits text into overlapping chunks (default: 1000 chars, 100 overlap)
- Returns structured output with chunk index, length, and timestamp
- Works standalone or chained after Website Content Crawler and similar actors

## Input

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data` | array | required | Array of objects from a previous scraper. Each object needs a `text`, `content`, `body`, or `html` field. |
| `chunk_size` | integer | 1000 | Max characters per chunk |
| `overlap` | integer | 100 | Character overlap between chunks |

### Example input

```json
{
  "data": [
    { "text": "Your raw scraped content goes here. It can be long, messy HTML or plain text." }
  ],
  "chunk_size": 1000,
  "overlap": 100
}
```

## Output

Each chunk is returned as a dataset item:

```json
{
  "original_id": "item_0",
  "chunk_index": 0,
  "total_chunks": 3,
  "chunk_text": "Cleaned and chunked text ready for embedding...",
  "chunk_length_chars": 487,
  "cleaned_at": "2026-06-20 04:46:29.330000+00:00"
}
```

## Pricing

$0.0003 per output row. No subscription required — pay only for what you use.

## Use with PowerShell

Install the companion PowerShell module to call this actor from your automation scripts:

```powershell
Import-Module RAGPostProcessor
Invoke-RAGPostProcessor -InputText "Your scraped text here" -VerboseOutput
```

## Chaining with other actors

Works directly after **Website Content Crawler**, **Cheerio Scraper**, or any actor that outputs a `text` or `content` field. Use the Apify actor-to-actor API to pipe output from a scraper straight into this processor.

## Common use cases

- Preparing scraped web content for vector databases (Pinecone, Weaviate, Chroma)
- Cleaning LangChain / LlamaIndex document ingestion pipelines
- Pre-processing data for OpenAI embeddings or similar APIs
- Automating RAG pipeline data prep without custom code
