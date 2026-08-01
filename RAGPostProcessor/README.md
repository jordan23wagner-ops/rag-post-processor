# RAGPostProcessor

PowerShell wrapper for the [RAG Post Processor](https://apify.com/jalicia/rag-post-processor) Apify actor.

Clean and chunk scraped text for RAG and LLM pipelines in one command. Strips HTML, collapses whitespace, and splits into overlapping chunks ready for embedding.

## Install

```powershell
Install-Module RAGPostProcessor
```

## Setup

Get your free Apify API token at [apify.com](https://console.apify.com/account/integrations) then set it:

```powershell
$env:APIFY_TOKEN = "apify_api_your_token_here"
```

## Usage

```powershell
Import-Module RAGPostProcessor

# Process a string
$chunks = Invoke-RAGPostProcessor -InputText "Your scraped content here" -VerboseOutput

# Process a file
$chunks = Invoke-RAGPostProcessor -InputFile ".\scraped-page.txt" -VerboseOutput

# Custom chunk size
$chunks = Invoke-RAGPostProcessor -InputText "..." -ChunkSize 500 -Overlap 50

# Export to CSV
$chunks = Invoke-RAGPostProcessor -InputFile ".\input.txt"
$chunks | Export-Csv ".\chunks.csv" -NoTypeInformation

# Chain after any Apify scraper using its dataset ID
$chunks = Invoke-RAGPostProcessor -InputText "" -VerboseOutput
# Pass datasetId via the actor directly for full chaining
```

## Output fields

Each chunk returned is a PowerShell object with:

| Field | Description |
|-------|-------------|
| ChunkText | Cleaned chunk text ready for embedding |
| CharCount | Character count of this chunk |
| ChunkIndex | Position in the sequence (0-based) |
| TotalChunks | Total chunks produced from this input |
| OriginalId | Source item identifier |
| CleanedAt | UTC timestamp of processing |

## Pricing

Uses the RAG Post Processor Apify actor at $0.0003 per output chunk. No subscription required.

## Links

- [Apify Actor](https://apify.com/jalicia/rag-post-processor)
- [PowerShell Gallery](https://www.powershellgallery.com/packages/RAGPostProcessor)
