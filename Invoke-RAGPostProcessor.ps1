function Invoke-RAGPostProcessor {
    [CmdletBinding()]
    param(
        [Parameter(ValueFromPipeline)]
        [string]$InputText,
        [string]$InputFile,
        [int]$ChunkSize = 1000,
        [int]$Overlap = 200,
        [switch]$VerboseOutput
    )

    begin {
        $ActorId = "rPRQKJP9bsGxsU9Ed"
        $ApiToken = $env:APIFY_TOKEN
        if (-not $ApiToken) { throw "Set your Apify token: `$env:APIFY_TOKEN = 'apify_api_...'" }
        $base = "https://api.apify.com/v2"
        $headers = @{ Authorization = "Bearer $ApiToken" }
    }

    process {
        if ($InputFile) { $InputText = Get-Content $InputFile -Raw }
        if ([string]::IsNullOrWhiteSpace($InputText)) { throw "Provide -InputText or -InputFile" }

        $payload = @{
            data       = @(@{ text = $InputText })
            chunk_size = $ChunkSize
            overlap    = $Overlap
        } | ConvertTo-Json -Depth 10

        if ($VerboseOutput) { Write-Host "Starting actor..." -ForegroundColor Cyan }

        $syncUrl = "$base/actors/$ActorId/run-sync-get-dataset-items?token=$ApiToken&timeout=300&memory=256"

        try {
            $raw = Invoke-RestMethod -Uri $syncUrl -Method Post -Headers $headers `
                -Body $payload -ContentType 'application/json' -TimeoutSec 310
        } catch {
            throw "Actor call failed: $_"
        }

        # Force results into an array so .Count always works correctly
        $results = @($raw)

        if ($results.Count -eq 0) {
            Write-Warning "No results returned."
            return @()
        }

        $chunks = $results | ForEach-Object {
            [PSCustomObject]@{
                ChunkText   = $_.chunk_text
                CharCount   = $_.chunk_length_chars
                ChunkIndex  = $_.chunk_index
                TotalChunks = $_.total_chunks
                OriginalId  = $_.original_id
                CleanedAt   = $_.cleaned_at
            }
        }

        # Force output as array so .Count is always reliable for callers
        $chunks = @($chunks)

        if ($VerboseOutput) {
            Write-Host "Returned $($chunks.Count) chunk(s)." -ForegroundColor Green
        }

        return $chunks
    }
}
