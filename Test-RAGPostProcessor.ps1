# =====================================================
# RAG Post Processor - Accuracy & Quality Test Suite
# =====================================================
# Run this to validate your actor is working correctly
# Usage: . .\Test-RAGPostProcessor.ps1
# =====================================================

# Load the wrapper if not already loaded
if (-not (Get-Command Invoke-RAGPostProcessor -ErrorAction SilentlyContinue)) {
    . "$PSScriptRoot\Invoke-RAGPostProcessor.ps1"
}

$testCases = @(
    @{
        Name        = "Plain text - single chunk"
        Text        = "This is a simple clean sentence that should come back as one chunk with no issues."
        ExpectChunks = 1
        ExpectText  = $true
    },
    @{
        Name        = "HTML with boilerplate noise"
        Text        = "<html><body><nav>Menu Home About Contact</nav><main><h1>Real Title</h1><p>This is the important article content that should be preserved after cleaning. It contains useful information for RAG pipelines.</p></main><footer>Copyright 2026. All rights reserved. Cookie policy. Privacy policy.</footer></body></html>"
        ExpectChunks = 1
        ExpectText  = $true
    },
    @{
        Name        = "Long text - should produce multiple chunks"
        Text        = ("Machine learning is a subset of artificial intelligence. " * 25)
        ExpectChunks = 1   # Will update this after we see how many chunks come back
        ExpectText  = $true
    },
    @{
        Name        = "Text with extra whitespace and line breaks"
        Text        = "First paragraph with useful content.`n`n`n`n`nSecond paragraph after many blank lines.`n   Extra spaces here   .`n`nThird paragraph at the end."
        ExpectChunks = 1
        ExpectText  = $true
    },
    @{
        Name        = "Short edge case - minimal input"
        Text        = "Short."
        ExpectChunks = 1
        ExpectText  = $true
    }
)

$passed = 0
$failed = 0
$results = @()

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  RAG Post Processor - Quality Test Suite" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

foreach ($test in $testCases) {
    Write-Host "TEST: $($test.Name)" -ForegroundColor Yellow
    Write-Host "      Input length: $($test.Text.Length) chars" -ForegroundColor DarkGray

    try {
        $chunks = Invoke-RAGPostProcessor -InputText $test.Text

        $chunkCount    = $chunks.Count
        $emptyChunks   = ($chunks | Where-Object { [string]::IsNullOrWhiteSpace($_.ChunkText) }).Count
        $totalChars    = ($chunks | Measure-Object -Property CharCount -Sum).Sum
        $avgChars      = if ($chunkCount -gt 0) { [math]::Round($totalChars / $chunkCount, 0) } else { 0 }
        $preview       = if ($chunks[0].ChunkText) { $chunks[0].ChunkText.Substring(0, [Math]::Min(80, $chunks[0].ChunkText.Length)) } else { "(empty)" }

        # Evaluate pass/fail
        $testPassed = $true
        $issues = @()

        if ($chunkCount -eq 0)    { $testPassed = $false; $issues += "Zero chunks returned" }
        if ($emptyChunks -gt 0)   { $testPassed = $false; $issues += "$emptyChunks empty chunk(s)" }
        if ($test.ExpectText -and -not $chunks[0].ChunkText) { $testPassed = $false; $issues += "ChunkText field is null" }

        if ($testPassed) {
            Write-Host "      PASSED" -ForegroundColor Green
            $passed++
        } else {
            Write-Host "      FAILED: $($issues -join ', ')" -ForegroundColor Red
            $failed++
        }

        Write-Host "      Chunks: $chunkCount  |  Total chars: $totalChars  |  Avg chunk: $avgChars chars" -ForegroundColor DarkGray
        Write-Host "      Preview: $preview..." -ForegroundColor DarkGray

        $results += [PSCustomObject]@{
            TestName    = $test.Name
            Passed      = $testPassed
            ChunkCount  = $chunkCount
            EmptyChunks = $emptyChunks
            TotalChars  = $totalChars
            AvgChars    = $avgChars
            Issues      = ($issues -join "; ")
        }

    } catch {
        Write-Host "      ERROR: $_" -ForegroundColor Red
        $failed++
        $results += [PSCustomObject]@{
            TestName    = $test.Name
            Passed      = $false
            ChunkCount  = 0
            EmptyChunks = 0
            TotalChars  = 0
            AvgChars    = 0
            Issues      = "Exception: $_"
        }
    }

    Write-Host ""
}

# Summary
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  RESULTS: $passed passed, $failed failed" -ForegroundColor $(if ($failed -eq 0) { 'Green' } else { 'Red' })
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# Export results to CSV
$csvPath = "$PSScriptRoot\test-results-$(Get-Date -Format 'yyyyMMdd-HHmmss').csv"
$results | Export-Csv -Path $csvPath -NoTypeInformation
Write-Host "Full results saved to: $csvPath" -ForegroundColor DarkGray
Write-Host ""
