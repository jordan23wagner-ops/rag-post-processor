# RAGPostProcessor.psm1
# Loads all public functions automatically

$Public = Get-ChildItem -Path "$PSScriptRoot\Public\*.ps1" -ErrorAction SilentlyContinue

foreach ($file in $Public) {
    try {
        . $file.FullName
    } catch {
        Write-Error "Failed to load $($file.FullName): $_"
    }
}

# Export all public functions
Export-ModuleMember -Function ($Public.BaseName)
