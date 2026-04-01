param(
    [Parameter(Mandatory = $false)]
    [string]$AudioPath
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH."
    exit 1
}

function Resolve-ExistingLiteralPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue
    )

    $trimmed = $PathValue.Trim().Trim('"').Trim("'")
    if (-not $trimmed) {
        return $null
    }

    try {
        return (Resolve-Path -LiteralPath $trimmed -ErrorAction Stop).Path
    }
    catch {
        return $null
    }
}

$transcribePy = Join-Path $PSScriptRoot "sarvam_speech_to_text.py"
$resolvedAudioPath = $null

if ($AudioPath) {
    $resolvedAudioPath = Resolve-ExistingLiteralPath -PathValue $AudioPath
    if (-not $resolvedAudioPath) {
        Write-Error "Audio file not found: $AudioPath"
        exit 1
    }
}

if ($resolvedAudioPath) {
    & python $transcribePy --audio $resolvedAudioPath
}
else {
    & python $transcribePy
}

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
