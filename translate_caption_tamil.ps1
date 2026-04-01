param(
    [Parameter(Mandatory = $false)]
    [string]$CaptionPath,

    [Parameter(Mandatory = $false)]
    [switch]$Colloquial,

    [Parameter(Mandatory = $false)]
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$captionDir = Join-Path $PSScriptRoot "downloads\captions"
$lastCaptionPtr = Join-Path $captionDir ".last_caption_path"
$transcribeScript = Join-Path $PSScriptRoot "transcribe_latest_audio.ps1"
$translatePy = Join-Path $PSScriptRoot "sarvam_translate_captions.py"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH."
    exit 1
}

$style = "formal"
if ($Colloquial) {
    $style = "colloquial"
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

$resolvedCaption = $null
if ($CaptionPath) {
    $resolvedCaption = Resolve-ExistingLiteralPath -PathValue $CaptionPath
    if (-not $resolvedCaption) {
        Write-Error "Caption file not found: $CaptionPath"
        exit 1
    }
}

if (-not $resolvedCaption) {
    if (Test-Path -LiteralPath $lastCaptionPtr) {
        $candidate = Get-Content -LiteralPath $lastCaptionPtr -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($candidate) {
            $resolvedCaption = Resolve-ExistingLiteralPath -PathValue $candidate
        }
    }
}

if (-not $resolvedCaption) {
    $sourceCaptions = Get-ChildItem -LiteralPath $captionDir -File -Filter *.txt -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notlike "*.ta-IN.txt" }

    if ($sourceCaptions) {
        $resolvedCaption = ($sourceCaptions | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName)
    }
}

if (-not $resolvedCaption) {
    Write-Host "No source caption .txt found. Running transcription first..." -ForegroundColor Yellow
    & $transcribeScript
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $sourceCaptions = Get-ChildItem -LiteralPath $captionDir -File -Filter *.txt -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notlike "*.ta-IN.txt" }
    if ($sourceCaptions) {
        $resolvedCaption = ($sourceCaptions | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName)
    }
}

if (-not $resolvedCaption) {
    Write-Error "No source caption found even after transcription."
    exit 1
}

$captionItem = Get-Item -LiteralPath $resolvedCaption -ErrorAction Stop
$expectedOut = Join-Path $captionItem.DirectoryName ($captionItem.BaseName + ".ta-IN.txt")
if ((-not $Force) -and (Test-Path -LiteralPath $expectedOut)) {
    $outItem = Get-Item -LiteralPath $expectedOut
    if ($outItem.LastWriteTime -ge $captionItem.LastWriteTime) {
        Write-Host "Translation already exists, skipping API call. Use -Force to regenerate." -ForegroundColor Yellow
        Write-Host "Existing translation: $expectedOut"
        exit 0
    }
}

& python $translatePy --caption $resolvedCaption --target-lang "ta-IN" --style $style --spell-fix

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
