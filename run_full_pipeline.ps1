param(
    [Parameter(Mandatory = $false)]
    [string]$Url,

    [Parameter(Mandatory = $false)]
    [string]$AudioPath,

    [Parameter(Mandatory = $false)]
    [string]$Speaker = "shubh",

    [Parameter(Mandatory = $false)]
    [double]$Pace = 1.0,

    [Parameter(Mandatory = $false)]
    [double]$Temperature = 0.4
)

$ErrorActionPreference = "Stop"

$downloadScript = Join-Path $PSScriptRoot "download_media.ps1"
$transcribeScript = Join-Path $PSScriptRoot "transcribe_latest_audio.ps1"
$translateScript = Join-Path $PSScriptRoot "translate_caption_tamil.ps1"
$ttsScript = Join-Path $PSScriptRoot "tts_tamil_bulbul_v3.ps1"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH."
    exit 1
}

if ($Url) {
    Write-Host "Step 1/4: Downloading video and extracting audio..." -ForegroundColor Cyan
    & $downloadScript -Url $Url
}
else {
    Write-Host "Step 1/4: Skipping download (no URL provided)." -ForegroundColor Yellow
}

Write-Host "Step 2/4: Transcribing audio..." -ForegroundColor Cyan
if ($AudioPath) {
    & $transcribeScript -AudioPath $AudioPath
}
else {
    & $transcribeScript
}

Write-Host "Step 3/4: Translating captions to Tamil..." -ForegroundColor Cyan
& $translateScript -Colloquial

Write-Host "Step 4/4: Generating Tamil audio with Bulbul v3..." -ForegroundColor Cyan
& $ttsScript -Speaker $Speaker -Pace $Pace -Temperature $Temperature

Write-Host "Pipeline complete (download + transcription + translation + TTS)." -ForegroundColor Green
