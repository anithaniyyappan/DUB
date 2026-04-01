param(
    [Parameter(Mandatory = $false)]
    [string]$Url
)

$ErrorActionPreference = "Stop"

if (-not $Url) {
    $Url = Read-Host "Enter YouTube URL"
}

$Url = $Url.Trim()

if ([string]::IsNullOrWhiteSpace($Url)) {
    Write-Error "No URL provided."
    exit 1
}

if (-not (Get-Command yt-dlp -ErrorAction SilentlyContinue)) {
    Write-Error "yt-dlp is not installed or not in PATH."
    exit 1
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Error "ffmpeg is not installed or not in PATH."
    exit 1
}

$videoDir = Join-Path $PSScriptRoot "downloads\video"
$audioDir = Join-Path $PSScriptRoot "downloads\audio"

New-Item -ItemType Directory -Force -Path $videoDir | Out-Null
New-Item -ItemType Directory -Force -Path $audioDir | Out-Null

$videoTemplate = Join-Path $videoDir "%(title)s [%(id)s].%(ext)s"
$audioTemplate = Join-Path $audioDir "%(title)s [%(id)s].%(ext)s"
$commonArgs = @(
    "--no-playlist",
    "--extractor-args", "youtube:player_client=web,android",
    "--js-runtimes", "deno,node"
)

Write-Host "`nDownloading video..." -ForegroundColor Cyan
yt-dlp @commonArgs -f "bv*+ba/b" --merge-output-format mp4 -o $videoTemplate $Url

Write-Host "`nExtracting audio (mp3)..." -ForegroundColor Cyan
yt-dlp @commonArgs -x --audio-format mp3 -o $audioTemplate $Url

Write-Host "`nDone." -ForegroundColor Green
Write-Host "Video folder: $videoDir"
Write-Host "Audio folder: $audioDir"
