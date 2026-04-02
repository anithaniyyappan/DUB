param(
    [Parameter(Mandatory = $false)]
    [string]$CaptionPath,

    [Parameter(Mandatory = $false)]
    [string]$Speaker = "shubh",

    [Parameter(Mandatory = $false)]
    [double]$Pace = 1.0,

    [Parameter(Mandatory = $false)]
    [double]$Temperature = 0.4,

    [Parameter(Mandatory = $false)]
    [int]$SampleRate = 24000,

    [Parameter(Mandatory = $false)]
    [int]$PauseMs = 350,

    [Parameter(Mandatory = $false)]
    [int]$BlankLinePauseMs = 650,

    [Parameter(Mandatory = $false)]
    [int]$MaxGapMs = 4000
    ,
    [Parameter(Mandatory = $false)]
    [switch]$NoSyncOriginal,

    [Parameter(Mandatory = $false)]
    [switch]$ForceGlobalSync,

    [Parameter(Mandatory = $false)]
    [double]$SyncToleranceSec = 0.15
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

$ttsPy = Join-Path $PSScriptRoot "sarvam_tamil_tts_bulbul_v3.py"
$commandArgs = @(
    $ttsPy,
    "--speaker", $Speaker,
    "--pace", $Pace,
    "--temperature", $Temperature,
    "--sample-rate", $SampleRate,
    "--pause-ms", $PauseMs,
    "--blank-line-pause-ms", $BlankLinePauseMs,
    "--max-gap-ms", $MaxGapMs,
    "--sync-tolerance-sec", $SyncToleranceSec
)

if ($CaptionPath) {
    $resolvedCaptionPath = Resolve-ExistingLiteralPath -PathValue $CaptionPath
    if (-not $resolvedCaptionPath) {
        Write-Error "Caption file not found: $CaptionPath"
        exit 1
    }
    $commandArgs += @("--caption", $resolvedCaptionPath)
}

if ($NoSyncOriginal) {
    $commandArgs += "--no-sync-original"
}

if ($ForceGlobalSync) {
    $commandArgs += "--force-global-sync"
}

& python @commandArgs

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
