# Keep this wrapper free of a param block: pass Python's --flags through verbatim.
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host 'Run this once first: powershell -ExecutionPolicy Bypass -File .\setup.ps1' -ForegroundColor Red
    exit 2
}
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
& $python (Join-Path $PSScriptRoot 'diarize.py') @args
exit $LASTEXITCODE
