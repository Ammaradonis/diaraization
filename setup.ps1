<#
.SYNOPSIS
Installs a private Python 3.12 environment and the local diarization dependencies.
.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\setup.ps1
#>
[CmdletBinding()]
param([switch]$UpdateDownloader)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = $PSScriptRoot
$toolsDir = Join-Path $root '.tools'
$bootstrap = Join-Path $toolsDir 'bootstrap'
$python = Join-Path $root '.venv\Scripts\python.exe'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $toolsDir 'python'
$env:UV_CACHE_DIR = Join-Path $toolsDir 'uv-cache'
$env:UV_LINK_MODE = 'hardlink'
$env:UV_CONCURRENT_DOWNLOADS = '8'
$env:UV_CONCURRENT_INSTALLS = '2'
$env:UV_HTTP_TIMEOUT = '180'

function Assert-NativeSuccess([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit $LASTEXITCODE). Re-run setup.ps1 to retry." }
}

New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null
$uv = @(Get-ChildItem -LiteralPath $bootstrap -Filter uv.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName | Select-Object -First 1)
if (-not $uv) {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m pip install --disable-pip-version-check --no-warn-script-location --target $bootstrap 'uv==0.12.20'
        Assert-NativeSuccess 'Installing uv'
        $uv = @(Get-ChildItem -LiteralPath $bootstrap -Filter uv.exe -Recurse | Select-Object -ExpandProperty FullName | Select-Object -First 1)
    } else {
        $archive = Join-Path $toolsDir 'uv.zip'
        Invoke-WebRequest -UseBasicParsing 'https://github.com/astral-sh/uv/releases/download/0.12.20/uv-x86_64-pc-windows-msvc.zip' -OutFile $archive
        Expand-Archive -LiteralPath $archive -DestinationPath $bootstrap -Force
        $uv = @(Get-ChildItem -LiteralPath $bootstrap -Filter uv.exe -Recurse | Select-Object -ExpandProperty FullName | Select-Object -First 1)
    }
}
if (-not $uv) { throw 'Could not locate uv.exe in .tools\bootstrap.' }
$uv = $uv[0]

if (-not (Test-Path -LiteralPath $python)) {
    & $uv python install 3.12
    Assert-NativeSuccess 'Installing Python 3.12'
    & $uv venv --python 3.12 (Join-Path $root '.venv')
    Assert-NativeSuccess 'Creating .venv'
}

Write-Host 'Installing CPU PyTorch (this can take several minutes)...'
& $uv pip install --python $python --index-url 'https://download.pytorch.org/whl/cpu' 'torch==2.8.0' 'torchaudio==2.8.0' 'torchvision==0.23.0'
Assert-NativeSuccess 'Installing PyTorch'
& $uv pip install --python $python -r (Join-Path $root 'requirements.txt')
Assert-NativeSuccess 'Installing WhisperX and yt-dlp'
if ($UpdateDownloader) {
    & $uv pip install --python $python --upgrade 'yt-dlp[default,curl-cffi]'
    Assert-NativeSuccess 'Updating yt-dlp'
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or -not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    Write-Warning 'FFmpeg/ffprobe missing. Install: winget install --id Gyan.FFmpeg -e, then reopen PowerShell.'
}
if (-not (Get-Command node -ErrorAction SilentlyContinue) -and -not (Get-Command deno -ErrorAction SilentlyContinue)) {
    Write-Warning 'For YouTube, install Node.js 22+: winget install --id OpenJS.NodeJS.LTS -e, then reopen PowerShell.'
}
Write-Host ''
Write-Host 'Installation complete. env.txt is loaded automatically if present; see guide.md for model access.'
Write-Host 'Check readiness: .\run.ps1 --check'
Write-Host 'Start the queue: .\run.ps1 --workers 3'
