$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "C:\Users\MANO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$depsDir = Join-Path $scriptDir ".deps"

New-Item -ItemType Directory -Force -Path $depsDir | Out-Null

& $python -m pip install --upgrade pip
& $python -m pip install --target $depsDir python-jobspy

Write-Host "Installed python-jobspy into $depsDir"
