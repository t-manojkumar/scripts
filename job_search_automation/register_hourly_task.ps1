$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "C:\Users\MANO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$scriptPath = Join-Path $scriptDir "resume_job_search.py"
$configPath = Join-Path $scriptDir "config.json"
$taskName = "ResumeJobSearchHourly"
$startTime = (Get-Date).AddMinutes(2).ToString("HH:mm")
$taskCommand = "`"$python`" `"$scriptPath`" --config `"$configPath`""

schtasks /Create /SC HOURLY /MO 1 /ST $startTime /TN $taskName /TR $taskCommand /F | Out-Host

Write-Host "Registered task '$taskName' to run every hour starting at $startTime."
