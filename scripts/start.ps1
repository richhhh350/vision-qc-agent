param([string]$Run = 'runs/demo', [string]$State = 'runs/state', [switch]$NoBrowser, [switch]$V2)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if ($V2) { $taskPython = Join-Path $taskRoot '.venv-v2\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run scripts/setup.ps1 first.' }
$taskListeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
if ($taskListeners.Port -contains 8000 -or $taskListeners.Port -contains 8501) {
    throw 'Port 8000 or 8501 is already in use. Stop the existing services with scripts/stop.ps1 before restarting.'
}
if (-not (Test-Path -LiteralPath (Join-Path $Run 'model.json'))) {
    if ($V2) { throw 'Train a v2 model with scripts/v2.py train before starting.' }
    & $taskPython -m vision_qc.cli demo --output $Run
    if ($LASTEXITCODE -ne 0) { throw 'Demo training failed.' }
}
$env:QC_RUN_DIR = $Run
$env:QC_STATE_DIR = $State
New-Item -ItemType Directory -Path 'runs' -Force | Out-Null
$taskApi = Start-Process -FilePath $taskPython -ArgumentList @('-m','uvicorn','vision_qc.api:create_app','--factory','--host','127.0.0.1','--port','8000') -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput 'runs/api.log' -RedirectStandardError 'runs/api-error.log'
$taskDashboard = Start-Process -FilePath $taskPython -ArgumentList @('-m','streamlit','run','dashboard.py','--server.address','127.0.0.1','--server.port','8501','--server.headless','true','--browser.gatherUsageStats','false') -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput 'runs/dashboard.log' -RedirectStandardError 'runs/dashboard-error.log'
@{api=$taskApi.Id; dashboard=$taskDashboard.Id; python=$taskPython} | ConvertTo-Json | Set-Content -LiteralPath 'runs/services.json'
Write-Output 'Dashboard: http://127.0.0.1:8501 | API: http://127.0.0.1:8000/docs'
if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:8501' }
