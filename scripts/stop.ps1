$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPidFile = Join-Path $taskRoot 'runs\services.json'
if (-not (Test-Path -LiteralPath $taskPidFile)) { Write-Output 'No saved service processes.'; exit 0 }
$taskServices = Get-Content -LiteralPath $taskPidFile -Raw | ConvertFrom-Json
foreach ($taskProcessId in @($taskServices.api, $taskServices.dashboard)) {
    $taskProcess = Get-Process -Id $taskProcessId -ErrorAction SilentlyContinue
    $taskAllowedPython = @((Join-Path $taskRoot '.venv\Scripts\python.exe'),(Join-Path $taskRoot '.venv-v2\Scripts\python.exe'))
    if ($taskProcess -and $taskProcess.Path -in $taskAllowedPython) {
        & taskkill.exe /PID $taskProcessId /T /F
        if ($LASTEXITCODE -ne 0) { throw "Could not stop project process $taskProcessId." }
    }
}
