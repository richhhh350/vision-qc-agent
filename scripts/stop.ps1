$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPidFile = Join-Path $taskRoot 'runs\services.json'
if (-not (Test-Path -LiteralPath $taskPidFile)) { Write-Output 'No saved service processes.'; exit 0 }
$taskServices = Get-Content -LiteralPath $taskPidFile -Raw | ConvertFrom-Json
foreach ($taskProcessId in @($taskServices.api, $taskServices.dashboard)) {
    $taskProcess = Get-Process -Id $taskProcessId -ErrorAction SilentlyContinue
    if ($taskProcess -and $taskProcess.Path -eq (Join-Path $taskRoot '.venv\Scripts\python.exe')) {
        & taskkill.exe /PID $taskProcessId /T /F
        if ($LASTEXITCODE -ne 0) { throw "Could not stop project process $taskProcessId." }
    }
}
