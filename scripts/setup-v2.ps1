param([string]$Environment = '.venv-v2')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $Environment 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    python -m venv $Environment
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
& $taskPython -m pip install -r requirements-v2-tested.txt
if ($LASTEXITCODE -ne 0) { throw 'Locked dependency installation failed.' }
& $taskPython -m pip install -e . --no-deps
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
& $taskPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
Write-Output 'V2 environment ready. Download data/teacher and use scripts/v2.py train.'
