param([string]$PythonExecutable = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
New-Item -ItemType Directory -Force -Path '.cache\tmp' | Out-Null
$env:TEMP = Join-Path $projectRoot '.cache\tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $projectRoot '.cache\pip'
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $PythonExecutable -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment.' }
}
& '.\.venv\Scripts\python.exe' -m ensurepip --upgrade --default-pip
if ($LASTEXITCODE -ne 0) { throw 'Could not bootstrap pip.' }
& '.\.venv\Scripts\python.exe' -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Training dependency installation failed.' }
& '.\.venv\Scripts\python.exe' -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed.' }
& '.\.venv\Scripts\python.exe' -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA GPU unavailable'; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'PyTorch GPU preflight failed.' }
