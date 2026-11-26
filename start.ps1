$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv313\Scripts\python.exe'
if (!(Test-Path -LiteralPath $python)) { throw '请先按 README 安装本机依赖。' }
if (!$env:MMD_INPUT) { New-Item -ItemType Directory -Path input -Force | Out-Null }
if (!$env:MMD_OUTPUT) { New-Item -ItemType Directory -Path output -Force | Out-Null }
$env:PYTHONUTF8 = '1'
& $python launch.py
if ($LASTEXITCODE -ne 0) { throw '启动失败，请查看 .runtime 中的日志。' }
