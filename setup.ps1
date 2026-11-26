$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$rev = '92c294e6624802dd78042ebd47370a127a6d6f96'
New-Item -ItemType Directory -Force .runtime | Out-Null
if (!(Test-Path -LiteralPath '.runtime/nunif/iw3/__main__.py')) {
    curl.exe -fsSL --retry 3 "https://codeload.github.com/nagadomi/nunif/zip/$rev" -o .runtime/nunif.zip
    if ($LASTEXITCODE -ne 0) { throw 'nunif download failed' }
    $sha = (Get-FileHash -LiteralPath .runtime/nunif.zip).Hash
    if ($sha -ne 'B75CAED25C2E4D903788F8D7998CE24B5CF722B0C41EC7C93E969378E029521D') { throw 'nunif archive checksum mismatch' }
    tar -xf .runtime/nunif.zip -C .runtime
    if ($LASTEXITCODE -ne 0) { throw 'nunif extraction failed' }
    Rename-Item -LiteralPath ".runtime/nunif-$rev" -NewName nunif
}
if (!(Test-Path -LiteralPath '.venv313/Scripts/python.exe')) {
    uv venv --python 3.13 .venv313
    if ($LASTEXITCODE -ne 0) { throw 'venv setup failed' }
}
$env:UV_LINK_MODE = 'copy'
uv pip install --python .venv313/Scripts/python.exe --index https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match 'torch==2.14.1+cu132' 'torchvision==0.29.1+cu132' -r .runtime/nunif/requirements.txt 'gradio==6.32.0'
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed' }
# The CPU and CUDA wheels install the same Python package.
$ortInstallArgs = @()
if (uv pip show --python .venv313/Scripts/python.exe onnxruntime 2>$null) {
    uv pip uninstall --python .venv313/Scripts/python.exe onnxruntime
    if ($LASTEXITCODE -ne 0) { throw '无法替换旧版 ONNX Runtime' }
    $ortInstallArgs = @('--reinstall-package', 'onnxruntime-gpu')
}
uv pip install --python .venv313/Scripts/python.exe @ortInstallArgs opencv-python-headless==4.13.0.92 segment-anything==1.0 spandrel==0.4.2 onnxruntime-gpu==1.32.0
if ($LASTEXITCODE -ne 0) { throw '推理依赖安装失败' }
& .venv313/Scripts/python.exe -c "import torch, av, onnxruntime as ort; assert torch.cuda.is_available() and 'CUDAExecutionProvider' in ort.get_available_providers(); print(torch.cuda.get_device_name(0), av.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed' }
