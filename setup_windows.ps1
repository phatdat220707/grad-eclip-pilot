$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$external = Join-Path $root 'external'
$grad = Join-Path $external 'Grad-Eclip'
$clip = Join-Path $external 'CLIP'
$venvPython = Join-Path $root '.venv\Scripts\python.exe'

$env:PIP_CACHE_DIR = Join-Path $root '.cache\pip'
$env:TEMP = Join-Path $root '.cache\tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $external, $env:PIP_CACHE_DIR, $env:TEMP -Force | Out-Null

if (-not (Test-Path -LiteralPath (Join-Path $grad '.git'))) {
    git clone https://github.com/Cyang-Zhao/Grad-Eclip.git $grad
    if ($LASTEXITCODE -ne 0) { throw 'Grad-ECLIP clone failed' }
}
git -C $grad checkout --detach e370e6cb194faf2020f5d1ed268f9d57e91a38e6
if ($LASTEXITCODE -ne 0) { throw 'Grad-ECLIP checkout failed' }

if (-not (Test-Path -LiteralPath (Join-Path $clip '.git'))) {
    git clone https://github.com/openai/CLIP.git $clip
    if ($LASTEXITCODE -ne 0) { throw 'OpenAI CLIP clone failed' }
}
git -C $clip checkout --detach d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
if ($LASTEXITCODE -ne 0) { throw 'OpenAI CLIP checkout failed' }

if (-not (Test-Path -LiteralPath $venvPython)) {
    python -m venv (Join-Path $root '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed' }
}
& $venvPython -m pip install torch==2.15.0+cpu torchvision==0.29.2+cpu --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'PyTorch install failed' }
& $venvPython -m pip install -r (Join-Path $root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Python dependency install failed' }
& $venvPython -m pip install --no-deps -e $clip
if ($LASTEXITCODE -ne 0) { throw 'OpenAI CLIP install failed' }

Write-Host 'Setup complete. Run:'
Write-Host "& '$venvPython' '$(Join-Path $root 'scripts\run_baseline.py')'"
