# =============================================================================
# setup.ps1 - One-shot setup for the Connect4 pipeline on a fresh Windows box.
#
# Installs:
#   1. CUDA Toolkit 12.6 (via winget)        - ~3 GB, system-level
#   2. cuDNN 9.x for CUDA 12  (via pip)     - ~700 MB wheel
#   3. PyTorch + deps (via requirements.txt)
#   4. (Optional) Rust toolchain (via winget)
#
# Run from the project root in a regular PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
#
# After setup, just `.\bench_cycle.ps1` works.
# =============================================================================

$ErrorActionPreference = "Stop"

Write-Host "[setup] Connect4 pipeline setup"
Write-Host "[setup] project root: $PSScriptRoot"
Write-Host ""

# --- 1. CUDA Toolkit 12.6 -----------------------------------------------------
$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
if (Test-Path (Join-Path $cudaRoot "bin\cudart64_12.dll")) {
    Write-Host "[setup] CUDA 12.6 already installed at $cudaRoot - skipping"
} else {
    Write-Host "[setup] Installing CUDA Toolkit 12.6 (winget) - this takes 3-5 min..."
    winget install Nvidia.CUDA --version 12.6 --accept-source-agreements --accept-package-agreements
    if (-not (Test-Path (Join-Path $cudaRoot "bin\cudart64_12.dll"))) {
        Write-Host "[setup] FAILED: CUDA 12.6 install did not produce expected DLL at $cudaRoot\bin"
        Write-Host "[setup]   (winget sometimes installs to a different path. Check 'Get-ChildItem C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA')"
        exit 1
    }
    Write-Host "[setup] CUDA 12.6 installed OK"
}

# --- 2. cuDNN 9.x for CUDA 12 (pip wheel) ------------------------------------
Write-Host "[setup] Installing nvidia-cudnn-cu12 (cuDNN 9.x wheel for CUDA 12)"
pip install --upgrade nvidia-cudnn-cu12 | Out-Null
$cudnnLoc = & python -c "import importlib.metadata, os; dist = importlib.metadata.distribution('nvidia-cudnn-cu12'); print(os.path.join(os.path.dirname(dist._path), 'nvidia', 'cudnn', 'bin'))" 2>$null
$cudnnLoc = ($cudnnLoc | Select-Object -Last 1).Trim()
if (-not $cudnnLoc -or -not (Test-Path (Join-Path $cudnnLoc "cudnn64_9.dll"))) {
    Write-Host "[setup] FAILED: nvidia-cudnn-cu12 wheel did not produce expected DLL"
    exit 1
}
Write-Host "[setup] cuDNN installed OK at $cudnnLoc"

# --- 3. Python deps -----------------------------------------------------------
Write-Host "[setup] Installing Python deps from requirements.txt"
pip install -r "$PSScriptRoot\src_python\requirements.txt" | Out-Null
Write-Host "[setup] Python deps installed"

# --- 4. (Optional) Rust toolchain --------------------------------------------
$cargo = (Get-Command cargo -ErrorAction SilentlyContinue)
if ($cargo) {
    Write-Host "[setup] Rust already installed: $($cargo.Source)"
} else {
    Write-Host "[setup] Installing Rust toolchain (winget)"
    winget install Rustlang.Rustup --accept-source-agreements --accept-package-agreements
    Write-Host "[setup] Rust installed. CLOSE AND REOPEN POWERSHELL for PATH to update."
}

Write-Host ""
Write-Host "[setup] Done. Next steps:"
Write-Host "  1. (If you just installed Rust) close and reopen PowerShell"
Write-Host "  2. cd $PSScriptRoot\src_rust ; cargo build --release --features cuda"
Write-Host "  3. cd $PSScriptRoot ; .\bench_cycle.ps1"