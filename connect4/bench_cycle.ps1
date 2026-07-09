# =============================================================================
# bench_cycle.ps1 - End-to-end cycle benchmark for the Connect4 pipeline.
#
# Compares two configurations:
#   A: Python=cuda (forced) + Rust=cpu   <-- baseline, no CUDA toolkit needed for Rust
#   B: Python=cuda (forced) + Rust=gpu   <-- needs `cargo build --release --features cuda`
#
# Both use the same self-play + training parameters (GAMES, SIMS, EPOCHS, BATCH).
# Python device is HARD-CODED to CUDA per the design contract.
#
# Usage (from the project root):
#   .\bench_cycle.ps1
#   $env:GAMES=128; $env:SIMS=1600; $env:EPOCHS=10; .\bench_cycle.ps1
#
# Each config runs MAX_CYCLES=1 (one full cycle: self-play + train + ONNX export).
# Wall-clock time is reported for each. Speedup = A / B.
# =============================================================================

$ErrorActionPreference = "Stop"

# --- Shared knobs (override via env before running) --------------------------
$GAMES  = if ($env:GAMES)  { [int]$env:GAMES }  else { 64 }
$SIMS   = if ($env:SIMS)   { [int]$env:SIMS }   else { 800 }
$EPOCHS = if ($env:EPOCHS) { [int]$env:EPOCHS } else { 5 }
$BATCH  = if ($env:BATCH)  { [int]$env:BATCH }  else { 256 }

# Python is ALWAYS GPU per the design contract. Don't make this configurable.
$PYTHON_DEVICE = "cuda"

# Make sure cuDNN (from the pip-installed nvidia-cudnn-cu12) and the matching
# CUDA toolkit runtime DLLs (cublasLt64_12.dll, etc.) are on PATH. The Rust
# `ort` crate's CUDA execution provider loads these via dlopen at runtime.
#
# Critical: nvidia-cudnn-cu12 wheel needs CUDA 12.x — the cuBLAS DLL
# it depends on is cublasLt64_12.dll, which only exists in v12.x toolkits.
# If the user has BOTH v12.x and v13.x installed, picking v13 silently
# breaks Config B. So:
#   1. Find the v12 toolkit that has cublasLt64_12.dll.
#   2. Make sure cuDNN's bin dir is on PATH (it lives under
#      <site-packages>/nvidia/cudnn/bin/, see the cuDNN discovery below).
#   3. Honor $env:CUDA_PATH if it's set AND points to a v12 dir.

$cudaBin = $null

# 1. Honor CUDA_PATH if it points to a usable v12.x toolkit.
if ($env:CUDA_PATH) {
    $candidate = Join-Path $env:CUDA_PATH 'bin'
    if ((Test-Path (Join-Path $candidate 'cublasLt64_12.dll'))) {
        $cudaBin = $candidate
    }
}

# 2. Otherwise, find the v12.x toolkit that has cublasLt64_12.dll.
if (-not $cudaBin) {
    $candidates = Get-ChildItem "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^v12\.' } |
        ForEach-Object { Join-Path $_.FullName 'bin' } |
        Where-Object { Test-Path (Join-Path $_ 'cublasLt64_12.dll') } |
        Sort-Object { ($_ -split '\\')[-2] } -Descending   # highest v12.x first
    if ($candidates.Count -gt 0) { $cudaBin = $candidates[0] }
}
if ($cudaBin) { $env:PATH = "$cudaBin;$env:PATH" }

$cudnnBin = $null
try {
    # nvidia-cudnn-cu12 is a data-only wheel: it ships DLLs under
    # `<site-packages>/nvidia/cudnn/bin/` but NO Python module. So we
    # ask pip directly for the install location of the wheel, then
    # derive the bin dir. Works for system-wide AND user-mode installs.
    $cudnnLoc = & $PYTHON -c "import importlib.metadata, os; dist = importlib.metadata.distribution('nvidia-cudnn-cu12'); print(os.path.join(os.path.dirname(dist._path), 'nvidia', 'cudnn', 'bin'))" 2>$null
    $cudnnLoc = ($cudnnLoc | Select-Object -Last 1).Trim()
    if ($cudnnLoc -and (Test-Path $cudnnLoc)) { $cudnnBin = $cudnnLoc }
} catch {}
if ($cudnnBin) { $env:PATH = "$cudnnBin;$env:PATH" }

Write-Host "[bench] params: GAMES=$GAMES SIMS=$SIMS EPOCHS=$EPOCHS BATCH=$BATCH"
Write-Host "[bench] python device: $PYTHON_DEVICE (hard-coded)"
if ($cudaBin) {
    Write-Host "[bench] CUDA bin on PATH:   yes ($cudaBin)"
} else {
    Write-Host "[bench] CUDA bin on PATH:   NO - no v12.x CUDA toolkit with cublasLt64_12.dll found."
    Write-Host "           (nvidia-cudnn-cu12 needs CUDA 12.x, not 13.x. Install CUDA 12.6 via winget.)"
}
if ($cudnnBin) {
    Write-Host "[bench] cuDNN bin on PATH:  yes ($cudnnBin)"
} else {
    Write-Host "[bench] cuDNN bin on PATH:  NO - nvidia-cudnn-cu12 not installed (run: pip install nvidia-cudnn-cu12)"
}
Write-Host ""

# --- Helper: run one cycle, return total seconds ----------------------------
function Invoke-Cycle([string]$RustDevice) {
    $env:RUST_DEVICE   = $RustDevice
    $env:PYTHON_DEVICE = $PYTHON_DEVICE
    $env:MAX_CYCLES    = "1"
    $env:GAMES         = "$GAMES"
    $env:SIMS          = "$SIMS"
    $env:EPOCHS        = "$EPOCHS"
    $env:BATCH         = "$BATCH"
    # Batching strategy: BATCH_SIZE=1 for CPU (tract already optimal at batch=1,
    # batching ADDS overhead via virtual loss bookkeeping). BATCH_SIZE=32 for GPU
    # (ort+CUDA is FFI-bound per-call, batching hides that latency).
    $env:BATCH_SIZE    = if ($RustDevice -eq "gpu") { "32" } else { "1" }

    Write-Host ""
    Write-Host "[bench] ============== Config: rust=$RustDevice + python=$PYTHON_DEVICE =============="
    $t = Measure-Command { & ".\run_pipeline.ps1" }
    Write-Host ""
    Write-Host ("[bench] >> total wall-clock for this config: {0:F1} s" -f $t.TotalSeconds)
    return $t.TotalSeconds
}

# --- Run both configs ------------------------------------------------------
$timeA = Invoke-Cycle -RustDevice "cpu"
$timeB = Invoke-Cycle -RustDevice "gpu"

# --- Report -----------------------------------------------------------------
Write-Host ""
Write-Host "================================================================="
Write-Host "  CYCLE BENCHMARK RESULTS"
Write-Host "================================================================="
Write-Host ("  Config A (rust-cpu + py-gpu):  {0,8:F1} s" -f $timeA)
Write-Host ("  Config B (rust-gpu + py-gpu):  {0,8:F1} s" -f $timeB)
if ($timeB -gt 0) {
    $speedup = [math]::Round($timeA / $timeB, 2)
    Write-Host ("  Speedup (A/B):                {0,8}x" -f $speedup)
    if ($speedup -gt 1.0) {
        Write-Host "  -> GPU is faster for Rust self-play." -ForegroundColor Green
    } elseif ($speedup -lt 1.0) {
        Write-Host "  -> CPU is faster for Rust self-play (MCTS overhead dominates)." -ForegroundColor Yellow
    } else {
        Write-Host "  -> Tied." -ForegroundColor Gray
    }
}
Write-Host "================================================================="