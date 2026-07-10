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

param (
    [string]$Mode = "both" # Peut être "cpu", "gpu" ou "both"
)

# --- Pull in the system PATH from registry (no admin needed) -------------
# The NVIDIA installer writes CUDA toolkit + cuDNN into the SYSTEM PATH
# (HKLM\...Session Manager\Environment\Path). Non-elevated shells only
# see the user PATH, so we explicitly merge in the system PATH here
# without needing UAC elevation.
$sysPath = (Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment' -Name Path -ErrorAction SilentlyContinue).Path
if ($sysPath) {
    $env:PATH = "$sysPath;$env:PATH"
}

$ErrorActionPreference = "Stop"

# --- Shared knobs (override via env before running) --------------------------
$GAMES       = 256
$SIMS        = 800
$EPOCHS      = 5
$BATCH_SIZE  = 32

# Python is ALWAYS GPU per the design contract. Don't make this configurable.
$PYTHON_DEVICE = "cuda"

# Make sure cuDNN (from the pip-installed nvidia-cudnn-cu12) and the CUDA toolkit
# runtime DLLs (cublasLt64_12.dll, etc.) are on PATH. The Rust `ort` crate's CUDA
# execution provider loads these via dlopen at runtime.
#
# Discovery (no hardcodes):
#   - cuDNN: ask pip directly for the install location of the nvidia-cudnn-cu12
#     wheel, then derive <site-packages>/nvidia/cudnn/bin/. Works for
#     system-wide AND user-mode installs.
#   - CUDA toolkit: scan ProgramFiles for any v1x.x toolkit whose bin/ contains
#     cublasLt64_12.dll (the cuDNN-cu12 compatibility marker — this DLL only
#     exists in CUDA 12.x toolkits). If found, prepend its bin to PATH. This
#     handles stale registry PATH entries and multiple CUDA installs.
$PYTHON   = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$cudnnBin = $null
$cudaBin  = $null
try {
    $cudnnLoc = & $PYTHON -c "import importlib.metadata, os; dist = importlib.metadata.distribution('nvidia-cudnn-cu12'); print(os.path.join(os.path.dirname(dist._path), 'nvidia', 'cudnn', 'bin'))" 2>$null
    $cudnnLoc = ($cudnnLoc | Select-Object -Last 1).Trim()
    if ($cudnnLoc -and (Test-Path $cudnnLoc)) { $cudnnBin = $cudnnLoc }
} catch {}
if ($cudnnBin) { $env:PATH = "$cudnnBin;$env:PATH" }

if (Test-Path "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA") {
    $cudaRoot = Get-ChildItem "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^v1[02]\.' } |  # v10.x or v12.x (skip v13+: no cublasLt64_12.dll)
        Where-Object { Test-Path (Join-Path $_.FullName 'bin\cublasLt64_12.dll') } |
        Sort-Object { $_.Name } -Descending |
        Select-Object -First 1 -ExpandProperty FullName
    if ($cudaRoot) { $cudaBin = Join-Path $cudaRoot 'bin' }
}
if ($cudaBin) { $env:PATH = "$cudaBin;$env:PATH" }

Write-Host "[bench] params: GAMES=$GAMES SIMS=$SIMS EPOCHS=$EPOCHS BATCH=$BATCH_SIZE"
Write-Host "[bench] python device: $PYTHON_DEVICE (hard-coded)"
if ($cudaBin) {
    Write-Host "[bench] CUDA bin on PATH:   yes ($cudaBin)"
} else {
    Write-Host "[bench] CUDA bin on PATH:   NO - no CUDA v12.x toolkit with cublasLt64_12.dll found."
    Write-Host "           (nvidia-cudnn-cu12 wheel needs CUDA 12.x. Run setup.ps1 or: winget install Nvidia.CUDA --version 12.6)"
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
    $env:BATCH_SIZE    = "$BATCH_SIZE"
    # Batching strategy: BATCH_SIZE=32. With recent optimizations, batching
    # works nicely on CPU as well by reducing tree-traversal overhead.
    # Détermination de la taille de batch MCTS
    if ($RustDevice -eq "cpu") {
        $env:BATCH_SIZE = [string]$env:NUMBER_OF_PROCESSORS  # Auto-détection (12 sur ton fixe, 8 sur laptop)
    } else {
        $env:BATCH_SIZE = [string]$BATCH_SIZE  # Utilise la variable globale du script (ex: 256 ou 128)
    }

    Write-Host ""
    Write-Host "[bench] ============== Config: rust=$RustDevice + python=$PYTHON_DEVICE =============="
    $t = Measure-Command {
        & ".\run_pipeline.ps1" -Games $GAMES -Sims $SIMS -Epochs $EPOCHS -Batch $env:BATCH_SIZE -BatchSize $env:BATCH_SIZE -RustDevice $RustDevice -PythonDevice $PYTHON_DEVICE
    }
    Write-Host ""
    Write-Host ("[bench] >> total wall-clock for this config: {0:F1} s" -f $t.TotalSeconds)
    return $t.TotalSeconds
}

# Déplacer temporairement la conf locale pour que le banc d'essai ait la priorité
$hasConf = Test-Path "pipeline.conf.ps1"
if ($hasConf) { Rename-Item "pipeline.conf.ps1" "pipeline.conf.ps1.bak" -Force }

# --- Run both configs ------------------------------------------------------
# --- Run selected configs --------------------------------------------------
$timeA = 0
$timeB = 0

if ($Mode -eq "cpu" -or $Mode -eq "both") {
    $timeA = Invoke-Cycle -RustDevice "cpu"
}
if ($Mode -eq "gpu" -or $Mode -eq "both") {
    $timeB = Invoke-Cycle -RustDevice "gpu"
}

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