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

# --- Self-elevate to admin if not already ---------------------------------
# The bench needs CUDA toolkit + cuDNN on PATH. NVIDIA's installer adds those
# to the SYSTEM PATH which is only fully inherited by elevated (admin) shells.
# If we're not admin, re-launch ourselves via UAC so the inherited PATH
# contains the NVIDIA entries.
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    $scriptPath = $MyInvocation.MyCommand.Path
    $argList = $args -join ' '
    Write-Host '[bench] not running as admin - re-launching elevated so the system PATH (CUDA, cuDNN) is inherited'
    $q = [char]34
    Start-Process -FilePath powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File $q$scriptPath$q $argList" -Verb RunAs
    exit $LASTEXITCODE
}

$ErrorActionPreference = "Stop"

# --- Shared knobs (override via env before running) --------------------------
$GAMES  = if ($env:GAMES)  { [int]$env:GAMES }  else { 64 }
$SIMS   = if ($env:SIMS)   { [int]$env:SIMS }   else { 800 }
$EPOCHS = if ($env:EPOCHS) { [int]$env:EPOCHS } else { 5 }
$BATCH  = if ($env:BATCH)  { [int]$env:BATCH }  else { 256 }

# Python is ALWAYS GPU per the design contract. Don't make this configurable.
$PYTHON_DEVICE = "cuda"

# Make sure cuDNN (from the pip-installed nvidia-cudnn-cu12) and the CUDA toolkit
# runtime DLLs (cublasLt64_12.dll, etc.) are on PATH. The Rust `ort` crate's CUDA
# execution provider loads these via dlopen at runtime.
#
# Best-effort discovery — relies on the user having either a CUDA toolkit
# installed system-wide (auto-discovered under ProgramFiles) or the
# nvidia-cudnn-cu12 wheel installed via pip. If the user runs this
# script in an admin shell, the system PATH (which includes the NVIDIA
# installer entries) is fully inherited, and the discovery below is
# usually a no-op confirmation.
$cudnnBin = $null
try {
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
    Write-Host "[bench] CUDA bin on PATH:   NO - CUDA toolkit not on PATH. Run this script in an admin shell, or install CUDA 12.x."
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