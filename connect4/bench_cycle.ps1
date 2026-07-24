# =============================================================================
# bench_cycle.ps1 - Isolated, ultra-fast benchmark for Connect4 pipeline.
#
# Benchmark Architecture (3 Stages):
#   Stage 1: Rust MCTS Self-Play (CPU)   — Benchmark multithreaded CPU inference.
#   Stage 2: Rust MCTS Self-Play (GPU)   — Benchmark ONNX Runtime GPU inference.
#   Stage 3: PyTorch Training (CUDA GPU) — Benchmark training & ONNX export once.
# 
# Note: Both CPU and GPU MCTS run on the exact same `model_init.onnx` to
# strictly benchmark raw compute speed without model capability bias.
# =============================================================================

param (
    [string]$Mode = "both" # "cpu", "gpu" or "both"
)

$ErrorActionPreference = "Stop"
$env:PYTHONWARNINGS = "ignore"

# --- Resolve Python Executable ---
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$PYTHON = if ($env:PYTHON) { $env:PYTHON } elseif (Test-Path $venvPy) { $venvPy } else { "python" }

# --- Load Central Configuration (config.py) ----------------------------------
$configScript = Join-Path $PSScriptRoot "config.py"
if (Test-Path $configScript) {
    try {
        $envCode = & $PYTHON $configScript --powershell 2>$null
        if ($LASTEXITCODE -eq 0 -and $envCode) {
            Invoke-Expression ($envCode -join "`n")
            Write-Host "[bench] Loaded configuration from config.py" -ForegroundColor Cyan
        }
    } catch {
        Write-Warning "[bench] Failed to parse config.py output: $_"
    }
}

# --- Parameters -------------------------------------------------------------
$GAMES            = if ($env:BENCH_GAMES)            { [int]$env:BENCH_GAMES }            else { 64 }
$SIMS             = if ($env:BENCH_SIMS)             { [int]$env:BENCH_SIMS }             else { 800 }
$EPOCHS           = if ($env:BENCH_EPOCHS)           { [int]$env:BENCH_EPOCHS }           else { 5 }
$TRAIN_BATCH_SIZE = if ($env:BENCH_TRAIN_BATCH_SIZE) { [int]$env:BENCH_TRAIN_BATCH_SIZE } else { 256 }
$CPU_BATCH_SIZE   = if ($env:BENCH_CPU_BATCH_SIZE)   { [int]$env:BENCH_CPU_BATCH_SIZE }   else { [int]$env:NUMBER_OF_PROCESSORS }
$GPU_BATCH_SIZE   = if ($env:BENCH_GPU_BATCH_SIZE)   { [int]$env:BENCH_GPU_BATCH_SIZE }   else { 32 }
$BENCH_SEED       = if ($env:BENCH_SEED)             { [int]$env:BENCH_SEED }             else { 42 }
$CARGO            = if ($env:CARGO)                  { $env:CARGO }                       else { "cargo" }

# Register cuDNN and CUDA PATH
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
        Where-Object { $_.Name -match '^v1[02]\.' } |
        Where-Object { Test-Path (Join-Path $_.FullName 'bin\cublasLt64_12.dll') } |
        Sort-Object { $_.Name } -Descending |
        Select-Object -First 1 -ExpandProperty FullName
    if ($cudaRoot) { $cudaBin = Join-Path $cudaRoot 'bin' }
}
if ($cudaBin) { $env:PATH = "$cudaBin;$env:PATH" }

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  CONNECT4 PIPELINE BENCHMARK" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "[bench] Params: GAMES=$GAMES SIMS=$SIMS CPU_BATCH=$CPU_BATCH_SIZE GPU_BATCH=$GPU_BATCH_SIZE EPOCHS=$EPOCHS SEED=$BENCH_SEED"
Write-Host "[bench] Using Python: $PYTHON"
Write-Host ""

$benchDir = Join-Path $PSScriptRoot ".bench_temp"
if (-not (Test-Path $benchDir)) { New-Item -ItemType Directory -Path $benchDir | Out-Null }

try {
    # --- Step 1: Pre-compile Rust binary --------------------------------------
    Write-Host "[bench/prep] Pre-building Rust release binary with CUDA support..." -ForegroundColor Yellow
    & $CARGO build --release --features cuda --manifest-path "$PSScriptRoot\src_rust\Cargo.toml"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[bench/prep] Failed to compile Rust binary."
        exit 1
    }
    Write-Host "[bench/prep] Rust release build complete." -ForegroundColor Green

    # --- Step 2: Pre-create baseline model ------------------------------------
    $initPt   = Join-Path $benchDir "model_init.pt"
    $initOnnx = Join-Path $benchDir "model_init.onnx"

    Write-Host "[bench/prep] Creating isolated baseline model (seed=$BENCH_SEED)..." -ForegroundColor Yellow
    Push-Location "src_python"
    & $PYTHON "init.py" --out-pt $initPt --out-onnx $initOnnx --seed $BENCH_SEED --force
    $rc = $LASTEXITCODE
    Pop-Location
    if ($rc -ne 0) {
        Write-Error "[bench/prep] Failed to create baseline model."
        exit 1
    }

    $cargoExe = Join-Path $PSScriptRoot "src_rust\target\release\connect4_mcts.exe"
    Write-Host "[bench/prep] Setup complete." -ForegroundColor Green

    # Helper function to format duration
    function Format-BenchDur([double]$sec) {
        if ($sec -ge 60) {
            return ("{0:D}m {1:D2}s" -f [int]($sec / 60), [int]($sec % 60))
        } else {
            return ("{0:F2}s" -f $sec)
        }
    }

    $resCpu = $null
    $resGpu = $null

    # --- Stage 1: Benchmark Rust MCTS Self-Play Inference (CPU) ----------------
    if ($Mode -eq "cpu" -or $Mode -eq "both") {
        Write-Host ""
        Write-Host "-----------------------------------------------------------------" -ForegroundColor Cyan
        Write-Host ("  STAGE 1: Rust MCTS Self-Play Inference (CPU, batch={0})" -f $CPU_BATCH_SIZE) -ForegroundColor Cyan
        Write-Host "-----------------------------------------------------------------" -ForegroundColor Cyan
        
        $cpuData = Join-Path $benchDir "selfplay_cpu.bin"
        $tStart = Get-Date
        & $cargoExe -g $GAMES -s $SIMS -b $CPU_BATCH_SIZE -o $cpuData -m $initOnnx -d cpu --seed $BENCH_SEED -v
        if ($LASTEXITCODE -eq 0) {
            $dur = (Get-Date) - $tStart
            $resCpu = @{ SelfplaySec = $dur.TotalSeconds; BatchSize = $CPU_BATCH_SIZE }
            Write-Host ("[bench/mcts-cpu] Self-play completed in {0}" -f (Format-BenchDur $dur.TotalSeconds)) -ForegroundColor Green
        }
    }

    # --- Stage 2: Benchmark Rust MCTS Self-Play Inference (GPU) ----------------
    if ($Mode -eq "gpu" -or $Mode -eq "both") {
        Write-Host ""
        Write-Host "-----------------------------------------------------------------" -ForegroundColor Cyan
        Write-Host ("  STAGE 2: Rust MCTS Self-Play Inference (GPU, batch={0})" -f $GPU_BATCH_SIZE) -ForegroundColor Cyan
        Write-Host "-----------------------------------------------------------------" -ForegroundColor Cyan
        
        $gpuData = Join-Path $benchDir "selfplay_gpu.bin"
        $tStart = Get-Date
        & $cargoExe -g $GAMES -s $SIMS -b $GPU_BATCH_SIZE -o $gpuData -m $initOnnx -d gpu --seed $BENCH_SEED -v
        if ($LASTEXITCODE -eq 0) {
            $dur = (Get-Date) - $tStart
            $resGpu = @{ SelfplaySec = $dur.TotalSeconds; BatchSize = $GPU_BATCH_SIZE }
            Write-Host ("[bench/mcts-gpu] Self-play completed in {0}" -f (Format-BenchDur $dur.TotalSeconds)) -ForegroundColor Green
        }
    }

    # --- Stage 3: Benchmark PyTorch Training & ONNX Export (ONCE) -------------
    Write-Host ""
    Write-Host "-----------------------------------------------------------------" -ForegroundColor Cyan
    Write-Host "  STAGE 3: PyTorch Training & ONNX Export (CUDA GPU)" -ForegroundColor Cyan
    Write-Host "-----------------------------------------------------------------" -ForegroundColor Cyan
    
    $benchReplay = Join-Path $benchDir "replay"
    if (Test-Path $benchReplay) { Remove-Item -Recurse -Force $benchReplay }
    New-Item -ItemType Directory -Path $benchReplay | Out-Null
    
    # Use the data generated by Stage 1 or Stage 2 for training
    if (Test-Path $gpuData) {
        Copy-Item -Force $gpuData (Join-Path $benchReplay "selfplay_0000.bin")
    } elseif (Test-Path $cpuData) {
        Copy-Item -Force $cpuData (Join-Path $benchReplay "selfplay_0000.bin")
    } else {
        Write-Error "[bench/train] No self-play data found to train on."
        exit 1
    }

    $trainedPt   = Join-Path $benchDir "model_trained.pt"

    Write-Host "[bench/train] Starting PyTorch Training (CUDA GPU, epochs=$EPOCHS, batch=$TRAIN_BATCH_SIZE)..." -ForegroundColor Yellow
    $tTrainStart = Get-Date
    Push-Location "src_python"
    & $PYTHON "train.py" --data-dir $benchReplay --out $trainedPt --epochs $EPOCHS --batch $TRAIN_BATCH_SIZE --device "cuda" --seed $BENCH_SEED --symmetry
    $trainRc = $LASTEXITCODE
    Pop-Location
    if ($trainRc -ne 0) {
        Write-Error "[bench/train] PyTorch Training failed."
        exit 1
    }
    $trainTime = (Get-Date) - $tTrainStart
    Write-Host ("[bench/train] PyTorch Training & ONNX export completed in {0}" -f (Format-BenchDur $trainTime.TotalSeconds)) -ForegroundColor Green

    # --- Benchmark Report -----------------------------------------------------
    Write-Host ""
    Write-Host "=================================================================" -ForegroundColor Cyan
    Write-Host "  BENCHMARK SUMMARY & PERFORMANCE REPORT" -ForegroundColor Cyan
    Write-Host "=================================================================" -ForegroundColor Cyan
    Write-Host ("  [Stage 3 - PyTorch Training (CUDA)] : {0}" -f (Format-BenchDur $trainTime.TotalSeconds)) -ForegroundColor Yellow
    Write-Host ""

    if ($resCpu) {
        Write-Host ("  [Stage 1 - Rust MCTS CPU  (batch={0})] :" -f $resCpu.BatchSize) -ForegroundColor Yellow
        Write-Host ("    - Self-Play (Inference) : {0}" -f (Format-BenchDur $resCpu.SelfplaySec))
    }

    if ($resGpu) {
        Write-Host ("  [Stage 2 - Rust MCTS GPU  (batch={0})] :" -f $resGpu.BatchSize) -ForegroundColor Yellow
        Write-Host ("    - Self-Play (Inference) : {0}" -f (Format-BenchDur $resGpu.SelfplaySec))
    }

    if ($resCpu -and $resGpu) {
        $speedupSelfplay = [math]::Round($resCpu.SelfplaySec / $resGpu.SelfplaySec, 2)
        Write-Host ""
        Write-Host ("  Self-Play Speedup (CPU -> GPU) : {0}x faster" -f $speedupSelfplay) -ForegroundColor Green
    }
    Write-Host "=================================================================" -ForegroundColor Cyan

} finally {
    # --- Clean up isolated benchmark directory ---
    Write-Host "[bench/clean] Cleaning up isolated benchmark directory .bench_temp/..." -ForegroundColor Gray
    if (Test-Path $benchDir) { Remove-Item -Recurse -Force $benchDir -ErrorAction SilentlyContinue }
    Write-Host "[bench/clean] Done. Real project files remain untouched." -ForegroundColor Gray
}