# =============================================================================
# run_pipeline.ps1 â€- Endless self-play â†- train loop for Connect4Net (Windows).
#
# Same logic as the previous bash version, but native PowerShell.
# Both scripts accept the same env-var overrides (GAMES, SIMS, EPOCHS, BATCH,
# DATA, MODEL, MODEL_ONNX, SLEEP, CARGO, PYTHON).
#
# On the very first cycle (no connect4_model.onnx yet), init.py is run to
# bootstrap a random-init model + ONNX export. Idempotent: subsequent
# cycles skip this step.
#
# Stop with Ctrl-C. The current iteration finishes; the next Start-Sleep is
# interrupted.
# =============================================================================

$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ScriptDir

# --- TensorRT Path Injection ---
if (-not ($env:PATH -like "*C:\TensorRT10\TensorRT-10.16.1.11\lib*")) {
    $env:PATH = "C:\TensorRT10\TensorRT-10.16.1.11\lib;" + $env:PATH
}

# --- Helpers ----------------------------------------------------------------
function Format-Seconds([datetime]$t0, [datetime]$t1) {
    $elapsed = ($t1 - $t0).TotalSeconds
    if ($elapsed -lt 1.0) {
        if ($elapsed -le 0.0) { return "0ms" }
        $milliseconds = [math]::Max(1, [int][math]::Round($elapsed * 1000.0))
        return "${milliseconds}ms"
    }
    return ("{0:F2}s" -f $elapsed)
}

# --- Resolve Python Executable ---
$venvPy   = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$PYTHON    = if ($env:PYTHON) { $env:PYTHON } elseif (Test-Path $venvPy) { $venvPy } else { "python" }

# --- Load Central Configuration (config.py) ----------------------------------
$configScript = Join-Path $ScriptDir "config.py"
if (Test-Path $configScript) {
    try {
        $envCode = & $PYTHON $configScript --powershell 2>$null
        if ($LASTEXITCODE -eq 0 -and $envCode) {
            Invoke-Expression ($envCode -join "`n")
            Write-Host "[pipeline] Loaded configuration from config.py" -ForegroundColor Cyan
        }
    } catch {
        Write-Warning "[pipeline] Failed to parse config.py output: $_"
    }
}

# --- Defaults / env --------------------------------------------------------
$GAMES            = if ($env:GAMES)            { [int]$env:GAMES }            else { 128 }
$SIMS             = if ($env:SIMS)             { [int]$env:SIMS }             else { 800 }
$PCR_FULL_PROB    = if ($env:PCR_FULL_PROBABILITY) { [float]$env:PCR_FULL_PROBABILITY } else { 0.25 }
$PCR_CHEAP_RATIO  = if ($env:PCR_CHEAP_RATIO)  { [float]$env:PCR_CHEAP_RATIO }  else { 0.1 }
$PCR_MIN_SIMS     = if ($env:PCR_MIN_SIMS)     { [int]$env:PCR_MIN_SIMS }     else { 32 }
$C_PUCT           = if ($env:C_PUCT)           { [float]$env:C_PUCT }         else { 1.5 }
$TEMPERATURE      = if ($env:TEMPERATURE)      { [float]$env:TEMPERATURE }    else { 1.0 }
$EPOCHS           = if ($env:EPOCHS)           { [int]$env:EPOCHS }           else { 1000000 }
$BATCH            = if ($env:TRAIN_BATCH_SIZE) { [int]$env:TRAIN_BATCH_SIZE } else { if ($env:BATCH) { [int]$env:BATCH } else { 256 } }
$LEARNING_RATE    = if ($env:LEARNING_RATE)    { $env:LEARNING_RATE }         else { "1e-3" }
$WEIGHT_DECAY     = if ($env:WEIGHT_DECAY)     { $env:WEIGHT_DECAY }          else { "1e-4" }
$COMPILE_MODE     = if ($env:COMPILE_MODE)     { $env:COMPILE_MODE }          else { "none" }
$INFER_PRECISION  = if ($env:INFER_PRECISION)  { $env:INFER_PRECISION }       else { "fp32" }
$ONNX_EVERY       = if ($env:ONNX_EVERY)       { [int]$env:ONNX_EVERY }       else { 0 }
$DATA             = if ($env:DATA)             { $env:DATA }                  else { "selfplay.bin" }
$MODEL            = if ($env:MODEL)            { $env:MODEL }                 else { "models\connect4_model.pt" }
$MODEL_ONNX       = if ($env:MODEL_ONNX)       { $env:MODEL_ONNX }            else { "models\connect4_model.onnx" }
$EVAL_GAMES       = if ($env:EVAL_GAMES)       { [int]$env:EVAL_GAMES }        else { 20 }
$EVAL_THINK_TIME  = if ($env:EVAL_THINK_TIME)  { [double]$env:EVAL_THINK_TIME } else { 0.25 }
$INFER_BACKEND    = if ($env:INFER_BACKEND)    { $env:INFER_BACKEND }          else { "auto" }
$BEST_MODEL_FILE  = if ($env:BEST_MODEL_FILE)  { $env:BEST_MODEL_FILE }        else { "models\best_model.json" }
$SLEEP            = if ($env:SLEEP)            { [int]$env:SLEEP }            else { 2 }
$MAX_CYCLES       = if ($env:MAX_CYCLES)       { [int]$env:MAX_CYCLES }       else { 0 }
$REPLAY_KEEP      = if ($env:REPLAY_KEEP)      { [int]$env:REPLAY_KEEP }      else { 10 }
$CPU_BATCH_SIZE   = if ($env:CPU_BATCH_SIZE)   { [int]$env:CPU_BATCH_SIZE }   else { [int]$env:NUMBER_OF_PROCESSORS }
$GPU_BATCH_SIZE   = if ($env:GPU_BATCH_SIZE)   { [int]$env:GPU_BATCH_SIZE }   else { 32 }
$SYMMETRY         = if ($env:SYMMETRY)         { [bool]($env:SYMMETRY -eq "1" -or $env:SYMMETRY -eq "true") } else { $true }
$CHANNELS_LAST    = if ($env:CHANNELS_LAST)    { [bool]($env:CHANNELS_LAST -eq "1" -or $env:CHANNELS_LAST -eq "true") } else { $false }
$FUSED_ADAMW      = if ($env:FUSED_ADAMW)      { [bool]($env:FUSED_ADAMW -eq "1" -or $env:FUSED_ADAMW -eq "true") } else { $false }
$CARGO            = if ($env:CARGO)            { $env:CARGO }                 else { "cargo" }
$IN_PROCESS       = if ($env:IN_PROCESS)       { [bool]($env:IN_PROCESS -eq "1" -or $env:IN_PROCESS -eq "true") } else { $false }

$RUST_DEVICE   = if ($env:RUST_DEVICE)   { $env:RUST_DEVICE }   else { if ($env:DEVICE) { $env:DEVICE } else { "auto" } }
$PYTHON_DEVICE = if ($env:PYTHON_DEVICE) { $env:PYTHON_DEVICE } else { if ($env:DEVICE) { $env:DEVICE } else { "cuda" } }
$BATCH_SIZE    = if ($RUST_DEVICE -eq "cpu") { $CPU_BATCH_SIZE } else { $GPU_BATCH_SIZE }
$FEATURES      = "--features cuda"

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "[pipeline] Starting Connect4 Self-Play Pipeline" -ForegroundColor Cyan
Write-Host "  * Self-Play : games=$GAMES sims=$SIMS c_puct=$C_PUCT temp=$TEMPERATURE batch_size=$BATCH_SIZE"
Write-Host ("  * Self-Play : PCR {0:P0} full / {1:P0} cheap (cheap=max({2:P0}*full,{3}) sims)" -f $PCR_FULL_PROB, (1.0 - $PCR_FULL_PROB), $PCR_CHEAP_RATIO, $PCR_MIN_SIMS)
Write-Host "  * Training  : epochs=$EPOCHS batch=$BATCH lr=$LEARNING_RATE wd=$WEIGHT_DECAY compile=$COMPILE_MODE"
Write-Host "  * Hardware  : rust_device=$RUST_DEVICE python_device=$PYTHON_DEVICE"
Write-Host "  * Paths     : model=$MODEL onnx=$MODEL_ONNX replay_keep=$REPLAY_KEEP"
Write-Host "=====================================================================" -ForegroundColor Cyan

# --- Bootstrap: if no ONNX model exists, create a random-init one. ---------
if (-not (Test-Path $MODEL_ONNX)) {
    Write-Host ""
    Write-Host "[pipeline] ===== bootstrap (no ONNX model found) ====="
    Write-Host "[pipeline] running init.py to create $MODEL + $MODEL_ONNX"
    Push-Location "src_python"
    & $PYTHON "init.py" --out-pt "../$MODEL" --out-onnx "../$MODEL_ONNX" --infer-precision $INFER_PRECISION
    $rc = $LASTEXITCODE
    Pop-Location
    if ($rc -ne 0) {
        Write-Host "[pipeline] init.py failed (rc=$rc) - aborting"
        Pop-Location
        exit $rc
    }
    Write-Host "[pipeline] bootstrap done."
}

# --- Zero-Disk Rust Pipe Streaming Setup -----------------------------------

    try {
        Write-Host ""
        Write-Host "[pipeline] ===== Starting Continuous Training Process ($EPOCHS Epochs) =====" -ForegroundColor Cyan

        $t0 = Get-Date
        $py_args = @("train.py", "--data", "-", "--out", "../$MODEL", "--epochs", $EPOCHS, "--batch", $BATCH, "--lr", $LEARNING_RATE, "--weight-decay", $WEIGHT_DECAY, "--compile-mode", $COMPILE_MODE, "--infer-precision", $INFER_PRECISION, "--onnx-every", $ONNX_EVERY, "--device", $PYTHON_DEVICE)
        if ($SYMMETRY) { $py_args += "--symmetry" }
        if ($CHANNELS_LAST) { $py_args += "--channels-last" }
        if ($FUSED_ADAMW) { $py_args += "--fused-adamw" }

        Write-Host "[pipeline] Mode: Zero-Disk Rust MCTS Pipe Streaming (100% RAM/VRAM, 0 files on disk)" -ForegroundColor Green
        Write-Host "[pipeline] Executing: $PYTHON train.py --data - --out $MODEL --epochs $EPOCHS --batch $BATCH"
        Push-Location "src_python"
        & $PYTHON @py_args
        $rc = $LASTEXITCODE
        Pop-Location
        $t1 = Get-Date
        $secs = Format-Seconds $t0 $t1

        if ($rc -eq 0) {
            Write-Host "[pipeline] Continuous training complete in $secs." -ForegroundColor Green
            
            $EMA_MODEL = $MODEL.Replace(".pt", "_ema.pt")
            if (Test-Path $EMA_MODEL) {
                Write-Host ""
                Write-Host "[pipeline] ===== Selecting Best Model with inference.py MCTS =====" -ForegroundColor Cyan
                Push-Location "src_python"
                & $PYTHON "evaluate.py" --model1 "../$EMA_MODEL" --model2 "../$MODEL" --games $EVAL_GAMES --think-time $EVAL_THINK_TIME --sims 0 --backend $INFER_BACKEND --selection-file "../$BEST_MODEL_FILE" --mcts --device $PYTHON_DEVICE
                Pop-Location
            }
        } else {
            Write-Host "[pipeline] Training process exited with code $rc after $secs." -ForegroundColor Red
        }
    } finally {
        Pop-Location
    }
