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

# --- Helpers ----------------------------------------------------------------
function Format-Seconds([datetime]$t0, [datetime]$t1) {
    $elapsed = [math]::Round(($t1 - $t0).TotalSeconds, 1)
    return "$elapsed s"
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
            Invoke-Expression $envCode
            Write-Host "[pipeline] Loaded configuration from config.py" -ForegroundColor Cyan
        }
    } catch {}
}

# --- Defaults / env --------------------------------------------------------
$GAMES            = if ($env:GAMES)            { [int]$env:GAMES }            else { 64 }
$SIMS             = if ($env:SIMS)             { [int]$env:SIMS }             else { 800 }
$EPOCHS           = if ($env:EPOCHS)           { [int]$env:EPOCHS }           else { 5 }
$BATCH            = if ($env:TRAIN_BATCH_SIZE) { [int]$env:TRAIN_BATCH_SIZE } else { if ($env:BATCH) { [int]$env:BATCH } else { 256 } }
$DATA             = if ($env:DATA)             { $env:DATA }                  else { "selfplay.bin" }
$MODEL            = if ($env:MODEL)            { $env:MODEL }                 else { "connect4_model.pt" }
$MODEL_ONNX       = if ($env:MODEL_ONNX)       { $env:MODEL_ONNX }            else { "connect4_model.onnx" }
$SLEEP            = if ($env:SLEEP)            { [int]$env:SLEEP }            else { 2 }
$MAX_CYCLES       = if ($env:MAX_CYCLES)       { [int]$env:MAX_CYCLES }       else { 0 }
$REPLAY_KEEP      = if ($env:REPLAY_KEEP)      { [int]$env:REPLAY_KEEP }      else { 10 }
$CPU_BATCH_SIZE   = if ($env:CPU_BATCH_SIZE)   { [int]$env:CPU_BATCH_SIZE }   else { [int]$env:NUMBER_OF_PROCESSORS }
$GPU_BATCH_SIZE   = if ($env:GPU_BATCH_SIZE)   { [int]$env:GPU_BATCH_SIZE }   else { 32 }
$SYMMETRY         = if ($env:SYMMETRY)         { [bool]($env:SYMMETRY -eq "1" -or $env:SYMMETRY -eq "true") } else { $true }
$CARGO            = if ($env:CARGO)            { $env:CARGO }                 else { "cargo" }
$venvPy          = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$PYTHON           = if ($env:PYTHON)           { $env:PYTHON }           elseif (Test-Path $venvPy) { $venvPy } else { "python" }

$RUST_DEVICE   = if ($env:RUST_DEVICE)   { $env:RUST_DEVICE }   else { if ($env:DEVICE) { $env:DEVICE } else { "auto" } }
$PYTHON_DEVICE = if ($env:PYTHON_DEVICE) { $env:PYTHON_DEVICE } else { if ($env:DEVICE) { $env:DEVICE } else { "cuda" } }
$BATCH_SIZE    = if ($RUST_DEVICE -eq "cpu") { $CPU_BATCH_SIZE } else { $GPU_BATCH_SIZE }
$FEATURES      = "--features cuda"

Write-Host "[pipeline] starting: games=$GAMES sims=$SIMS epochs=$EPOCHS batch=$BATCH batch_size=$BATCH_SIZE"
Write-Host "[pipeline] devices: rust=$RUST_DEVICE python=$PYTHON_DEVICE (python is always GPU)"
Write-Host "[pipeline] data=$DATA model=$MODEL onnx=$MODEL_ONNX"
Write-Host "[pipeline] project root: $ScriptDir"

# --- Bootstrap: if no ONNX model exists, create a random-init one. ---------
if (-not (Test-Path $MODEL_ONNX)) {
    Write-Host ""
    Write-Host "[pipeline] ===== bootstrap (no ONNX model found) ====="
    Write-Host "[pipeline] running init.py to create $MODEL + $MODEL_ONNX"
    Push-Location "src_python"
    & $PYTHON "init.py" --out-pt "../$MODEL" --out-onnx "../$MODEL_ONNX"
    $rc = $LASTEXITCODE
    Pop-Location
    if ($rc -ne 0) {
        Write-Host "[pipeline] init.py failed (rc=$rc) - aborting"
        Pop-Location
        exit $rc
    }
    Write-Host "[pipeline] bootstrap done. Starting cycle 1."
}

$cycle = 0
try {
    while ($true) {
        $cycle++
        if ($MAX_CYCLES -gt 0 -and $cycle -gt $MAX_CYCLES) {
            Write-Host "[pipeline] MAX_CYCLES=$MAX_CYCLES reached - stopping."
            break
        }
        Write-Host ""
        Write-Host "[pipeline] ===== cycle $cycle ====="

        # 1. Rust self-play (network-guided MCTS).
        $t0 = Get-Date
        # Build the cargo arg list. `--features cuda` is added when
        # RUST_DEVICE is gpu or auto so the binary has CUDA support
        # available at runtime (it's a no-op if --device cpu is selected
        # at runtime).
        $cargo_args = @("run", "--release")
        if ($FEATURES -ne "") { $cargo_args += $FEATURES.Split(' ') }
        $cargo_args += @("--manifest-path", "src_rust/Cargo.toml", "--",
                         "-g", $GAMES, "-s", $SIMS, "-b", $BATCH_SIZE, "-o", $DATA,
                         "-m", $MODEL_ONNX, "-d", $RUST_DEVICE, "-v")
        Write-Host "[pipeline] ($cycle) cargo run --release $FEATURES -- ...  (rust-device=$RUST_DEVICE)"
        & $CARGO @cargo_args
        $rc = $LASTEXITCODE
        $t1 = Get-Date
        if ($rc -ne 0) {
            $secs = Format-Seconds $t0 $t1
            Write-Host "[pipeline] ($cycle) cargo failed (rc=$rc) after $secs - sleeping $SLEEP s"
            Start-Sleep -Seconds $SLEEP
            continue
        }
        $secs = Format-Seconds $t0 $t1
        Write-Host "[pipeline] ($cycle) self-play done in $secs"

        # 2. Python train + ONNX export.
        $t0 = Get-Date
        $replayDir = Join-Path $ScriptDir "replay"
        if (-not (Test-Path $replayDir)) { New-Item -ItemType Directory -Path $replayDir | Out-Null }
        $cycleFile = Join-Path $replayDir ("selfplay_{0:D4}.bin" -f $cycle)
        # Move the just-produced selfplay.bin into the replay buffer BEFORE
        # training, so train.py sees it via --data-dir.
        Move-Item -Force $DATA $cycleFile
        # Trim to REPLAY_KEEP most-recent files.
        $oldFiles = Get-ChildItem $replayDir -Filter "selfplay_*.bin" |
            Sort-Object Name -Descending |
            Select-Object -Skip $REPLAY_KEEP
        foreach ($f in $oldFiles) { Remove-Item -Force $f.FullName }
        $replayCount = (Get-ChildItem $replayDir -Filter "selfplay_*.bin").Count
        Write-Host "[pipeline] ($cycle) saved to $cycleFile (replay buffer: $replayCount file(s))"
        Write-Host "[pipeline] ($cycle) $PYTHON train.py --data-dir ../replay --out $MODEL --epochs $EPOCHS --batch $BATCH --device $PYTHON_DEVICE"
        Push-Location "src_python"
        $symFlag = if ($SYMMETRY) { "--symmetry" } else { "" }
        & $PYTHON "train.py" --data-dir "../replay" --out "../$MODEL" --epochs $EPOCHS --batch $BATCH --device $PYTHON_DEVICE $symFlag
        $rc = $LASTEXITCODE
        Pop-Location
        $t1 = Get-Date
        if ($rc -ne 0) {
            $secs = Format-Seconds $t0 $t1
            Write-Host "[pipeline] ($cycle) train failed (rc=$rc) after $secs - keeping replay files for debug"
            Start-Sleep -Seconds $SLEEP
            continue
        }
        $secs = Format-Seconds $t0 $t1
        Write-Host "[pipeline] ($cycle) train done in $secs"
        Write-Host "[pipeline] ($cycle) sleeping $SLEEP s before next cycle"
        Start-Sleep -Seconds $SLEEP
    }
} finally {
    Pop-Location
}
