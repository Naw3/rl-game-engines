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

# --- Config file (optional) -------------------------------------------------
# pipeline.conf.ps1 sets env vars before the defaults below are read.
# Edit that file to change devices, games, sims, epochs, etc.
$_conf = Join-Path $ScriptDir "pipeline.conf.ps1"
if (Test-Path $_conf) {
    . $_conf
    Write-Host "[pipeline] loaded config: $_conf"
}

# --- Defaults / env --------------------------------------------------------
$GAMES       = if ($env:GAMES)       { [int]$env:GAMES }       else { 64 }
$SIMS        = if ($env:SIMS)        { [int]$env:SIMS }        else { 800 }
$EPOCHS      = if ($env:EPOCHS)      { [int]$env:EPOCHS }      else { 5 }
$BATCH       = if ($env:BATCH)       { [int]$env:BATCH }       else { 256 }
$DATA        = if ($env:DATA)        { $env:DATA }              else { "selfplay.bin" }
$MODEL       = if ($env:MODEL)       { $env:MODEL }             else { "connect4_model.pt" }
$MODEL_ONNX  = if ($env:MODEL_ONNX)  { $env:MODEL_ONNX }        else { "connect4_model.onnx" }
$SLEEP       = if ($env:SLEEP)       { [int]$env:SLEEP }       else { 2 }
$MAX_CYCLES  = if ($env:MAX_CYCLES)  { [int]$env:MAX_CYCLES }  else { 0 }   # 0 = infinite (default); >0 = stop after N cycles (for benchmarking)
$REPLAY_KEEP = if ($env:REPLAY_KEEP) { [int]$env:REPLAY_KEEP } else { 10 }  # Keep last N selfplay files in replay/
$BATCH_SIZE  = if ($env:BATCH_SIZE)  { [int]$env:BATCH_SIZE }  else { 32 }  # NN inference batch size; 1 = sequential, 32 = good default
$SYMMETRY    = if ($env:SYMMETRY)    { [bool]($env:SYMMETRY -eq "1" -or $env:SYMMETRY -eq "true") } else { $true }  # horizontal-flip augmentation
$CARGO       = if ($env:CARGO)       { $env:CARGO }             else { "cargo" }
$PYTHON      = if ($env:PYTHON)      { $env:PYTHON }            else { "python" }
# Inference device for the Rust self-play.
# Values: cpu | gpu | auto (default).
#   cpu  = tract-onnx in Rust (fastest on small models, no CUDA needed)
#   gpu  = ort + CUDA in Rust (needs --features cuda at build time)
#   auto = GPU if available, else CPU
#
# Per design, the Python side is ALWAYS trained on GPU (CUDA). The
# benchmark we're targeting compares (py-gpu + rust-cpu) vs
# (py-gpu + rust-gpu); the python side is the constant, the rust side
# is the variable.
#
# Backwards compat: if you only set the legacy `DEVICE` env var, both
# RUST_DEVICE and PYTHON_DEVICE inherit it. PYTHON_DEVICE defaults to
# `cuda` when nothing is set.
$RUST_DEVICE   = if ($env:RUST_DEVICE)   { $env:RUST_DEVICE }   else { if ($env:DEVICE) { $env:DEVICE } else { "auto" } }
$PYTHON_DEVICE = if ($env:PYTHON_DEVICE) { $env:PYTHON_DEVICE } else { if ($env:DEVICE) { $env:DEVICE } else { "cuda" } }
# Always build with `--features cuda` — the resulting binary supports BOTH backends
# (tract for `-d cpu`, ort+CUDA for `-d gpu`), selected at runtime via `--device`.
# This avoids a full rebuild every time we switch between the CPU and GPU benches.
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
