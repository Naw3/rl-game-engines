$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ScriptDir\..

# Ensure TensorRT is in PATH
if (-not ($env:PATH -like "*C:\TensorRT10\TensorRT-10.16.1.11\bin*")) {
    $env:PATH = "C:\TensorRT10\TensorRT-10.16.1.11\bin;" + $env:PATH
}

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "   BENCHMARK FP32 vs INT8 (TensorRT) FOR CONNECT4" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

# Ensure TRT cache directory exists!
New-Item -ItemType Directory -Force -Path "trt_cache" | Out-Null

$env:PYTHONUTF8 = 1

# 1. Export both models
uv run python utils/export_models_bench.py

# 2. Compile Rust
Write-Host "`n[bench] Compiling Rust with TensorRT..." -ForegroundColor Yellow
cargo build --release --manifest-path src_rust/Cargo.toml --features tensorrt

# 3. Benchmark FP32
Write-Host "`n=========================================================" -ForegroundColor Cyan
Write-Host "   RUNNING FP32 MODEL" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
src_rust/target/release/connect4_mcts.exe --duration 10 --sims 800 --batch-size 64 --model models/test_fp32.onnx --device gpu --seed 42

# 4. Benchmark INT8 (QDQ)
Write-Host "`n=========================================================" -ForegroundColor Cyan
Write-Host "   RUNNING INT8 (QDQ) MODEL" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
src_rust/target/release/connect4_mcts.exe --duration 10 --sims 800 --batch-size 64 --model models/test_int8_qdq.onnx --device gpu --seed 42

Pop-Location
Write-Host "`n[bench] Complete!" -ForegroundColor Green
