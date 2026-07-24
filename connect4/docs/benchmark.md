Baseline: 6:21pm 7/24/2026
=================================================================
  BENCHMARK SUMMARY & PERFORMANCE REPORT
=================================================================
  [Stage 3 - PyTorch Training (CUDA)] : 43.54s

  [Stage 1 - Rust MCTS CPU  (batch=8)] :
    - Self-Play (Inference) : 3m 02s
  [Stage 2 - Rust MCTS GPU  (batch=32)] :
    - Self-Play (Inference) : 22.25s

  Self-Play Speedup (CPU -> GPU) : 8.16x faster
=================================================================
