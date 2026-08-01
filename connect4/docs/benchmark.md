Baseline:
8:00pm 7/24/2026
=================================================================
  BENCHMARK SUMMARY & PERFORMANCE REPORT (THROUGHPUT)
=================================================================
  [Stage 3 - PyTorch Training (CUDA)] : 7662.1 samples/sec (in 60s)

  [Stage 1 - Rust MCTS CPU  (batch=8)] :
    - Generation Throughput : 4.6 samples/sec (in 60s)
    - Games Played          : 8 games (0.1 games/sec)
  [Stage 2 - Rust MCTS GPU  (batch=32)] :
    - Generation Throughput : 69.5 samples/sec (in 60s)
    - Games Played          : 122 games (2.0 games/sec)

  Self-Play Speedup (CPU -> GPU) : 15.25x faster
=================================================================

Update from requirements.txt to pyproject.toml, python 3.11 -> 3.14 - torch 2.0.6 cu124 -> 2.13.0 cu126 + other deps:
8:42pm 7/24/2026
=================================================================
  BENCHMARK SUMMARY & PERFORMANCE REPORT (THROUGHPUT)
=================================================================
  [Stage 3 - PyTorch Training (CUDA)] : 8387.1 samples/sec (in 60s)

  [Stage 1 - Rust MCTS CPU  (batch=8)] :
    - Generation Throughput : 4.5 samples/sec (in 60s)
    - Games Played          : 8 games (0.1 games/sec)
  [Stage 2 - Rust MCTS GPU  (batch=32)] :
    - Generation Throughput : 69.9 samples/sec (in 60s)
    - Games Played          : 124 games (2.1 games/sec)

  Self-Play Speedup (CPU -> GPU) : 15.53x faster
=================================================================

amp -> fp32 training - fp32 inference:
10:02am 7/25/2026
=================================================================
  BENCHMARK SUMMARY & PERFORMANCE REPORT (THROUGHPUT)
=================================================================
  [Stage 3 - PyTorch Training (CUDA)] : 17005.7 samples/sec (in 60s)

  [Stage 1 - Rust MCTS CPU  (batch=8)] :
    - Generation Throughput : 4.5 samples/sec (in 60s)
    - Games Played          : 8 games (0.1 games/sec)
  [Stage 2 - Rust MCTS GPU  (batch=32)] :
    - Generation Throughput : 78.3 samples/sec (in 60s)
    - Games Played          : 139 games (2.3 games/sec)

  Self-Play Speedup (CPU -> GPU) : 17.4x faster
=================================================================

no-compil -> compil reduce-overhead + cudastream and no gil calculation + channel last (off) + fused adamw (off):
11:03am 7/25/206
=================================================================
  BENCHMARK SUMMARY & PERFORMANCE REPORT (THROUGHPUT)
=================================================================
  [Stage 3 - PyTorch Training (CUDA)] : 25827.6 samples/sec (in 60s)

  [Stage 1 - Rust MCTS CPU  (batch=8)] :
    - Generation Throughput : 4.5 samples/sec (in 60s)
    - Games Played          : 8 games (0.1 games/sec)
  [Stage 2 - Rust MCTS GPU  (batch=32)] :
    - Generation Throughput : 81.1 samples/sec (in 60s)
    - Games Played          : 144 games (2.4 games/sec)

  Self-Play Speedup (CPU -> GPU) : 18.02x faster
=================================================================

After break ...:
2:00pm 7/28/2026
=================================================================
  BENCHMARK SUMMARY & PERFORMANCE REPORT (THROUGHPUT)
=================================================================
  [Stage 3 - PyTorch Training (CUDA)] : 15902.4 samples/sec (in 60s)

  [Stage 1 - Rust MCTS CPU  (batch=8)] :
    - Generation Throughput : 4.6 samples/sec (in 60s)
    - Games Played          : 8 games (0.1 games/sec)
  [Stage 2 - Rust MCTS GPU  (batch=32)] :
    - Generation Throughput : 67.7 samples/sec (in 60s)
    - Games Played          : 120 games (2.0 games/sec)

  Self-Play Speedup (CPU -> GPU) : 14.87x faster
=================================================================

=================================================================
  BENCHMARK SUMMARY & PERFORMANCE REPORT (THROUGHPUT)
=================================================================
  [Stage 2 - Rust MCTS GPU  (batch=32)] :
    - Generation Throughput : 151.9 samples/sec (in 60s)
    - Games Played          : 507 games (8.4 games/sec)
=================================================================

=================================================================
  BENCHMARK SUMMARY & PERFORMANCE REPORT (THROUGHPUT)
=================================================================
  [Stage 2 - Rust MCTS GPU  (batch=32)] :
    - Generation Throughput : 168.8 samples/sec (in 60s)
    - Games Played          : 564 games (9.4 games/sec)
=================================================================

=================================================================
  BENCHMARK SUMMARY & PERFORMANCE REPORT (THROUGHPUT)
=================================================================
  [Stage 2 - Rust MCTS GPU  (batch=32)] :
    - Generation Throughput : 175.5 samples/sec (in 60s)
    - Games Played          : 567 games (9.4 games/sec)
=================================================================