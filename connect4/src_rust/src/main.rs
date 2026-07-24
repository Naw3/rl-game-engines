// =============================================================================
// main.rs — Network-guided self-play data generator for Connect 4.
//
// This is the real AlphaZero self-play loop: the Rust MCTS is guided by
// a neural network (loaded from an ONNX file produced by the Python
// `train.py` step) at every expansion. There is no random rollout.
//
// What it does
// ------------
// 1. Parse CLI flags (games, sims, output, model path, RNG seed, ...).
// 2. Load the ONNX model from disk via `Network::load`. If the file
//    doesn't exist, fall back to a "null" network (uniform priors +
//    value=0) and warn loudly. The `init.py` script in the Python
//    side ensures a model exists before the very first self-play.
// 3. Spawn `num_games` independent self-play games in parallel via
//    rayon. Each worker gets its own MCTS instance (per-game tree)
//    and its own seeded RNG, but they ALL share the same
//    `Arc<Network>` — the inference is the only synchronisation point
//    and tract serialises that internally.
// 4. For each move, MCTS runs `simulations` PUCT-guided simulations,
//    using the network's policy as the prior and the network's value
//    as the leaf value. The visit-count distribution is the policy
//    target for the training data.
// 5. After each game, walk the recorded (state, policy) pairs and
//    assign a value `z ∈ {-1, 0, +1}` to each from the perspective
//    of the player to move at that state.
// 6. Serialise all (state, policy, value) triples into a single
//    binary file with the C4D1 format (see docs/architecture.md §3).
//
// C4D1 binary format
// ------------------
//   Header (16 bytes):
//     4 bytes : magic = b"C4D1"
//     4 bytes : u32 LE  sample count N
//     8 bytes : reserved, zero
//
//   Sample (56 bytes) — repeated N times:
//     8 bytes : u64 LE  own bitboard       (current player's pieces)
//     8 bytes : u64 LE  opponent bitboard  (the other player's pieces)
//     8 bytes : u64 LE  turn mask          (all 1s — constant bias plane)
//     28 bytes: 7 × f32 LE  MCTS policy π
//     4 bytes : f32 LE    game outcome z  (from THIS sample's perspective)
//
//   Total: 16 + 56·N bytes. The Python `dataset.py` reads this directly
//   via `numpy.frombuffer`.
// =============================================================================

mod bitboard;
mod mcts;
mod network;

use bitboard::{Board, MoveResult};
use mcts::{MCTS, MCTSConfig};
use network::Network;

use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

/// 56 bytes per sample. Manually serialized to keep the wire format tight.
///
/// `policy` is stored as `Vec<f32>` rather than `[f32; 7]` because the
/// MCTS returns a `Vec<f32>` (length always 7) and the on-disk format
/// writes 7 `f32`s regardless of the in-memory representation. This
/// keeps `play_game` zero-copy: it just hands the policy straight to
/// the Sample without any fixed-size array shuffle.
#[derive(Clone)]
struct Sample {
    own: u64,
    opp: u64,
    turn_mask: u64,
    policy: Vec<f32>,
    value: f32,
}

#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
struct ConfigJson {
    mcts: Option<MctsJson>,
    gui: Option<GuiJson>,
    paths: Option<PathsJson>,
    device: Option<DeviceJson>,
}

#[derive(Debug, serde::Deserialize)]
struct MctsJson {
    games: Option<usize>,
    sims: Option<usize>,
    cpu_batch_size: Option<usize>,
    gpu_batch_size: Option<usize>,
    c_puct: Option<f32>,
    dirichlet_alpha: Option<f32>,
    dirichlet_epsilon: Option<f32>,
    temperature: Option<f32>,
    seed: Option<u64>,
    warmup: Option<usize>,
    bench_iterations: Option<usize>,
}

#[derive(Debug, serde::Deserialize)]
struct GuiJson {
    progress_bar_width: Option<usize>,
}

#[derive(Debug, serde::Deserialize)]
struct PathsJson {
    model_onnx: Option<String>,
    selfplay_bin: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
struct DeviceJson {
    rust_device: Option<String>,
}

fn load_config_from_python() -> Option<ConfigJson> {
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent()?;
    let config_py = parent.join("config.py");
    if !config_py.exists() {
        eprintln!("[selfplay] WARNING: config.py not found at {}; using fallback defaults", config_py.display());
        return None;
    }

    let venv_py = parent.join(".venv").join("Scripts").join("python.exe");
    let py_cmd = if venv_py.exists() {
        venv_py.to_string_lossy().to_string()
    } else {
        "python".to_string()
    };

    let output = match std::process::Command::new(&py_cmd)
        .arg(&config_py)
        .current_dir(parent)
        .output() {
            Ok(o) => o,
            Err(e) => {
                eprintln!("[selfplay] WARNING: Failed to execute python config.py ({}); using fallback defaults", e);
                return None;
            }
        };

    if output.status.success() {
        let stdout_str = String::from_utf8_lossy(&output.stdout);
        match serde_json::from_str::<ConfigJson>(&stdout_str) {
            Ok(cfg) => return Some(cfg),
            Err(e) => eprintln!("[selfplay] WARNING: Failed to parse JSON output from config.py ({}); using fallback defaults", e),
        }
    } else {
        let stderr_str = String::from_utf8_lossy(&output.stderr);
        eprintln!("[selfplay] WARNING: config.py exited with failure ({}); using fallback defaults", stderr_str.trim());
    }
    None
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    // --- Subcommand dispatch ---------------------------------------------
    if args.len() > 1 && !args[1].starts_with('-') {
        match args[1].as_str() {
            "benchmark" => return run_benchmark(&args[2..]),
            "help" | "--help" | "-h" => {
                print_help();
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown subcommand: {}", other);
                print_help();
                return ExitCode::from(2);
            }
        }
    }

    // --- Default: self-play mode -----------------------------------------
    let json_cfg = load_config_from_python();

    // Defaults initialized from config.json if present, fallback to built-in defaults.
    let mut num_games: usize = json_cfg.as_ref().and_then(|c| c.mcts.as_ref()?.games).unwrap_or(64);
    let mut simulations: usize = json_cfg.as_ref().and_then(|c| c.mcts.as_ref()?.sims).unwrap_or(mcts::DEFAULT_SIMS);
    let mut batch_size_opt: Option<usize> = None;
    let mut seed: u64 = json_cfg.as_ref().and_then(|c| c.mcts.as_ref()?.seed).unwrap_or(0xC0FFEE_u64);
    let mut output: String = json_cfg.as_ref().and_then(|c| c.paths.as_ref()?.selfplay_bin.clone()).unwrap_or_else(|| "selfplay.bin".to_string());
    let mut model_path: String = json_cfg.as_ref().and_then(|c| c.paths.as_ref()?.model_onnx.clone()).unwrap_or_else(|| default_model_path().to_string_lossy().to_string());
    let mut temperature: f32 = json_cfg.as_ref().and_then(|c| c.mcts.as_ref()?.temperature).unwrap_or(1.0);
    let mut c_puct: f32 = json_cfg.as_ref().and_then(|c| c.mcts.as_ref()?.c_puct).unwrap_or(mcts::DEFAULT_C_PUCT);
    let dirichlet_alpha: f32 = json_cfg.as_ref().and_then(|c| c.mcts.as_ref()?.dirichlet_alpha).unwrap_or(mcts::DEFAULT_DIRICHLET_ALPHA);
    let mut dirichlet_eps: f32 = json_cfg.as_ref().and_then(|c| c.mcts.as_ref()?.dirichlet_epsilon).unwrap_or(mcts::DEFAULT_DIRICHLET_EPSILON);
    let mut device: network::Device = network::Device::Auto;
    let mut verbose: bool = false;

    // Tiny flag parser. Supports `--flag value` and `--flag=value`.
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        let next = |i: &mut usize| -> &String {
            *i += 1;
            &args[*i]
        };
        match a.as_str() {
            "--games" | "-g" => {
                num_games = next(&mut i).parse().unwrap_or(64);
            }
            "--sims" | "-s" => {
                simulations = next(&mut i).parse().unwrap_or(mcts::DEFAULT_SIMS);
            }
            "--batch-size" | "-b" => {
                let b: usize = next(&mut i).parse().unwrap_or(32);
                if b < 1 {
                    eprintln!("--batch-size must be >= 1, got {}", b);
                    return ExitCode::from(2);
                }
                batch_size_opt = Some(b);
            }
            "--c-puct" => {
                c_puct = next(&mut i).parse().unwrap_or(mcts::DEFAULT_C_PUCT);
            }
            "--seed" => {
                seed = next(&mut i).parse().unwrap_or(0xC0FFEE_u64);
            }
            "--output" | "-o" => {
                output = next(&mut i).clone();
            }
            "--model" | "-m" => {
                model_path = next(&mut i).clone();
            }
            "--device" | "-d" => {
                let raw = next(&mut i).clone();
                match network::Device::from_str(&raw) {
                    Ok(d) => device = d,
                    Err(e) => {
                        eprintln!("{}", e);
                        return ExitCode::from(2);
                    }
                }
            }
            "--temperature" | "-t" => {
                temperature = next(&mut i).parse().unwrap_or(1.0);
            }
            "--no-noise" => {
                dirichlet_eps = 0.0;
            }
            "--verbose" | "-v" => verbose = true,
            "--help" | "-h" => {
                print_help();
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown flag: {}", other);
                print_help();
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    if num_games == 0 {
        eprintln!("--games must be > 0");
        return ExitCode::from(2);
    }

    // Load the network. We try the .onnx first; if it's missing, fall
    // back to the null network (uniform priors, value=0). The null
    // network is a safety net — the orchestrator's init.py ensures
    // a real model exists before the very first self-play.
    let network: Arc<Network> = match Network::load(&model_path, device) {
        Ok(net) => {
            if verbose {
                eprintln!(
                    "[selfplay] loaded model from {} on {:?}",
                    model_path,
                    net.device()
                );
            }
            Arc::new(net)
        }
        Err(e) => {
            eprintln!(
                "[selfplay] WARNING: could not load model from {} ({})",
                model_path, e
            );
            eprintln!(
                "[selfplay] WARNING: falling back to null network (uniform priors, value=0)"
            );
            eprintln!(
                "[selfplay] WARNING: run `python init.py` to bootstrap a random-init model"
            );
            Arc::new(Network::null())
        }
    };

    let batch_size = match batch_size_opt {
        Some(b) => b,
        None => {
            let cpu_b = json_cfg.as_ref().and_then(|c| c.mcts.as_ref()?.cpu_batch_size).unwrap_or_else(|| rayon::current_num_threads());
            let gpu_b = json_cfg.as_ref().and_then(|c| c.mcts.as_ref()?.gpu_batch_size).unwrap_or(32);
            match network.device() {
                network::Device::Cpu => cpu_b,
                network::Device::Gpu | network::Device::Auto => gpu_b,
            }
        }
    };

    let config = MCTSConfig {
        simulations,
        batch_size,
        c_puct,
        dirichlet_alpha,
        dirichlet_epsilon: dirichlet_eps,
        temperature,
    };

    if verbose {
        eprintln!(
            "[selfplay] games={} sims={} batch_size={} seed=0x{:X} output={} tau={} eps={}",
            num_games, simulations, batch_size, seed, output, temperature, dirichlet_eps
        );
        // Warn if --device gpu is requested but cuda feature wasn't compiled in.
        if device == network::Device::Gpu && !cfg!(feature = "cuda") {
            eprintln!(
                "[selfplay] WARNING: --device gpu requested but this binary was built WITHOUT \
                 the 'cuda' feature. Re-run `cargo build --release --features cuda` to enable GPU."
            );
        }
        eprintln!(
            "[selfplay] network: {}",
            if network.is_null() {
                "NULL (uniform priors)"
            } else {
                "loaded from .onnx"
            }
        );
    }

    let start = Instant::now();

    // Parallel self-play. Each worker builds its own MCTS (per-game
    // tree) and its own RNG, but ALL workers share the same
    // Arc<Network>. The network is the only synchronisation point
    // across threads; tract serialises concurrent run() calls internally.
    use std::sync::atomic::{AtomicUsize, Ordering};
    let completed_games = Arc::new(AtomicUsize::new(0));
    let total_plies = Arc::new(AtomicUsize::new(0));

    if verbose {
        let bar_width = json_cfg.as_ref().and_then(|c| c.gui.as_ref()?.progress_bar_width).unwrap_or(20);
        let bar = format!("[{}]", " ".repeat(bar_width));
        eprint!(
            "\r\x1B[K[selfplay]   0% {}    0/{} | avg 0.0 moves | 0.0s | 0.0s/game | ETA: --",
            bar, num_games
        );
        let _ = std::io::stderr().flush();
    }

    let samples: Vec<Sample> = (0..num_games)
        .into_par_iter()
        .enumerate()
        .map(|(game_idx, _)| {
            let game_seed = seed
                .wrapping_add(game_idx as u64)
                .wrapping_mul(0x9E3779B97F4A7C15);
            let mut rng = StdRng::seed_from_u64(game_seed);
            let mut mcts = MCTS::new(config, Arc::clone(&network));
            let game_samples = play_game(&mut mcts, &mut rng);

            let completed = completed_games.fetch_add(1, Ordering::SeqCst) + 1;
            let current_plies = total_plies.fetch_add(game_samples.len(), Ordering::SeqCst) + game_samples.len();

            if verbose {
                let elapsed = start.elapsed().as_secs_f64();
                let avg_plies = current_plies as f64 / completed as f64;
                let sec_per_game = elapsed / completed as f64;
                let remaining_games = num_games.saturating_sub(completed);
                let eta_sec = sec_per_game * remaining_games as f64;

                let format_dur = |s: f64| -> String {
                    if s < 0.001 {
                        "< 1ms".to_string()
                    } else if s < 1.0 {
                        format!("{:.0}ms", s * 1000.0)
                    } else if s < 60.0 {
                        format!("{:.1}s", s)
                    } else if s < 3600.0 {
                        let u = s.round() as u64;
                        format!("{}m {:02}s", u / 60, u % 60)
                    } else {
                        let u = s.round() as u64;
                        format!("{}h {:02}m {:02}s", u / 3600, (u % 3600) / 60, u % 60)
                    }
                };

                let time_str = format_dur(elapsed);
                let eta_str = format_dur(eta_sec);
                let per_game_str = format!("{}/game", format_dur(sec_per_game));

                let pct = (completed * 100) / num_games;
                let bar_width = json_cfg.as_ref().and_then(|c| c.gui.as_ref()?.progress_bar_width).unwrap_or(20);
                let filled = (completed * bar_width) / num_games;
                let empty = bar_width - filled;
                let bar = format!(
                    "[{}{}{}]",
                    "=".repeat(filled),
                    if filled < bar_width { ">" } else { "" },
                    " ".repeat(if empty > 0 { empty - 1 } else { 0 })
                );

                eprint!(
                    "\r\x1B[K[selfplay] {:3}% {} {:4}/{} | avg {:.1} moves | {} | {} | ETA: {}",
                    pct, bar, completed, num_games, avg_plies, time_str, per_game_str, eta_str
                );
                let _ = std::io::stderr().flush();
            }
            game_samples
        })
        .reduce(Vec::new, |mut a, mut b| {
            a.append(&mut b);
            a
        });

    if verbose {
        eprintln!();
        eprintln!(
            "[selfplay] {} games → {} samples in {:?}",
            num_games,
            samples.len(),
            start.elapsed()
        );
    }

    match write_binary(&output, &samples) {
        Ok(()) => {
            if verbose {
                let bytes = 16 + 56 * samples.len();
                eprintln!("[selfplay] wrote {} bytes to {}", bytes, output);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("[selfplay] write failed: {}", e);
            ExitCode::from(1)
        }
    }
}

fn print_help() {
    eprintln!("connect4-selfplay — generate AlphaZero-style training data");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("    connect4-selfplay [OPTIONS]");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("    -g, --games <N>        Number of self-play games (default 64)");
    eprintln!("    -s, --sims <N>         MCTS simulations per move (default 800)");
    eprintln!("    -b, --batch-size <N>   NN inference batch size (default 32). 1 = sequential.");
    eprintln!("        --seed <u64>       Master RNG seed (default 0xC0FFEE)");
    eprintln!("    -o, --output <PATH>    Output binary file (default selfplay.bin)");
    eprintln!("    -m, --model <PATH>     ONNX model path (default connect4_model.onnx)");
    eprintln!("    -d, --device <KIND>    Inference device: cpu, gpu, auto (default auto)");
    eprintln!("                           cpu = tract-onnx (fastest, pure Rust)");
    eprintln!("                           gpu = ort + CUDA (needs --features cuda at build)");
    eprintln!("                           auto = gpu if available else cpu");
    eprintln!("    -t, --temperature <F>  Visit-count temperature (default 1.0)");
    eprintln!("        --no-noise         Disable Dirichlet noise at the root");
    eprintln!("    -v, --verbose          Print progress to stderr");
    eprintln!("    -h, --help             Print this help");
    eprintln!();
    eprintln!("SUBCOMMANDS:");
    eprintln!("    benchmark [OPTIONS]    Time pure NN inference (no MCTS, no self-play).");
    eprintln!("                           Compare backends by running `--device cpu` vs `--device gpu`.");
    eprintln!();
    eprintln!("    benchmark options:");
    eprintln!("        -n, --iterations <N>   Number of forward passes (default 1000)");
    eprintln!("        -m, --model <PATH>     ONNX model path (default connect4_model.onnx)");
    eprintln!("        -d, --device <KIND>    cpu | gpu | auto (default auto)");
    eprintln!("        --warmup <N>           Warmup iterations before timing (default 20)");
}

/// Default model path: `<project_root>/connect4_model.onnx`. The
/// project root is the parent of `CARGO_MANIFEST_DIR` (i.e. the
/// directory holding this crate's Cargo.toml, which is `src_rust/`).
/// Works from any cwd.
fn default_model_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")))
        .join("connect4_model.onnx")
}

/// Run a pure NN inference benchmark: load the model, run N forward
/// passes, report mean/throughput. Used to measure the per-call cost
/// of the inference backend (tract-cpu vs ort-cpu vs ort-gpu) so you
/// can decide if the GPU setup is worth the build/runtime cost.
fn run_benchmark(args: &[String]) -> ExitCode {
    let json_cfg = load_config_from_python();
    let mut iterations: usize = json_cfg.as_ref().and_then(|c| c.mcts.as_ref()?.bench_iterations).unwrap_or(1000);
    let mut warmup: usize = json_cfg.as_ref().and_then(|c| c.mcts.as_ref()?.warmup).unwrap_or(20);
    let mut model_path = json_cfg.as_ref().and_then(|c| c.paths.as_ref()?.model_onnx.clone()).unwrap_or_else(|| default_model_path().to_string_lossy().to_string());
    let mut device = network::Device::Auto;

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let next = |k: &mut usize| -> &String {
            *k += 1;
            &args[*k]
        };
        match a.as_str() {
            "--iterations" | "-n" => {
                iterations = next(&mut i).parse().unwrap_or(1000);
            }
            "--warmup" => {
                warmup = next(&mut i).parse().unwrap_or(20);
            }
            "--model" | "-m" => {
                model_path = next(&mut i).clone();
            }
            "--device" | "-d" => {
                let raw = next(&mut i).clone();
                match network::Device::from_str(&raw) {
                    Ok(d) => device = d,
                    Err(e) => {
                        eprintln!("{}", e);
                        return ExitCode::from(2);
                    }
                }
            }
            _ => {
                eprintln!("unknown benchmark flag: {}", a);
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    eprintln!(
        "[benchmark] model={} device={:?} iterations={} warmup={}",
        model_path, device, iterations, warmup
    );

    let network = match network::Network::load(&model_path, device) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("[benchmark] failed to load model: {}", e);
            return ExitCode::from(1);
        }
    };
    eprintln!("[benchmark] backend bound: {:?}", network.device());

    // Use a fresh empty board as the benchmark input. The cost is
    // representative of any typical position (the model's compute is
    // dominated by the convolutions, which are the same for any input).
    let board = Board::new();

    // Warmup: prime the caches, JIT-compile kernels, etc.
    eprintln!("[benchmark] warmup...");
    for _ in 0..warmup {
        let _ = network.evaluate(board);
    }

    // Timed run.
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = network.evaluate(board);
    }
    let elapsed = start.elapsed();

    let mean_nanos = elapsed.as_nanos() as f64 / iterations as f64;
    let mean_micros = mean_nanos / 1000.0;
    let throughput = iterations as f64 / elapsed.as_secs_f64();

    eprintln!(
        "[benchmark] total: {:?}  mean: {:.2} µs/call  throughput: {:.0} calls/sec",
        elapsed, mean_micros, throughput
    );
    eprintln!(
        "[benchmark] (compare with: cargo run --release -- benchmark -d cpu vs -d gpu)"
    );
    ExitCode::SUCCESS
}

/// Play one self-play game. Returns the (state, policy, value) samples.
fn play_game<R: Rng + ?Sized>(mcts: &mut MCTS, rng: &mut R) -> Vec<Sample> {
    let mut board = Board::new();
    let mut samples: Vec<Sample> = Vec::with_capacity(42);
    let mut last_was_terminal: Option<MoveResult> = None;

    loop {
        // Run network-guided MCTS to get the policy at this state.
        let (policy, _q) = mcts.run(board, rng);

        // Sample the move BEFORE pushing the sample so we still own
        // the `Vec<f32>` (the policy would otherwise be moved into
        // the Sample and we couldn't borrow it for sampling).
        let legal = board.legal_moves();
        let action = sample_action_from_policy(&policy, legal, rng);

        // Record the sample (value filled in below).
        let [own, opp, turn_mask] = board.to_planes();
        samples.push(Sample {
            own,
            opp,
            turn_mask,
            policy,
            value: 0.0,
        });

        // Apply. After `make_move`, `board` is the child's state (own/opp
        // swapped). The result tells us if the game is over.
        let result = board.make_move(action);
        match result {
            MoveResult::Win | MoveResult::Draw => {
                last_was_terminal = Some(result);
                break;
            }
            MoveResult::Continue => { /* keep going */ }
            MoveResult::Illegal => {
                // Should never happen — MCTS only returns policies with
                // zero prior on illegal columns. Bail out gracefully.
                break;
            }
        }
    }

    // Assign values. The winner is always the LAST mover (the player who made
    // the move at index n-1). So the player to move at index n-1 gets +1.0.
    // The opponent (player to move at index n-2) lost, so they get -1.0.
    // The player to move at index n-3 gets +1.0, and so on, alternating backwards.
    // On a draw, all values are 0.
    let n = samples.len();
    if let Some(MoveResult::Draw) = last_was_terminal {
        for s in samples.iter_mut() {
            s.value = 0.0;
        }
    } else {
        for (i, s) in samples.iter_mut().enumerate() {
            s.value = if (n - 1 - i) % 2 == 0 { 1.0 } else { -1.0 };
        }
    }

    samples
}

/// Sample one action from a categorical distribution given by `policy`.
/// `legal` is a 7-bit bitmask; columns with a 0 bit are skipped.
fn sample_action_from_policy<R: Rng + ?Sized>(
    policy: &[f32],
    legal: u8,
    rng: &mut R,
) -> usize {
    let r: f32 = rng.gen();
    let mut cumsum = 0.0f32;
    for c in 0..7 {
        if legal & (1 << c) == 0 {
            continue;
        }
        let p = policy[c].max(0.0);
        cumsum += p;
        if r < cumsum {
            return c;
        }
    }
    // Numerical fallthrough: pick the last legal column.
    for c in (0..7).rev() {
        if legal & (1 << c) != 0 {
            return c;
        }
    }
    0
}

/// Write the C4D1 binary file.
fn write_binary(path: &str, samples: &[Sample]) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::with_capacity(1 << 20, f);

    // Header.
    w.write_all(b"C4D1")?;
    w.write_all(&(samples.len() as u32).to_le_bytes())?;
    w.write_all(&[0u8; 8])?;

    // Samples.
    for s in samples {
        w.write_all(&s.own.to_le_bytes())?;
        w.write_all(&s.opp.to_le_bytes())?;
        w.write_all(&s.turn_mask.to_le_bytes())?;
        for &p in &s.policy {
            w.write_all(&p.to_le_bytes())?;
        }
        w.write_all(&s.value.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}
