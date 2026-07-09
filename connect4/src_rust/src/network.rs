// =============================================================================
// network.rs — ONNX inference with two backends and a CLI-selected device.
//
// Why two backends?
// -----------------
// We support BOTH `tract-onnx` (CPU) AND `ort` (GPU via CUDA) so the user
// can pick the right tool per scenario:
//
//   * `--device cpu`  → tract-onnx. Pure Rust, no FFI overhead. Fastest
//     for our 50K-param model. Always available (no external dep).
//
//   * `--device gpu`  → ort with CUDA execution provider. Requires
//     `--features cuda` at build time, plus `onnxruntime-gpu` and the
//     CUDA toolkit at runtime.
//
//   * `--device auto` → tries GPU first, falls back to CPU if CUDA is
//     unavailable or fails to initialise.
//
// GPU Inference Architecture: Centralised Dispatcher
// ---------------------------------------------------
// When `--device gpu`, a single dedicated OS thread owns the one-and-only
// ort `Session`. All 64 rayon MCTS worker threads send their board states
// via a `crossbeam_channel` bounded MPSC queue. The dispatcher:
//   1. Blocks on the first request (recv()).
//   2. Drains additional pending requests without blocking (try_recv()),
//      up to `MAX_DISPATCHER_BATCH` boards.
//   3. Builds one (N, 3, 6, 7) tensor and runs a single ort forward pass.
//   4. Sends each result back to its requester via a oneshot sync channel.
// This achieves maximum GPU utilisation (large batches) while using only
// one ort Session (minimum VRAM).
//
// CPU Inference Architecture: Tract (no pooling needed)
// ------------------------------------------------------
// On CPU, we use tract-onnx. `RunnableModel` is `Send + Sync` and
// serialises concurrent `run` calls internally — safe by construction,
// no extra locking needed on the Rust side.
//
// What "null" means
// -----------------
// `Network::null()` returns a Network with no backend. It produces
// uniform priors + value=0. The orchestrator's `init.py` ensures a real
// model exists before the very first self-play.
// =============================================================================

use crate::bitboard::{col_mask, Board};
use std::path::Path;
use std::sync::Arc;
use crossbeam_channel::{bounded, Receiver, Sender};

// --- tract imports (CPU backend) ------------------------------------------
use tract_onnx::prelude::{
    Framework, InferenceModelExt, RunnableModel, Tensor,
    TypedFact, TypedModel, TypedOp, tvec,
};

// --- ort imports (GPU backend) --------------------------------------------
use ort::execution_providers::CPUExecutionProvider;
#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;
use ort::session::builder::SessionBuilder;
use ort::session::Session;
use ort::value::Value;

/// Maximum number of boards batched together in a single GPU forward pass.
/// With 64 concurrent MCTS threads, each submitting 1 board per select step,
/// 64 is the natural ceiling. Larger = more GPU utilisation; smaller = lower
/// latency per individual request.
const MAX_DISPATCHER_BATCH: usize = 64;

/// Inference device. `Auto` resolves at session load time based on
/// whether the `cuda` feature was compiled in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu,
    Auto,
}

impl Device {
    /// Resolve `Auto` to a concrete device.
    pub fn resolve(self) -> Self {
        match self {
            Device::Auto => {
                if cfg!(feature = "cuda") {
                    Device::Gpu
                } else {
                    Device::Cpu
                }
            }
            other => other,
        }
    }

    /// Parse from a CLI string (`"cpu"`, `"gpu"`, `"auto"`).
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Device::Cpu),
            "gpu" | "cuda" => Ok(Device::Gpu),
            "auto" => Ok(Device::Auto),
            _ => Err(format!(
                "invalid device '{}' (expected: cpu, gpu, auto)",
                s
            )),
        }
    }
}

/// The inference network. Holds an optional `Backend` (Tract or OrtDispatcher)
/// shared via `Arc` across all rayon worker threads.
pub struct Network {
    backend: Option<Arc<Backend>>,
    device: Device,
}

/// Inference backend enum — dispatched at `evaluate` time.
enum Backend {
    /// Pure-Rust CPU inference via tract-onnx. `RunnableModel` is Send+Sync.
    Tract(Arc<TractSession>),
    /// GPU inference via a centralised dispatcher thread. All worker threads
    /// push `InferRequest` structs into the shared sender; the dispatcher owns
    /// the single ort `Session` and batches incoming requests.
    OrtDispatcher(InferQueue),
}

/// The sender half of the dispatcher's request queue. Cloned cheaply per thread.
type InferQueue = Sender<InferRequest>;

/// One inference request sent from an MCTS worker to the GPU dispatcher.
struct InferRequest {
    /// The board state to evaluate.
    board: Board,
    /// Channel to send the result back on. The dispatcher calls `send` after
    /// completing the batch that contains this request.
    reply: Sender<Eval>,
}

type TractSession = RunnableModel<TypedFact, Box<dyn TypedOp>, TypedModel>;

/// Output of a single NN forward pass. Policy is over the 7 columns
/// (sums to 1 over the 7 actions). Value is the position evaluation
/// in [-1, +1] from the perspective of the player to move.
#[derive(Debug, Clone, Copy)]
pub struct Eval {
    pub policy: [f32; 7],
    pub value: f32,
}

impl Network {
    /// Load an ONNX model from disk using the backend implied by `device`.
    pub fn load<P: AsRef<Path>>(path: P, device: Device) -> Result<Self, Box<dyn std::error::Error>> {
        let device = device.resolve();
        let backend = match device {
            Device::Cpu => {
                let sess = load_tract(path.as_ref())?;
                Backend::Tract(Arc::new(sess))
            }
            Device::Gpu => {
                // Spawn the dedicated GPU dispatcher thread. It owns the
                // single ort Session and processes batched requests.
                let sender = start_dispatcher(path.as_ref())?;
                Backend::OrtDispatcher(sender)
            }
            Device::Auto => unreachable!("Device::Auto must be resolved before load"),
        };
        Ok(Network {
            backend: Some(Arc::new(backend)),
            device,
        })
    }

    /// Construct a "null" network with no backend. Returns uniform
    /// priors and value=0 — used as a safety net when the ONNX
    /// file is missing.
    pub fn null() -> Self {
        Network {
            backend: None,
            device: Device::Cpu,
        }
    }

    /// True if this is a null network (no underlying backend).
    pub fn is_null(&self) -> bool {
        self.backend.is_none()
    }

    /// The device this network was bound to.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Evaluate a single position. Dispatches to the right backend.
    pub fn evaluate(&self, board: Board) -> Eval {
        match self.backend.as_deref() {
            Some(Backend::Tract(sess)) => tract_eval(sess, board),
            Some(Backend::OrtDispatcher(queue)) => dispatcher_eval(queue, board),
            None => null_eval(board),
        }
    }

    /// Evaluate a batch of positions. On GPU, each board is submitted
    /// individually to the dispatcher — the dispatcher will merge them with
    /// concurrent requests from other threads into a single large GPU batch.
    /// On CPU (tract), they are forwarded as a single batch directly.
    pub fn evaluate_batch(&self, boards: &[Board]) -> Vec<Eval> {
        if boards.is_empty() {
            return Vec::new();
        }
        match self.backend.as_deref() {
            Some(Backend::Tract(sess)) => tract_batch_eval(sess, boards),
            Some(Backend::OrtDispatcher(queue)) => dispatcher_batch_eval(queue, boards),
            None => boards.iter().map(|&b| null_eval(b)).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Backend loaders
// ---------------------------------------------------------------------------

/// Load the model with tract-onnx. CPU inference, fastest on this size.
///
/// Note: the Python exporter uses `dynamic_shapes` with the batch dimension
/// symbolic, so the ONNX graph accepts any batch size. We therefore do NOT
/// call `with_input_fact` here — that would lock the batch dim to 1.
fn load_tract(path: &Path) -> Result<TractSession, Box<dyn std::error::Error>> {
    let session = tract_onnx::onnx()
        .model_for_path(path)?
        .into_optimized()?
        .into_runnable()?;
    Ok(session)
}

/// Load the model with ort. CUDA if `device == Gpu` and `cuda` feature
/// compiled in; CPU otherwise.
fn load_ort(path: &Path, device: Device) -> Result<Session, Box<dyn std::error::Error>> {
    let mut builder = SessionBuilder::new()?;

    #[cfg(feature = "cuda")]
    {
        if device == Device::Gpu {
            builder = builder.with_execution_providers([
                CUDAExecutionProvider::default()
                    .with_device_id(0)
                    .build()
                    .error_on_failure(),
                CPUExecutionProvider::default().build(),
            ])?;
        }
    }

    if device == Device::Cpu {
        builder = builder.with_execution_providers([
            CPUExecutionProvider::default().build(),
        ])?;
    }

    let session = builder.commit_from_file(path)?;
    Ok(session)
}

// ---------------------------------------------------------------------------
// GPU Dispatcher
// ---------------------------------------------------------------------------

/// Spawn the GPU dispatcher thread. Returns the `Sender` half of the request
/// queue; the `Receiver` is owned by the dispatcher thread.
///
/// The dispatcher thread exits automatically when all senders are dropped
/// (i.e., when the `Network` is dropped at end of self-play).
fn start_dispatcher(path: &Path) -> Result<InferQueue, Box<dyn std::error::Error>> {
    // Bounded queue so backpressure prevents unbounded queuing.
    // 256 slots: with 64 threads each queuing ~32 requests per sim batch,
    // this gives enough headroom without unbounded memory growth.
    let (tx, rx): (Sender<InferRequest>, Receiver<InferRequest>) = bounded(256);

    let session = load_ort(path, Device::Gpu)?;

    std::thread::Builder::new()
        .name("ort-gpu-dispatcher".to_string())
        .spawn(move || run_dispatcher(session, rx))?;

    Ok(tx)
}

/// The main loop of the GPU dispatcher thread.
fn run_dispatcher(mut session: Session, rx: Receiver<InferRequest>) {
    loop {
        // Block until at least one request arrives.
        let first = match rx.recv() {
            Ok(req) => req,
            Err(_) => return, // all senders dropped → clean shutdown
        };

        // Drain as many additional requests as are immediately available,
        // then wait up to 500 µs for stragglers before flushing.
        // This "batching window" prevents degenerate batch-size-1 flushes
        // when only a few MCTS games remain active at end-of-cycle.
        let mut batch: Vec<InferRequest> = Vec::with_capacity(MAX_DISPATCHER_BATCH);
        batch.push(first);

        let deadline = std::time::Instant::now() + std::time::Duration::from_micros(500);
        loop {
            if batch.len() >= MAX_DISPATCHER_BATCH {
                break;
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            match rx.recv_timeout(remaining) {
                Ok(req) => batch.push(req),
                Err(_) => break, // timeout or disconnected
            }
        }

        let n = batch.len();

        // Build the (N, 3, 6, 7) input tensor.
        let boards: Vec<Board> = batch.iter().map(|r| r.board).collect();
        let (shape, data) = boards_to_input(&boards);
        let input_val = match Value::from_array((shape, data)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[ort-dispatcher] failed to build input tensor: {e}");
                for req in batch {
                    let _ = req.reply.send(null_eval(req.board));
                }
                continue;
            }
        };

        // Single GPU forward pass — the key to saturating the 4060.
        let outputs = match session.run(ort::inputs!["input" => input_val]) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("[ort-dispatcher] inference error: {e}");
                for req in batch {
                    let _ = req.reply.send(null_eval(req.board));
                }
                continue;
            }
        };

        let policy_view = outputs[0]
            .try_extract_array::<f32>()
            .expect("ort policy extraction failed");
        let value_view = outputs[1]
            .try_extract_array::<f32>()
            .expect("ort value extraction failed");
        let policy_data: Vec<f32> = policy_view.iter().copied().collect();
        let value_data: Vec<f32> = value_view.iter().copied().collect();

        // Dispatch results back to each requester.
        for (i, req) in batch.into_iter().enumerate() {
            let row = &policy_data[i * 7..(i + 1) * 7];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_row: Vec<f32> = row.iter().map(|x| (x - max).exp()).collect();
            let sum: f32 = exp_row.iter().sum();
            let mut policy = [0.0f32; 7];
            for c in 0..7 {
                policy[c] = exp_row[c] / sum;
            }
            let eval = Eval {
                policy,
                value: value_data[i],
            };
            let _ = req.reply.send(eval); // ignore if receiver already dropped
        }

        let _ = n; // suppress unused warning in non-debug builds
    }
}

/// Submit one board to the GPU dispatcher and wait for its result.
fn dispatcher_eval(queue: &InferQueue, board: Board) -> Eval {
    let (reply_tx, reply_rx) = bounded(1);
    queue
        .send(InferRequest { board, reply: reply_tx })
        .expect("GPU dispatcher thread has exited unexpectedly");
    reply_rx
        .recv()
        .expect("GPU dispatcher closed reply channel")
}

/// Submit N boards to the GPU dispatcher, collecting results in order.
/// The dispatcher merges these with concurrent requests from other
/// threads into the largest possible batch before executing.
fn dispatcher_batch_eval(queue: &InferQueue, boards: &[Board]) -> Vec<Eval> {
    // Allocate one reply channel per board upfront.
    let channels: Vec<(Sender<Eval>, Receiver<Eval>)> =
        (0..boards.len()).map(|_| bounded(1)).collect();

    // Send all requests before waiting on any reply — maximises batching.
    for (board, (tx, _)) in boards.iter().zip(channels.iter()) {
        queue
            .send(InferRequest { board: *board, reply: tx.clone() })
            .expect("GPU dispatcher thread has exited unexpectedly");
    }

    // Collect results in submission order.
    channels
        .into_iter()
        .map(|(_, rx)| rx.recv().expect("GPU dispatcher closed reply channel"))
        .collect()
}

// ---------------------------------------------------------------------------
// Tensor construction (shared by both backends)
// ---------------------------------------------------------------------------

/// Build a (1, 3, 6, 7) f32 tensor from a single Board.
///
/// Plane 0 = "own" (current player's pieces, 1.0 at each cell).
/// Plane 1 = "opponent" (1.0 at each cell where the opponent has a piece).
/// Plane 2 = "turn" (all 1.0 — constant bias plane, as per C4D1 format).
///
/// The bit layout is the same column-major 7-bits-per-column encoding
/// used in `bitboard.rs`: cell (row r, col c) → bit (c * 7 + r).
fn board_to_input(board: Board) -> (Vec<usize>, Vec<f32>) {
    let shape = vec![1, 3, 6, 7];
    let mut data = Vec::with_capacity(1 * 3 * 6 * 7);

    for r in 0..6 {
        for col in 0..7 {
            let bit = 1u64 << (col * 7 + r);
            data.push(if board.own & bit != 0 { 1.0 } else { 0.0 });
        }
    }
    for r in 0..6 {
        for col in 0..7 {
            let bit = 1u64 << (col * 7 + r);
            data.push(if board.opp & bit != 0 { 1.0 } else { 0.0 });
        }
    }
    for _ in 0..42 {
        data.push(1.0);
    }

    (shape, data)
}

/// Build a (N, 3, 6, 7) f32 tensor from a batch of Boards.
fn boards_to_input(boards: &[Board]) -> (Vec<usize>, Vec<f32>) {
    let n = boards.len();
    let shape = vec![n, 3, 6, 7];
    let mut data = Vec::with_capacity(n * 3 * 6 * 7);
    for board in boards {
        for r in 0..6 {
            for col in 0..7 {
                let bit = 1u64 << (col * 7 + r);
                data.push(if board.own & bit != 0 { 1.0 } else { 0.0 });
            }
        }
        for r in 0..6 {
            for col in 0..7 {
                let bit = 1u64 << (col * 7 + r);
                data.push(if board.opp & bit != 0 { 1.0 } else { 0.0 });
            }
        }
        for _ in 0..42 {
            data.push(1.0);
        }
    }
    (shape, data)
}

// ---------------------------------------------------------------------------
// Backend-specific eval (Tract)
// ---------------------------------------------------------------------------

fn tract_eval(sess: &TractSession, board: Board) -> Eval {
    let (shape, data) = board_to_input(board);
    let input_t: Tensor = Tensor::from_shape(&shape, &data).expect("tract shape");
    let result = sess
        .run(tvec![input_t.into()])
        .expect("tract inference failed for single position");
    let policy_view = result[0]
        .to_array_view::<f32>()
        .expect("tract policy output extraction failed");
    let value_view = result[1]
        .to_array_view::<f32>()
        .expect("tract value output extraction failed");
    let value = value_view
        .as_slice()
        .and_then(|s| s.first().copied())
        .expect("tract value output has no elements");
    let max = policy_view
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_row: Vec<f32> = policy_view.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp_row.iter().sum();
    let mut policy = [0.0f32; 7];
    for c in 0..7 {
        policy[c] = exp_row[c] / sum;
    }
    Eval { policy, value }
}

fn tract_batch_eval(sess: &TractSession, boards: &[Board]) -> Vec<Eval> {
    let n = boards.len();
    let (shape, data) = boards_to_input(boards);
    let input_t: Tensor = Tensor::from_shape(&shape, &data).expect("tract shape");
    let result = sess
        .run(tvec![input_t.into()])
        .expect("tract inference failed for batch");
    let policy_view = result[0]
        .to_array_view::<f32>()
        .expect("tract policy output extraction failed");
    let value_view = result[1]
        .to_array_view::<f32>()
        .expect("tract value output extraction failed");
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        use tract_core::ndarray::Axis;
        let row = policy_view.index_axis(Axis(0), i);
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_row: Vec<f32> = row.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exp_row.iter().sum();
        let mut policy = [0.0f32; 7];
        for c in 0..7 {
            policy[c] = exp_row[c] / sum;
        }
        out.push(Eval {
            policy,
            value: value_view[i],
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Fallback: null network
// ---------------------------------------------------------------------------

/// Fallback evaluation: uniform over legal moves, value = 0.
fn null_eval(board: Board) -> Eval {
    let mut policy = [0.0f32; 7];
    let occupied = board.own | board.opp;
    let mut n_legal = 0u32;
    for c in 0..7 {
        if (occupied & col_mask(c)) != col_mask(c) {
            n_legal += 1;
        }
    }
    if n_legal > 0 {
        let p = 1.0 / n_legal as f32;
        for c in 0..7 {
            if (occupied & col_mask(c)) != col_mask(c) {
                policy[c] = p;
            }
        }
    }
    Eval { policy, value: 0.0 }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_eval_is_uniform_over_legal_moves() {
        let board = Board::new();
        let eval = null_eval(board);
        let expected = 1.0 / 7.0;
        for c in 0..7 {
            assert!((eval.policy[c] - expected).abs() < 1e-6);
        }
        assert_eq!(eval.value, 0.0);
    }

    #[test]
    fn null_eval_masks_illegal_columns() {
        let mut board = Board::new();
        for _ in 0..6 {
            let _ = board.make_move(0);
        }
        let eval = null_eval(board);
        assert_eq!(eval.policy[0], 0.0);
        let expected = 1.0 / 6.0;
        for c in 1..7 {
            assert!((eval.policy[c] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn board_to_input_layout() {
        let own = 1u64 << 21;
        let opp = 1u64 << 37;
        let board = Board { own, opp };
        let (shape, data) = board_to_input(board);
        assert_eq!(shape, vec![1, 3, 6, 7]);
        // Plane 0 (own), row 0, col 3 → 1.0
        assert_eq!(data[0 * 6 * 7 + 0 * 7 + 3], 1.0);
        // Plane 1 (opp), row 2, col 5 → 1.0
        assert_eq!(data[1 * 6 * 7 + 2 * 7 + 5], 1.0);
        // Plane 2 (turn) all 1.0
        for i in 2 * 6 * 7..3 * 6 * 7 {
            assert_eq!(data[i], 1.0);
        }
    }

    #[test]
    fn device_auto_resolves_correctly() {
        let resolved = Device::Auto.resolve();
        if cfg!(feature = "cuda") {
            assert_eq!(resolved, Device::Gpu);
        } else {
            assert_eq!(resolved, Device::Cpu);
        }
    }

    #[test]
    fn device_from_str_parses_all_variants() {
        assert_eq!(Device::from_str("cpu").unwrap(), Device::Cpu);
        assert_eq!(Device::from_str("CPU").unwrap(), Device::Cpu);
        assert_eq!(Device::from_str("gpu").unwrap(), Device::Gpu);
        assert_eq!(Device::from_str("cuda").unwrap(), Device::Gpu);
        assert_eq!(Device::from_str("auto").unwrap(), Device::Auto);
        assert!(Device::from_str("tpu").is_err());
    }
}
