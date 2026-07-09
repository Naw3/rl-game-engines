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
//     `--features cuda` at build time, plus `onnnxruntime-gpu` and the
//     CUDA toolkit at runtime. ~5x speedup over CPU on large workloads.
//
//   * `--device auto` → tries GPU first, falls back to CPU if CUDA is
//     unavailable or fails to initialise. (Auto-resolves to Gpu iff the
//     `cuda` feature was compiled in.)
//
// Performance note: `tract-onnx` is typically 1.5–3x faster than `ort`
// on CPU for small models, because there's no FFI marshalling per call.
// For our 227K-param model, CPU is the bottleneck either way, so the
// tract path is the default for `--device auto` when CUDA isn't set up.
//
// What "null" means
// ----------------
// `Network::null()` returns a Network with no backend. It produces
// uniform priors + value=0. The orchestrator's `init.py` ensures a real
// model exists before the very first self-play, so the null path is
// only hit if the `.onnx` was deleted manually.
// =============================================================================

use crate::bitboard::{col_mask, Board};
use std::path::Path;
use std::sync::{Arc, Mutex};

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

/// The inference network. Holds an optional `Backend` (Tract or Ort)
/// shared via `Arc` across all rayon worker threads.
pub struct Network {
    backend: Option<Arc<Backend>>,
    device: Device,
}

/// Inference backend enum — dispatched at `evaluate` time. Both variants
/// hold a thread-safe session handle.
enum Backend {
    /// Pure-Rust CPU inference via tract-onnx. Fastest on small models.
    Tract(Arc<TractSession>),
    /// FFI to onnxruntime, supports CUDA via the `cuda` feature.
    /// Wrapped in Mutex because ort 2.0 `Session::run` requires `&mut self`,
    /// and we share the session across rayon worker threads via Arc.
    Ort(Arc<Mutex<Session>>),
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
                let sess = load_ort(path.as_ref(), Device::Gpu)?;
                Backend::Ort(Arc::new(Mutex::new(sess)))
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
            Some(Backend::Ort(sess)) => ort_eval(sess, board),
            None => null_eval(board),
        }
    }

    /// Evaluate a batch of positions in a single forward pass.
    #[allow(dead_code)] // planned for v1.1 batching pipeline
    pub fn evaluate_batch(&self, boards: &[Board]) -> Vec<Eval> {
        if boards.is_empty() {
            return Vec::new();
        }
        match self.backend.as_deref() {
            Some(Backend::Tract(sess)) => tract_batch_eval(sess, boards),
            Some(Backend::Ort(sess)) => ort_batch_eval(sess, boards),
            None => boards.iter().map(|&b| null_eval(b)).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Backend loaders
// ---------------------------------------------------------------------------

/// Load the model with tract-onnx. CPU inference, fastest on this size.
///
/// Note: the Python exporter sets `dynamic_axes={"input": {0: "batch"}}`, so
/// the ONNX graph accepts any batch size. We therefore do NOT call
/// `with_input_fact` here — that would lock the batch dim to 1 and break
/// the batched inference path in `MCTS::run_with_batch`. Tract's shape
/// inference reads the dynamic dim from the ONNX graph and lets us feed
/// arbitrary batch sizes at runtime.
fn load_tract(path: &Path) -> Result<TractSession, Box<dyn std::error::Error>> {
    let session = tract_onnx::onnx()
        .model_for_path(path)?
        .into_optimized()?
        .into_runnable()?;
    Ok(session)
}

/// Load the model with ort. CUDA if `device == Gpu` and `cuda` feature
/// compiled in; CPU otherwise. The CUDA execution provider is added
/// FIRST so it has priority; CPU is the fallback.
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
// Tensor construction (shared by both backends)
// ---------------------------------------------------------------------------

/// Build a (1, 3, 6, 7) f32 tensor from a single Board.
///
/// Plane 0 = "own" (current player's pieces, 1.0 at each cell).
/// Plane 1 = "opponent" (1.0 at each cell where the opponent has a piece).
/// Plane 2 = "turn" (all 1.0 — the third plane is a constant bias, as
/// the model's contract requires 3 input planes per the C4D1 format).
///
/// The bit layout is the same column-major 7-bits-per-column encoding
/// used in `bitboard.rs`: cell (row r, col c) → bit (c * 7 + r).
///
/// Returns `(shape, data)` so both backends can build their own tensor type
/// from a plain `Vec<f32>` — this avoids ndarray version conflicts between
/// tract (0.16) and ort (0.17) that show up at the type-identity level.
fn board_to_input(board: Board) -> (Vec<usize>, Vec<f32>) {
    let shape = vec![1, 3, 6, 7];
    let mut data = Vec::with_capacity(1 * 3 * 6 * 7);
    for c in 0..3 {
        for r in 0..6 {
            for col in 0..7 {
                let bit = 1u64 << (col * 7 + r);
                let v = match c {
                    0 => {
                        if board.own & bit != 0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    1 => {
                        if board.opp & bit != 0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    _ => 1.0, // c == 2: turn plane, constant 1
                };
                data.push(v);
            }
        }
    }
    (shape, data)
}

/// Build a (N, 3, 6, 7) f32 tensor from a batch of Boards.
#[allow(dead_code)] // planned for v1.1 batching pipeline
fn boards_to_input(boards: &[Board]) -> (Vec<usize>, Vec<f32>) {
    let n = boards.len();
    let shape = vec![n, 3, 6, 7];
    let mut data = Vec::with_capacity(n * 3 * 6 * 7);
    for b in 0..n {
        let board = boards[b];
        for c in 0..3 {
            for r in 0..6 {
                for col in 0..7 {
                    let bit = 1u64 << (col * 7 + r);
                    let v = match c {
                        0 => {
                            if board.own & bit != 0 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        1 => {
                            if board.opp & bit != 0 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        _ => 1.0,
                    };
                    data.push(v);
                }
            }
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

#[allow(dead_code)] // planned for v1.1 batching pipeline
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
// Backend-specific eval (ort)
// ---------------------------------------------------------------------------

fn ort_eval(sess: &Mutex<Session>, board: Board) -> Eval {
    let (shape, data) = board_to_input(board);
    let value = Value::from_array((shape, data)).expect("failed to build ort input Value");
    // Extract everything we need from the session inside the lock guard.
    // `outputs` holds references into the session, so we materialise the
    // policy + value as owned data before releasing the lock.
    let (policy, value_scalar): (Vec<f32>, f32) = {
        let mut guard = sess.lock().expect("ort Mutex poisoned");
        let outputs = guard
            .run(ort::inputs!["input" => value])
            .expect("ort inference failed for single position");
        let policy_view = outputs[0]
            .try_extract_array::<f32>()
            .expect("ort policy output extraction failed");
        let value_view = outputs[1]
            .try_extract_array::<f32>()
            .expect("ort value output extraction failed");
        let policy: Vec<f32> = policy_view.iter().copied().collect();
        let v = *value_view
            .as_slice()
            .and_then(|s| s.first())
            .expect("ort value output has no elements");
        (policy, v)
    };
    let max = policy.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_row: Vec<f32> = policy.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp_row.iter().sum();
    let mut policy_arr = [0.0f32; 7];
    for c in 0..7 {
        policy_arr[c] = exp_row[c] / sum;
    }
    Eval {
        policy: policy_arr,
        value: value_scalar,
    }
}

#[allow(dead_code)] // planned for v1.1 batching pipeline
fn ort_batch_eval(sess: &Mutex<Session>, boards: &[Board]) -> Vec<Eval> {
    let n = boards.len();
    let (shape, data) = boards_to_input(boards);
    let value = Value::from_array((shape, data)).expect("failed to build batched ort input Value");
    let (policy_data, value_data) = {
        let mut guard = sess.lock().expect("ort Mutex poisoned");
        let outputs = guard
            .run(ort::inputs!["input" => value])
            .expect("ort inference failed for batch");
        let policy_view = outputs[0]
            .try_extract_array::<f32>()
            .expect("ort policy output extraction failed");
        let value_view = outputs[1]
            .try_extract_array::<f32>()
            .expect("ort value output extraction failed");
        let p = policy_view.iter().copied().collect::<Vec<f32>>();
        let v = value_view.iter().copied().collect::<Vec<f32>>();
        (p, v)
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let row = &policy_data[i * 7..(i + 1) * 7];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_row: Vec<f32> = row.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exp_row.iter().sum();
        let mut policy = [0.0f32; 7];
        for c in 0..7 {
            policy[c] = exp_row[c] / sum;
        }
        out.push(Eval {
            policy,
            value: value_data[i],
        });
    }
    out
}

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
