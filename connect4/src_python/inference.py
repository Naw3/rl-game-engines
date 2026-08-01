"""
inference.py — Standalone inference engine & AI agent for Connect 4.

Decouples neural network inference, bitboard-to-tensor plane transformations,
and action selection logic from UI rendering (gui.py).
"""

from __future__ import annotations

import math
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import CONFIG
from src_python.model import Connect4Net

# Connect 4 dimensions
ROWS, COLS = 6, 7
_BIT_TABLE = np.array(
    [[1 << (c * 7 + r) for c in range(COLS)] for r in range(ROWS)],
    dtype=np.uint64,
)
_DLL_HANDLES = []
_EPOCH_MODEL_RE = re.compile(
    r"^(?P<base>.+?)_epoch(?P<epoch>\d+)(?P<ema>_ema)?$",
    re.IGNORECASE,
)


def _prepare_cuda_dll_paths() -> None:
    """Expose pip-installed CUDA/TensorRT DLLs to ONNX Runtime on Windows."""
    if os.name != "nt":
        return

    candidates: list[Path] = []
    for root in map(Path, sys.path):
        nvidia_root = root / "nvidia"
        if nvidia_root.exists():
            candidates.extend(nvidia_root.glob("*/bin"))
        torch_lib = root / "torch" / "lib"
        if torch_lib.exists():
            candidates.append(torch_lib)
            
    candidates.extend([
        Path(r"C:\TensorRT10\TensorRT-10.16.1.11\bin"),
        Path(r"C:\TensorRT10\TensorRT-10.16.1.11\lib"),
        _ROOT / "src_rust" / "target" / "release",
        _ROOT / "src_rust" / "target" / "debug",
    ])

    for directory in candidates:
        if not directory.is_dir():
            continue

        # Alias CUDA 12 DLLs to CUDA 13 names if requested by ONNX Runtime provider
        for dll12 in directory.glob("*64_12.dll"):
            dll13 = directory / dll12.name.replace("64_12.dll", "64_13.dll")
            if not dll13.exists():
                try:
                    import shutil
                    shutil.copyfile(dll12, dll13)
                except Exception:
                    pass

        directory_str = str(directory)
        if directory_str not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = directory_str + os.pathsep + os.environ.get("PATH", "")
        try:
            _DLL_HANDLES.append(os.add_dll_directory(directory_str))
        except (AttributeError, OSError):
            pass


def _manifest_selected_model(manifest_path: Path) -> Path | None:
    """Read the optional MCTS selection manifest."""
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        selected = manifest.get("model_pt") or manifest.get("model")
        if not selected:
            return None
        selected_path = Path(selected)
        if not selected_path.is_absolute():
            selected_path = manifest_path.parent / selected_path
        return selected_path if selected_path.exists() else None
    except Exception as e:
        print(f"[inference] WARNING: Could not read {manifest_path}: {e}")
        return None


def _epoch_model_candidates(base_path: Path) -> list[tuple[int, bool, Path]]:
    """Return raw/EMA epoch checkpoints for the configured model stem."""
    candidates: list[tuple[int, bool, Path]] = []
    for candidate in base_path.parent.glob(f"{base_path.stem}_epoch*.pt"):
        match = _EPOCH_MODEL_RE.match(candidate.stem)
        if match and match.group("base") == base_path.stem:
            candidates.append((int(match.group("epoch")), bool(match.group("ema")), candidate))
    return candidates


def resolve_best_model(model_path: str | Path | None = None) -> Path:
    """Resolve an explicit path or the highest cumulative epoch checkpoint."""
    if model_path is not None and str(model_path).lower() not in {"", "auto"}:
        return Path(model_path)

    base_path = Path(CONFIG.paths.model_pt)
    manifest_path = Path(CONFIG.paths.model_dir) / "best_model.json"
    candidates = _epoch_model_candidates(base_path)
    if candidates:
        highest_epoch = max(epoch for epoch, _is_ema, _path in candidates)
        highest = [item for item in candidates if item[0] == highest_epoch]
        manifest_selected = _manifest_selected_model(manifest_path)

        selected = next(
            (item for item in highest if manifest_selected and item[2].resolve() == manifest_selected.resolve()),
            None,
        )
        if selected is None:
            # With no MCTS preference for this exact epoch, use the raw model.
            selected = next((item for item in highest if not item[1]), highest[0])

        print(
            f"[inference] Auto-selected highest checkpoint {selected[2].name} "
            f"(epoch {selected[0]})"
        )
        return selected[2]

    manifest_selected = _manifest_selected_model(manifest_path)
    if manifest_selected is not None:
        print(
            f"[inference] Auto-selected {manifest_selected.name} "
            f"from {manifest_path.name}"
        )
        return manifest_selected

    return base_path


def _ensure_onnx_checkpoint(pt_path: Path, onnx_path: Path) -> None:
    """Export an epoch checkpoint when its matching ONNX file is missing/stale."""
    if not pt_path.exists():
        return
    if onnx_path.exists() and onnx_path.stat().st_mtime_ns >= pt_path.stat().st_mtime_ns:
        return

    try:
        from src_python.utils.onnx_export import export_onnx

        model = Connect4Net.load(pt_path)
        export_onnx(model, onnx_path, infer_precision="fp32", verify=True)
        print(f"[inference] Exported ONNX checkpoint to {onnx_path}")
    except Exception as e:
        print(f"[inference] WARNING: Could not export {pt_path.name} to ONNX: {e}")


def board_to_planes(own: int, opp: int) -> np.ndarray:
    """Convert bitboard ints (own, opp) to a (3, 6, 7) float32 input plane tensor.

    Planes:
        0: own pieces
        1: opponent pieces
        2: turn indicator mask (all 1.0s)
    """
    planes = np.empty((3, ROWS, COLS), dtype=np.float32)
    planes[0] = (np.bitwise_and(_BIT_TABLE, np.uint64(own)) != np.uint64(0))
    planes[1] = (np.bitwise_and(_BIT_TABLE, np.uint64(opp)) != np.uint64(0))
    planes[2] = 1.0
    return planes


def get_legal_actions_mask(own: int, opp: int) -> np.ndarray:
    """Return a 7-element boolean array where True indicates playable column."""
    occ = own | opp
    mask = np.zeros(COLS, dtype=bool)
    for c in range(COLS):
        # Column is legal if top row (row 5, bit c*7 + 5) is empty
        if (occ & (1 << (c * 7 + 5))) == 0:
            mask[c] = True
    return mask


def _check_win_bitboard(bitboard: int) -> bool:
    """Return whether a bitboard contains four connected pieces."""
    for shift in (1, 7, 6, 8):
        aligned = bitboard & (bitboard >> shift)
        if aligned & (aligned >> (2 * shift)):
            return True
    return False


def format_duration(seconds: float) -> str:
    """Show sub-second timings in milliseconds instead of rounding to 0s."""
    seconds = max(0.0, float(seconds))
    if seconds < 1.0:
        return f"{max(1, round(seconds * 1000.0))}ms" if seconds > 0.0 else "0ms"
    return f"{seconds:.2f}s"


def decode_confidence(confidence: float | None) -> float | None:
    """Clamp the optional learned confidence output to [0, 1]."""
    if confidence is None or not np.isfinite(confidence):
        return None
    return float(np.clip(confidence, 0.0, 1.0))


def _resolve_inference_backend(
    backend: str,
    device: str | None,
) -> tuple[str, torch.device, bool, str | None, str]:
    """Resolve CLI/config names into an engine, device and provider policy."""
    requested = (backend or "auto").strip().lower().replace("_", "-").replace(" ", "-")
    configured = str(getattr(CONFIG.infer, "infer_backend", "auto")).strip().lower().replace("_", "-").replace(" ", "-")
    if requested == "auto" and configured != "auto":
        requested = configured

    aliases = {"torch": "pytorch", "onnx": "onnx"}
    requested = aliases.get(requested, requested)
    allowed = {"auto", "pytorch", "pytorch-cuda", "pytorch-cpu", "onnx", "onnx-cuda", "tensorrt"}
    if requested not in allowed:
        raise ValueError(
            f"Unsupported inference backend {backend!r}; choose auto, "
            "pytorch-cuda, pytorch-cpu, onnx-cuda or tensorrt."
        )

    forced_device = None
    require_gpu = False
    required_provider = None
    if requested == "pytorch-cuda":
        engine = "torch"
        forced_device = "cuda"
        require_gpu = True
    elif requested == "pytorch-cpu":
        engine = "torch"
        forced_device = "cpu"
    elif requested == "pytorch":
        engine = "torch"
    elif requested == "onnx-cuda":
        engine = "onnx"
        forced_device = "cuda"
        require_gpu = True
        required_provider = "CUDAExecutionProvider"
    elif requested == "tensorrt":
        engine = "tensorrt"
        forced_device = "cuda"
        require_gpu = True
        required_provider = "TensorrtExecutionProvider"
    else:
        engine = requested

    resolved_device = forced_device or device or getattr(CONFIG.infer, "device", "auto")
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_device_obj = torch.device(resolved_device)
    if require_gpu and resolved_device_obj.type != "cuda":
        raise RuntimeError(f"Inference backend {requested} requires a CUDA device.")
    if requested == "pytorch-cuda" and not torch.cuda.is_available():
        raise RuntimeError("Inference backend pytorch-cuda requested, but CUDA is unavailable.")
    return engine, resolved_device_obj, require_gpu, required_provider, requested


class Connect4Agent:
    """Standalone MCTS agent with ONNX/TensorRT/CUDA and PyTorch fallbacks."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
        backend: str = "auto",
        verbose: bool = True,
        early_stop: bool | None = None,
    ) -> None:
        (
            backend,
            self.device,
            require_gpu,
            required_provider,
            self.infer_backend,
        ) = _resolve_inference_backend(backend, device)

        self.verbose = verbose
        self.early_stop_enabled = (
            bool(early_stop)
            if early_stop is not None
            else bool(getattr(CONFIG.infer, "confidence_stop_enabled", True))
        )
        self.model: Connect4Net | None = None
        self.ort_session = None
        self._ort_input_name: str | None = None
        self._ort_policy_name: str | None = None
        self._ort_value_name: str | None = None
        self._ort_moves_left_name: str | None = None
        self._ort_confidence_name: str | None = None
        self.has_confidence_head = False
        self._ort_input_dtype = np.float32
        self.eval_cache: dict[tuple[int, int], tuple[np.ndarray, float, float | None]] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0

        path = resolve_best_model(model_path)
        explicit_path = model_path is not None and str(model_path).lower() not in {"", "auto"}
        explicit_onnx = explicit_path and path.suffix.lower() == ".onnx"
        pt_path = path if path.suffix.lower() == ".pt" else path.with_suffix(".pt")
        onnx_path = path if path.suffix.lower() == ".onnx" else path.with_suffix(".onnx")

        loaded_onnx = False
        if backend != "torch" and onnx_path.exists():
            if not explicit_onnx:
                _ensure_onnx_checkpoint(pt_path, onnx_path)
            loaded_onnx = self._try_load_onnx(
                onnx_path,
                backend=backend,
                allow_cpu=(not require_gpu and (backend != "auto" or self.device.type != "cuda" or explicit_onnx)),
                required_provider=required_provider,
            )
        elif backend != "torch" and not explicit_onnx:
            _ensure_onnx_checkpoint(pt_path, onnx_path)
            if onnx_path.exists():
                loaded_onnx = self._try_load_onnx(
                    onnx_path,
                    backend=backend,
                    allow_cpu=(not require_gpu and (backend != "auto" or self.device.type != "cuda")),
                    required_provider=required_provider,
                )

        if not loaded_onnx:
            if require_gpu:
                raise RuntimeError(
                    f"Could not load requested inference backend {self.infer_backend} "
                    f"from {onnx_path}."
                )
            if backend in {"onnx", "tensorrt"} and not pt_path.exists():
                raise RuntimeError(f"Could not load requested {backend} backend from {onnx_path}")
            self._load_torch(pt_path)

    def _try_load_onnx(
        self,
        path: Path,
        backend: str,
        allow_cpu: bool,
        required_provider: str | None = None,
    ) -> bool:
        try:
            _prepare_cuda_dll_paths()
            import onnxruntime as ort

            available = ort.get_available_providers()
            if required_provider is not None:
                preferred_providers = [required_provider]
            elif backend == "tensorrt":
                preferred_providers = ["TensorrtExecutionProvider"]
            else:
                # CUDA is the reliable default; TensorRT remains available via
                # --backend tensorrt or as a fallback when explicitly exposed.
                preferred_providers = ["CUDAExecutionProvider", "TensorrtExecutionProvider"]
            gpu_providers = [provider for provider in preferred_providers if provider in available]
            if required_provider is not None and required_provider not in available:
                raise RuntimeError(
                    f"{required_provider} is unavailable "
                    f"(available: {', '.join(available)})"
                )
            if self.device.type == "cuda" and backend == "auto" and not gpu_providers and not allow_cpu:
                print(
                    "[inference] ONNX GPU providers unavailable; "
                    "falling back to PyTorch CUDA"
                )
                return False

            providers = gpu_providers if backend in {"auto", "onnx", "tensorrt"} else []
            if not providers or allow_cpu:
                providers = providers + ["CPUExecutionProvider"]

            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            inputs = session.get_inputs()
            outputs = {output.name for output in session.get_outputs()}
            active_providers = session.get_providers()
            active_gpu = {
                "CUDAExecutionProvider",
                "TensorrtExecutionProvider",
            }.intersection(active_providers)
            if self.device.type == "cuda" and backend == "auto" and not active_gpu and not allow_cpu:
                print(
                    "[inference] ONNX Runtime could not activate a CUDA/TensorRT provider; "
                    "falling back to PyTorch CUDA"
                )
                return False
            if required_provider is not None and required_provider not in active_providers:
                raise RuntimeError(
                    f"{required_provider} could not be activated "
                    f"(active: {', '.join(active_providers)})"
                )
            self.ort_session = session
            self._ort_input_name = inputs[0].name
            self._ort_policy_name = "policy" if "policy" in outputs else session.get_outputs()[0].name
            self._ort_value_name = "value" if "value" in outputs else session.get_outputs()[1].name
            self._ort_moves_left_name = "moves_left" if "moves_left" in outputs else None
            self._ort_confidence_name = "confidence" if "confidence" in outputs else None
            self.has_confidence_head = self._ort_confidence_name is not None
            self._ort_input_dtype = np.float16 if "float16" in inputs[0].type else np.float32
            provider = active_providers[0]
            print(f"[inference] Loaded ONNX model from {path} via {provider}")
            return True
        except Exception as e:
            if backend in {"onnx", "tensorrt"}:
                print(f"[inference] WARNING: Could not load {backend} model from {path}: {e}")
            return False

    def _load_torch(self, path: Path) -> None:
        self.model = Connect4Net().to(self.device)
        self.model.eval()
        if path.exists():
            try:
                state_dict = torch.load(path, map_location=self.device, weights_only=True)
                if isinstance(state_dict, dict) and isinstance(state_dict.get("model_state_dict"), dict):
                    state_dict = state_dict["model_state_dict"]
                missing, _unexpected = self.model.load_state_dict(state_dict, strict=False)
                self.model.has_confidence_head = not any(
                    name.startswith("confidence_") for name in missing
                )
                self.has_confidence_head = self.model.has_confidence_head
                print(
                    f"[inference] Loaded model weights from {path} "
                    f"on {self.device.type.upper()} via PyTorch"
                )
            except Exception as e:
                print(f"[inference] WARNING: Could not load model from {path} ({e})")
        else:
            print(
                f"[inference] WARNING: Model path {path} does not exist; "
                f"using un-trained weights on {self.device.type.upper()}"
            )

    def reset_cache(self) -> None:
        """Clear evaluation cache between games."""
        self.eval_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def evaluate_batch(
        self, positions: list[tuple[int, int]]
    ) -> list[tuple[np.ndarray, float, float | None]]:
        """Evaluate positions and return policy, value, and learned confidence."""
        if not positions:
            return []

        results: list[tuple[np.ndarray, float, float | None] | None] = [None] * len(positions)
        uncached_indices: list[int] = []
        uncached_planes: list[np.ndarray] = []

        for idx, (own, opp) in enumerate(positions):
            cache_key = (own, opp)
            if cache_key in self.eval_cache:
                self.cache_hits += 1
                p, v, b = self.eval_cache[cache_key]
                results[idx] = (p.copy(), v, b)
            else:
                self.cache_misses += 1
                uncached_indices.append(idx)
                uncached_planes.append(board_to_planes(own, opp))

        if uncached_planes:
            batch_np = np.stack(uncached_planes).astype(self._ort_input_dtype, copy=False)
            if self.ort_session is not None:
                output_names = [self._ort_policy_name, self._ort_value_name]
                if self._ort_moves_left_name is not None:
                    output_names.append(self._ort_moves_left_name)
                if self._ort_confidence_name is not None:
                    output_names.append(self._ort_confidence_name)
                ort_outputs = self.ort_session.run(
                    output_names,
                    {self._ort_input_name: batch_np},
                )
                log_ps, values = ort_outputs[:2]
                policies = np.exp(np.asarray(log_ps))
                values = np.asarray(values).reshape(-1)
                confidences = (
                    np.asarray(ort_outputs[-1]).reshape(-1)
                    if self._ort_confidence_name is not None
                    else None
                )
            else:
                batch_tensor = torch.from_numpy(batch_np).to(self.device)
                with torch.inference_mode():
                    log_ps, vs, _moves_left, confidence = self.model(batch_tensor)
                policies = torch.exp(log_ps).cpu().numpy()
                values = vs.squeeze(-1).cpu().numpy()
                confidences = (
                    confidence.cpu().numpy()
                    if self.has_confidence_head
                    else None
                )

            for b_i, orig_i in enumerate(uncached_indices):
                p = policies[b_i]
                v = float(values[b_i]) if values.ndim > 0 else float(values.item())
                confidence = (
                    decode_confidence(float(confidences[b_i]))
                    if confidences is not None
                    else None
                )
                own_pos, opp_pos = positions[orig_i]
                legal_mask = get_legal_actions_mask(own_pos, opp_pos)
                p[~legal_mask] = 0.0
                sum_p = p.sum()
                if sum_p > 0:
                    p /= sum_p
                else:
                    p[legal_mask] = 1.0 / max(1, legal_mask.sum())

                self.eval_cache[(own_pos, opp_pos)] = (p, v, confidence)
                results[orig_i] = (p, v, confidence)

        return [r for r in results if r is not None]

    def evaluate_position(self, own: int, opp: int) -> tuple[np.ndarray, float, float | None]:
        """Evaluate a single position via evaluate_batch."""
        return self.evaluate_batch([(own, opp)])[0]

    def mcts_search(
        self,
        own: int,
        opp: int,
        sims: int | None = None,
        max_think_time: float | None = None,
        batch_size: int | None = None,
        c_puct: float | None = None,
        early_stop: bool | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Run batched MCTS search using fixed InferConfig parameters from config.py."""
        if sims is None:
            sims = CONFIG.infer.sims
        if max_think_time is None:
            max_think_time = CONFIG.infer.max_think_time
        if batch_size is None:
            batch_size = CONFIG.infer.batch_size
        if c_puct is None:
            c_puct = CONFIG.infer.c_puct

        legal_mask = get_legal_actions_mask(own, opp)
        if not legal_mask.any():
            return np.ones(COLS, dtype=np.float32) / COLS, 0, 0.0

        hits_start = self.cache_hits
        misses_start = self.cache_misses

        # Explicit simulation counts are deterministic. With sims=0, the
        # confidence rule may stop early, otherwise the wall-clock deadline
        # is the fallback.
        sim_limit = float("inf") if sims == 0 else sims

        # Thread-safe node representation
        import threading
        class _MCTSNode:
            def __init__(self, own_b: int, opp_b: int, prior_p: np.ndarray):
                self.own = own_b
                self.opp = opp_b
                self.N = np.zeros(COLS, dtype=np.uint32)
                self.W = np.zeros(COLS, dtype=np.float32)
                self.P = prior_p.copy()
                self.children: dict[int, _MCTSNode] = {}
                self.lock = threading.Lock()

        # Include root evaluation in the reported move time and in the time
        # budget. The previous code started the clock after this evaluation,
        # which made the displayed duration less representative of the move.
        move_start = time.perf_counter()
        deadline = move_start + max(0.0, float(max_think_time))

        # Evaluate root
        root_policy, _, root_confidence = self.evaluate_position(own, opp)
        root = _MCTSNode(own, opp, root_policy)
        root_eval_elapsed = time.perf_counter() - move_start
        sim_count = 0

        confidence_stop_enabled = (
            sims == 0
            and (self.early_stop_enabled if early_stop is None else bool(early_stop))
            and root_confidence is not None
        )
        confidence_threshold = float(
            np.clip(getattr(CONFIG.infer, "confidence_threshold", 0.99), 0.0, 1.0)
        )
        confidence_min_sims = max(
            1, int(getattr(CONFIG.infer, "confidence_min_sims", 32))
        )
        confidence_stop = False
        visit_confidence = 0.0

        # Fixed GPU batching loop
        while time.perf_counter() < deadline and sim_count < sim_limit:
            batch_leaves: list[tuple[_MCTSNode, list[tuple[_MCTSNode, int]], int, int]] = []
            positions_to_eval: list[tuple[int, int]] = []

            for _ in range(batch_size):
                if sim_count >= sim_limit or time.perf_counter() >= deadline:
                    break
                sim_count += 1

                node = root
                search_path: list[tuple[_MCTSNode, int]] = []
                current_own, current_opp = own, opp
                term_val = None

                # Selection
                while True:
                    total_n = node.N.sum()
                    sqrt_total = math.sqrt(float(total_n)) if total_n > 0 else 1.0

                    best_score = -1e9
                    best_a = -1

                    for a in range(COLS):
                        if node.P[a] <= 0.0:
                            continue
                        q = node.W[a] / node.N[a] if node.N[a] > 0 else 0.0
                        u = c_puct * node.P[a] * sqrt_total / (1.0 + node.N[a])
                        score = q + u
                        if score > best_score:
                            best_score = score
                            best_a = a

                    if best_a == -1:
                        break

                    search_path.append((node, best_a))

                    # Drop piece
                    occ = current_own | current_opp
                    col_shift = best_a * 7
                    row_bit = None
                    for r in range(ROWS):
                        bit = 1 << (col_shift + r)
                        if (occ & bit) == 0:
                            row_bit = bit
                            break

                    if row_bit is None:
                        break

                    new_own = current_own | row_bit
                    if _check_win_bitboard(new_own):
                        term_val = 1.0
                        break
                    if (new_own | current_opp).bit_count() == 42:
                        term_val = 0.0
                        break

                    current_own, current_opp = current_opp, new_own

                    if best_a not in node.children:
                        positions_to_eval.append((current_own, current_opp))
                        batch_leaves.append((node, search_path, best_a, len(positions_to_eval) - 1))
                        term_val = None
                        break
                    else:
                        node = node.children[best_a]

                if term_val is not None and not search_path:
                    pass
                elif term_val is not None:
                    # Backprop terminal state immediately
                    v = term_val
                    for path_node, path_a in reversed(search_path):
                        path_node.N[path_a] += 1
                        path_node.W[path_a] += v
                        v = -v

            # Evaluate GPU batch of uncached leaves
            if positions_to_eval:
                eval_results = self.evaluate_batch(positions_to_eval)
                for parent_node, path, action, pos_idx in batch_leaves:
                    if pos_idx < len(eval_results):
                        child_policy, child_val, _child_confidence = eval_results[pos_idx]
                        c_own, c_opp = positions_to_eval[pos_idx]
                        child_node = _MCTSNode(c_own, c_opp, child_policy)
                        parent_node.children[action] = child_node

                        v = child_val
                        for path_node, path_a in reversed(path):
                            path_node.N[path_a] += 1
                            path_node.W[path_a] += v
                            v = -v

            # Confidence stop check: evaluate search convergence (visit distribution)
            # combined with model confidence head output (if available).
            if confidence_stop_enabled and sim_count >= confidence_min_sims:
                visits = root.N.astype(np.float32)
                visit_total = float(visits.sum())
                if visit_total > 0.0:
                    visit_confidence = float(visits.max() / visit_total)
                else:
                    visit_confidence = 0.0

                if root_confidence is not None:
                    confidence_index = float(max(visit_confidence, (root_confidence + visit_confidence) / 2.0))
                else:
                    confidence_index = visit_confidence

                if confidence_index >= confidence_threshold or visit_confidence >= confidence_threshold:
                    confidence_stop = True
                    break

        elapsed = time.perf_counter() - move_start
        search_elapsed = max(0.0, elapsed - root_eval_elapsed)
        hits_this_move = self.cache_hits - hits_start
        misses_this_move = self.cache_misses - misses_start
        total_evals = hits_this_move + misses_this_move
        hit_rate = (hits_this_move / max(1, total_evals)) * 100.0
        sps = sim_count / max(0.001, elapsed)
        visits = root.N.astype(np.float32)
        visit_total = float(visits.sum())
        if visit_total > 0.0:
            visit_confidence = float(visits.max() / visit_total)
        if root_confidence is not None:
            confidence_index = float(max(visit_confidence, (root_confidence + visit_confidence) / 2.0))
        else:
            confidence_index = visit_confidence

        if self.verbose:
            stop_note = (
                f" | Confidence stop triggered (index={confidence_index:.1%}, visits={visit_confidence:.1%})"
                if confidence_stop
                else ""
            )
            confidence_note = (
                f" | confidence index={confidence_index:.1%}, model={root_confidence:.1%}, visits={visit_confidence:.1%}, threshold={confidence_threshold:.1%}"
                if root_confidence is not None
                else f" | confidence index={confidence_index:.1%}, visits={visit_confidence:.1%}, threshold={confidence_threshold:.1%}"
            )
            print(
                f"[inference] Move completed in {format_duration(elapsed)} "
                f"(root={format_duration(root_eval_elapsed)}, search={format_duration(search_elapsed)}) | "
                f"{sim_count:,} sims ({sps:.0f} sims/s) | "
                f"Cache hits: {hits_this_move:,}/{total_evals:,} ({hit_rate:.1f}%) | RAM Cache: {len(self.eval_cache):,} entries"
                f"{confidence_note}"
                f"{stop_note}"
            )

        sum_v = visits.sum()
        if sum_v > 0:
            return visits / sum_v, sim_count, elapsed
        return root_policy, sim_count, elapsed

    def select_action(
        self,
        own: int,
        opp: int,
        sims: int | None = None,
        max_think_time: float | None = None,
        temperature: float | None = None,
        early_stop: bool | None = None,
    ) -> tuple[int, int, float]:
        """Select an action using time-budgeted parallel MCTS search with InferConfig defaults."""
        if sims is None:
            sims = CONFIG.infer.sims
        if max_think_time is None:
            max_think_time = CONFIG.infer.max_think_time
        if temperature is None:
            temperature = CONFIG.infer.temperature

        legal_mask = get_legal_actions_mask(own, opp)
        if not legal_mask.any():
            return 0, 0, 0.0

        if max_think_time > 0.0 or sims > 0:
            probs, sims_done, elapsed = self.mcts_search(
                own,
                opp,
                sims=sims,
                max_think_time=max_think_time,
                early_stop=early_stop,
            )
        else:
            probs, _value, _confidence = self.evaluate_position(own, opp)
            sims_done, elapsed = 1, 0.0

        probs[~legal_mask] = 0.0
        sum_p = probs.sum()
        if sum_p > 0:
            probs /= sum_p
        else:
            probs[legal_mask] = 1.0 / max(1, legal_mask.sum())

        if temperature == 0.0:
            return int(np.argmax(probs)), sims_done, elapsed

        logits = np.log(probs + 1e-12) / temperature
        exp_logits = np.exp(logits - np.max(logits))
        p = exp_logits / exp_logits.sum()
        return int(np.random.choice(COLS, p=p)), sims_done, elapsed
