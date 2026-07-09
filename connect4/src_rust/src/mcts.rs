// =============================================================================
// mcts.rs — Monte Carlo Tree Search guided by the neural network.
//
// This is the AlphaZero MCTS: the network provides both the prior over
// actions (used in PUCT) and the leaf value (no random rollouts).
//
// PUCT score for a child action `a` from a state `s`:
//
//   UCB(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a))
//
// where:
//   Q(s, a) = W(s, a) / N(s, a)
//             is the empirical mean of the leaf values seen from the
//             perspective of the player to move at `s` (after playing
//             `a`). W and N are accumulated during backup.
//   P(s, a) is the prior probability of action `a` from the policy
//             head of the network. The very first time we visit `s`,
//             we call `network.evaluate(s)` to get (P, V) and store P
//             on the children as priors.
//
// Batched inference (v1.1)
// -----------------------
// The default mode calls the network once per leaf (batch size 1). With
// `batch_size > 1`, the MCTS queues up to `batch_size` leaves per round
// before flushing them through `network.evaluate_batch` in one call. This
// is critical on GPU: a 50 µs compute call is dwarfed by 500 µs of
// per-call FFI overhead, so batching 32 leaves at once is ~10× faster.
//
// Virtual loss is used to keep the K selections within a round diverse.
// Each path taken during selection gets a virtual visit-count increment
// before the next selection; other selections see those nodes as just-
// visited and prefer different branches. When the real eval returns,
// the virtual increment is undone and the real value is backed up.
//
// Leaf evaluation
// ---------------
// At a never-before-seen state, we call `network.evaluate` to get the
// value V(s). This is the L2-regularised prediction of the value head
// — the network's estimate of the position's outcome from the current
// player's perspective. We back V up the tree as the leaf value.
//
// (The earlier random-rollout variant has been removed entirely. There
// is no Monte Carlo playout. The network IS the evaluator. This is
// what makes the search a deep learned prior + a learned value, not a
// uniform-prior Monte Carlo estimate.)
//
// Value convention
// ----------------
// - Every node stores (own, opp) where `own` is the side to move AT
//   THAT NODE. The Board invariant applies recursively.
// - The value `v` returned from a search call is from the perspective
//   of the player to move at THAT node. So `v = +1` means "the player
//   to move here wins".
// - Backup flips the sign as the value bubbles up the tree: a child
//   that returns `+v` from its own-perspective is `−v` from the
//   parent's perspective (the parent just made the move, so the
//   parent and child "own" are different players).
//
// Dirichlet noise at the root
// ---------------------------
// To encourage exploration across independent games, the prior at the
// root is mixed with Dirichlet noise:
//
//   P'(s, a) = (1 - eps) * P_network(a) + eps * eta_a
//             eta ~ Dir(alpha)
//
// with `alpha = 0.3` and `eps = 0.25` (defaults). The noise is
// injected at the start of each MCTS run, so the search itself uses
// the noised priors. Disable with `--no-noise` on the Rust CLI.
//
// Move selection policy
// ---------------------
// After `simulations` simulations, the visit counts are converted to
// a move-sampling distribution:
//
//   pi(a | s) = N(s, a)^(1/tau) / sum_b N(s, b)^(1/tau)
//
// `tau = 1.0` (default for self-play) makes pi proportional to visit
// counts. `tau → 0` collapses to argmax — that's the mode used by the
// GUI. The temperature does NOT affect the value backup.
//
// Threading
// ---------
// The MCTS itself is single-threaded (each game owns one MCTS). The
// network is shared across games via `Arc<Network>` (see `main.rs`)
// so the 64 games spawned by rayon all share the same inference backend.
//
// On CPU (tract), `RunnableModel` is `Send + Sync` and serialises
// concurrent `run` calls internally — safe by construction.
//
// On GPU (ort), a single dispatcher thread owns the one ort `Session`
// and receives board states from all 64 MCTS threads via a
// `crossbeam_channel` queue. It batches incoming requests (up to 64
// boards at once) into a single GPU forward pass, then routes results
// back to their requesting threads. This eliminates per-call FFI
// overhead and fully saturates the GPU.
//
// Reference: Silver et al., 2017, §3.3.2 (PUCT), §3.4.1 (value
// backup, dirichlet noise), Algorithm 1 (the full self-play loop).
// =============================================================================

use crate::bitboard::Board;
use crate::network::{Eval, Network};
use rand::Rng;
use rand::distributions::Distribution;
use rand_distr::Dirichlet;
use std::sync::Arc;

/// Default PUCT exploration constant. AlphaZero uses 1.5; we keep the same.
pub const DEFAULT_C_PUCT: f32 = 1.5;
/// Default number of simulations per move during self-play data generation.
pub const DEFAULT_SIMS: usize = 800;
/// Default batch size for NN inference. 1 = sequential (legacy). 32 is a
/// reasonable default for both CPU tract (no slowdown) and GPU ort (huge
/// speedup). CLI flag `--batch-size` overrides.
pub const DEFAULT_BATCH_SIZE: usize = 32;
/// Dirichlet concentration parameter for root noise (legal-move uniform).
pub const DEFAULT_DIRICHLET_ALPHA: f32 = 0.3;
/// Mixing weight for Dirichlet noise at the root: P' = (1 - eps) P + eps * eta.
pub const DEFAULT_DIRICHLET_EPSILON: f32 = 0.25;

#[derive(Debug, Clone, Copy)]
pub struct MCTSConfig {
    pub simulations: usize,
    pub batch_size: usize,
    pub c_puct: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
    /// Temperature for the final move sampling. 1.0 = visit-count softmax.
    /// Near zero = argmax (greedy, used during evaluation / GUI play).
    pub temperature: f32,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        MCTSConfig {
            simulations: DEFAULT_SIMS,
            batch_size: DEFAULT_BATCH_SIZE,
            c_puct: DEFAULT_C_PUCT,
            dirichlet_alpha: DEFAULT_DIRICHLET_ALPHA,
            dirichlet_epsilon: DEFAULT_DIRICHLET_EPSILON,
            temperature: 1.0,
        }
    }
}

/// One node in the search tree.
struct Node {
    own: u64,
    opp: u64,
    /// Index of the child node corresponding to action `a`. u32::MAX = not yet
    /// created (we expand on first visit).
    children: [u32; 7],
    /// Visit count N(s, a) for each action.
    n: [u32; 7],
    /// Sum of leaf values W(s, a) for each action, from the perspective of
    /// the player to move at THIS node (not at the child — see backup).
    w: [f32; 7],
    /// Prior P(s, a) for each action. Set during expansion (from the
    /// network) or during root injection (network + Dirichlet).
    p: [f32; 7],
    /// True after expansion (children + priors created).
    is_expanded: bool,
    /// If the position is terminal, Some(value) where value is from the
    /// perspective of the player to move at this node. None otherwise.
    is_terminal: Option<f32>,
}

impl Node {
    fn new(own: u64, opp: u64) -> Self {
        Node {
            own,
            opp,
            children: [u32::MAX; 7],
            n: [0; 7],
            w: [0.0; 7],
            p: [0.0; 7],
            is_expanded: false,
            is_terminal: None,
        }
    }
}

/// One step on a path from root to leaf: the node index we were at, and the
/// action we took from there. The leaf's own entry has `action = usize::MAX`
/// (sentinel — there is no "action from the leaf"). The full path lets us
/// walk the tree backwards to apply / undo virtual loss or backup real value.
#[derive(Debug, Clone, Copy)]
struct PathEntry {
    node_idx: usize,
    action: usize,
}

/// One MCTS pass over a single position. Reuse the same `MCTS` to
/// avoid reallocating the tree between calls. The `Network` is shared
/// across all MCTS instances (typically one per rayon worker game).
pub struct MCTS {
    config: MCTSConfig,
    /// Flat arena of nodes. Index 0 is reserved as a sentinel; the
    /// actual root is pushed by `run`.
    tree: Vec<Node>,
    /// Shared, thread-safe reference to the inference network.
    network: Arc<Network>,
    /// Pre-allocated buffers for paths to avoid allocation in `select`
    scratch_paths: Vec<Vec<PathEntry>>,
}

impl MCTS {
    pub fn new(config: MCTSConfig, network: Arc<Network>) -> Self {
        let mut cfg = config;
        if cfg.batch_size < 1 {
            cfg.batch_size = 1;
        }
        MCTS {
            config: cfg,
            tree: Vec::with_capacity(8192),
            network,
            scratch_paths: (0..cfg.batch_size).map(|_| Vec::with_capacity(64)).collect(),
        }
    }

    /// Run `config.simulations` simulations from `root` and return:
    ///   - `policy`: length-7 vector, softmax over visit counts raised to
    ///                `1/temperature`, renormalized over legal moves.
    ///   - `q_values`: length-7 vector, mean leaf value per action (Q(s, a))
    ///                 from the perspective of the player to move at `root`.
    ///                 `q_values[c] = 0.0` if the child was never visited.
    ///
    /// Internally calls `run_with_batch` with the config's batch_size.
    pub fn run<R: Rng + ?Sized>(&mut self, root: Board, rng: &mut R) -> (Vec<f32>, Vec<f32>) {
        self.run_with_batch(root, self.config.batch_size, rng)
    }

    /// Run with an explicit batch_size override. Useful for tests and
    /// benchmarking different batching strategies.
    pub fn run_with_batch<R: Rng + ?Sized>(
        &mut self,
        root: Board,
        batch_size: usize,
        rng: &mut R,
    ) -> (Vec<f32>, Vec<f32>) {
        let batch_size = batch_size.max(1);

        // Clear the tree arena. We keep the allocation (capacity grows).
        self.tree.clear();
        let root_idx = self.tree.len();
        self.tree.push(Node::new(root.own, root.opp));

        let legal = root.legal_moves();
        let n_legal = (0..7).filter(|c| legal & (1 << c) != 0).count();

        // Get the network's policy for the root. Used as the prior that
        // Dirichlet noise is mixed into. (If the network is null, the
        // `evaluate` call returns a uniform policy + value 0.)
        let root_eval = self.network.evaluate(root);

        // Inject Dirichlet noise at the root (mix with the network prior).
        // Skip if dirichlet_epsilon is 0 (e.g. --no-noise flag) — in that
        // case the network's policy is used as-is.
        if n_legal > 0 && self.config.dirichlet_epsilon > 0.0 {
            self.add_root_dirichlet(root_idx, legal, n_legal, &root_eval.policy, rng);
        } else {
            self.set_root_priors(root_idx, legal, n_legal, &root_eval.policy);
        }

        // Batched simulation loop. We process `batch_size` selections per
        // round, then flush them through a single network call.
        let total_sims = self.config.simulations;
        let mut sims_done = 0usize;
        while sims_done < total_sims {
            let this_batch = batch_size.min(total_sims - sims_done);

            // Phase 1: select `this_batch` leaves with virtual loss.
            // A pending entry is either a real (non-terminal) leaf to eval,
            // or None if the selection hit a terminal (no NN call needed).
            let mut paths: Vec<Vec<PathEntry>> = Vec::with_capacity(this_batch);
            let mut leaf_boards: Vec<Board> = Vec::with_capacity(this_batch);
            let mut terminal_values: Vec<Option<f32>> = Vec::with_capacity(this_batch);

            for _ in 0..this_batch {
                let mut path = self.scratch_paths.pop().unwrap_or_else(|| Vec::with_capacity(64));
                path.clear();
                let terminal_value = self.select(root_idx, &mut path);
                
                if let Some(v) = terminal_value {
                    // Terminal: back up immediately, don't queue.
                    self.backup(&path, v);
                    self.scratch_paths.push(path);
                    continue;
                }
                // Apply virtual loss along the path (skip the leaf's
                // sentinel action entry).
                for entry in &path {
                    if entry.action != usize::MAX {
                        self.tree[entry.node_idx].n[entry.action] += 1;
                    }
                }
                let leaf_idx = path.last().unwrap().node_idx;
                let leaf_node = &self.tree[leaf_idx];
                leaf_boards.push(Board { own: leaf_node.own, opp: leaf_node.opp });
                paths.push(path);
                terminal_values.push(None);
            }

            // Phase 2: batched network eval for non-terminal leaves.
            if !paths.is_empty() {
                let evals: Vec<Eval> = self.network.evaluate_batch(&leaf_boards);

                // Phase 3: undo virtual loss, expand leaf with real priors,
                // back up real value.
                for ((path, _), eval) in paths.into_iter().zip(terminal_values.iter()).zip(evals.iter()) {
                    // Undo virtual loss.
                    for entry in path.iter() {
                        if entry.action != usize::MAX {
                            self.tree[entry.node_idx].n[entry.action] -= 1;
                        }
                    }
                    // Expand the leaf (sets priors + marks expanded).
                    let leaf_idx = path.last().unwrap().node_idx;
                    self.expand_with_eval(leaf_idx, eval);
                    // Backup real value.
                    self.backup(&path, eval.value);
                    self.scratch_paths.push(path);
                }
            }

            sims_done += this_batch;
        }

        // Extract policy + q_values from root.
        self.extract_root_outputs(root_idx, legal, n_legal)
    }

    /// Walk from `node_idx` down to a leaf (unexpanded node or terminal).
    /// Mutates the given path (each step's node index + action taken) and returns:
    ///   - `Some(value)` if the leaf is terminal (no NN eval needed), or
    ///   - `None` if the leaf is a normal unexpanded node to evaluate.
    fn select(&mut self, node_idx: usize, path: &mut Vec<PathEntry>) -> Option<f32> {
        let mut current = node_idx;

        loop {
            // Terminal: return cached value. The action entry for the leaf
            // is a sentinel (usize::MAX) since there is no "action from a
            // terminal that we expand into".
            if let Some(v) = self.tree[current].is_terminal {
                path.push(PathEntry { node_idx: current, action: usize::MAX });
                return Some(v);
            }

            // Leaf (unexpanded): return for batched eval.
            if !self.tree[current].is_expanded {
                path.push(PathEntry { node_idx: current, action: usize::MAX });
                return None;
            }

            // Internal: pick best PUCT child.
            let action = self.select_puct(current);
            path.push(PathEntry { node_idx: current, action });

            // Apply move on a temporary board.
            let mut child_board = Board {
                own: self.tree[current].own,
                opp: self.tree[current].opp,
            };
            let result = child_board.make_move(action);

            // Get or create the child node.
            let child = self.tree[current].children[action];
            let child_idx = if child != u32::MAX {
                child as usize
            } else {
                let new_idx = self.tree.len();
                self.tree.push(Node::new(child_board.own, child_board.opp));
                self.tree[current].children[action] = new_idx as u32;
                new_idx
            };

            // If the move ended the game, mark terminal and return.
            match result {
                crate::bitboard::MoveResult::Win => {
                    // The mover just won. The new position is from the
                    // opponent's perspective (board was swapped by make_move),
                    // so the player to move at the child has lost.
                    self.tree[child_idx].is_terminal = Some(-1.0);
                    self.tree[child_idx].is_expanded = true;
                    path.push(PathEntry { node_idx: child_idx, action: usize::MAX });
                    return Some(-1.0);
                }
                crate::bitboard::MoveResult::Draw => {
                    self.tree[child_idx].is_terminal = Some(0.0);
                    self.tree[child_idx].is_expanded = true;
                    path.push(PathEntry { node_idx: child_idx, action: usize::MAX });
                    return Some(0.0);
                }
                crate::bitboard::MoveResult::Continue => {
                    current = child_idx;
                }
                crate::bitboard::MoveResult::Illegal => {
                    // Should never happen if priors are correct.
                    return Some(0.0);
                }
            }
        }
    }

    /// Expand a leaf with priors from a real network eval (no NN call here).
    /// Sets priors, marks expanded. Idempotent on already-expanded nodes.
    fn expand_with_eval(&mut self, leaf_idx: usize, eval: &Eval) {
        let node = &mut self.tree[leaf_idx];
        if node.is_expanded {
            return; // already expanded (shouldn't happen but safe)
        }
        let occupied = node.own | node.opp;
        for c in 0..7 {
            let legal = (occupied & (1u64 << (c * 7 + 5))) == 0;
            node.p[c] = if legal { eval.policy[c] } else { 0.0 };
        }
        node.is_expanded = true;
    }

    /// Backup a value up the path. Value is from the LEAF's perspective;
    /// each step up the tree flips the sign because the player to move
    /// alternates. Skips the sentinel entries (action == usize::MAX).
    fn backup(&mut self, path: &[PathEntry], mut value: f32) {
        // path[0] is the root (or earlier node), path[last] is the leaf.
        // We want to update (n, w) at every entry that has a real action.
        for entry in path.iter() {
            if entry.action == usize::MAX {
                continue;
            }
            let node = &mut self.tree[entry.node_idx];
            node.n[entry.action] += 1;
            node.w[entry.action] += value;
            value = -value; // flip for parent's perspective
        }
    }

    /// PUCT child selection. Returns the action with the highest UCB.
    fn select_puct(&self, node_idx: usize) -> usize {
        let node = &self.tree[node_idx];
        let total_n: u32 = node.n.iter().sum();
        let sqrt_total = (total_n as f32).sqrt();
        let mut best_a = 0;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..7 {
            if node.p[c] <= 0.0 {
                continue; // illegal move
            }
            let q = if node.n[c] > 0 {
                node.w[c] / node.n[c] as f32
            } else {
                0.0 // unvisited child, neutral Q
            };
            let u = self.config.c_puct * node.p[c] * sqrt_total / (1.0 + node.n[c] as f32);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best_a = c;
            }
        }
        if self.tree[node_idx].p[best_a] > 0.0 {
            best_a
        } else {
            (0..7)
                .find(|c| self.tree[node_idx].p[*c] > 0.0)
                .unwrap_or(0)
        }
    }

    /// Extract policy + q_values from the root node after all simulations.
    fn extract_root_outputs(
        &self,
        root_idx: usize,
        legal: u8,
        n_legal: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut policy = [0.0f32; 7];
        let mut q_values = [0.0f32; 7];
        for c in 0..7 {
            let n = self.tree[root_idx].n[c];
            if n > 0 {
                q_values[c] = self.tree[root_idx].w[c] / n as f32;
            }
        }

        if self.config.temperature < 1e-3 {
            // Greedy: one-hot on the most-visited legal action.
            let mut best_a = 0usize;
            let mut best_n = 0u32;
            for c in 0..7 {
                if legal & (1 << c) != 0 && self.tree[root_idx].n[c] > best_n {
                    best_n = self.tree[root_idx].n[c];
                    best_a = c;
                }
            }
            if best_n > 0 {
                policy[best_a] = 1.0;
            } else if n_legal > 0 {
                for c in 0..7 {
                    if legal & (1 << c) != 0 {
                        policy[c] = 1.0 / n_legal as f32;
                    }
                }
            }
        } else {
            // Visit-count softmax with temperature.
            let inv_t = 1.0 / self.config.temperature;
            let mut weighted = [0.0f32; 7];
            for c in 0..7 {
                let n = self.tree[root_idx].n[c];
                if n > 0 && legal & (1 << c) != 0 {
                    weighted[c] = (n as f32).powf(inv_t);
                }
            }
            let sum_w: f32 = weighted.iter().sum();
            if sum_w > 0.0 {
                for c in 0..7 {
                    if legal & (1 << c) != 0 {
                        policy[c] = weighted[c] / sum_w;
                    }
                }
            } else if n_legal > 0 {
                for c in 0..7 {
                    if legal & (1 << c) != 0 {
                        policy[c] = 1.0 / n_legal as f32;
                    }
                }
            }
        }

        // Renormalize (in case some small numerical drift occurred).
        let sum: f32 = policy.iter().sum();
        if sum > 0.0 {
            for c in 0..7 {
                policy[c] /= sum;
            }
        }

        (policy.to_vec(), q_values.to_vec())
    }

    /// Sample Dirichlet noise and mix it into the root's priors.
    /// The result is `P'(a) = (1 - eps) * P_network(a) + eps * eta_a`.
    fn add_root_dirichlet<R: Rng + ?Sized>(
        &mut self,
        root_idx: usize,
        legal: u8,
        n_legal: usize,
        network_policy: &[f32; 7],
        rng: &mut R,
    ) {
        let alpha = vec![self.config.dirichlet_alpha; n_legal];
        let dist = match Dirichlet::new(&alpha) {
            Ok(d) => d,
            Err(_) => {
                // alpha <= 0 (shouldn't happen with default 0.3) — fall
                // back to plain network policy.
                self.set_root_priors(root_idx, legal, n_legal, network_policy);
                return;
            }
        };
        let eta = dist.sample(rng);
        let eps = self.config.dirichlet_epsilon;

        let mut i = 0;
        for c in 0..7 {
            if legal & (1 << c) != 0 {
                let p_net = network_policy[c];
                self.tree[root_idx].p[c] = (1.0 - eps) * p_net + eps * eta[i];
                i += 1;
            } else {
                self.tree[root_idx].p[c] = 0.0;
            }
        }
        // Mark root as expanded-via-noise so the first search uses the
        // noised priors. The search will create child nodes on first
        // selection, just as for any other expanded node.
        self.tree[root_idx].is_expanded = true;
    }

    /// Set root priors to the network's policy (no Dirichlet mixing).
    /// Used when `--no-noise` is passed or when n_legal == 0.
    fn set_root_priors(
        &mut self,
        root_idx: usize,
        legal: u8,
        n_legal: usize,
        network_policy: &[f32; 7],
    ) {
        if n_legal == 0 {
            self.tree[root_idx].is_expanded = true;
            return;
        }
        for c in 0..7 {
            self.tree[root_idx].p[c] = if legal & (1 << c) != 0 {
                network_policy[c]
            } else {
                0.0
            };
        }
        self.tree[root_idx].is_expanded = true;
    }
}

// =============================================================================
// Unit tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn null_mcts(simulations: usize, batch_size: usize) -> MCTS {
        let net = Arc::new(Network::null());
        MCTS::new(
            MCTSConfig {
                simulations,
                batch_size,
                ..Default::default()
            },
            net,
        )
    }

    #[test]
    fn policy_sums_to_one() {
        let b = Board::new();
        let mut mcts = null_mcts(16, 1);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let (policy, _q) = mcts.run(b, &mut rng);
        let total: f32 = policy.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "policy must sum to 1, got {}",
            total
        );
    }

    #[test]
    fn null_network_first_move_is_roughly_uniform() {
        let b = Board::new();
        let mut mcts = null_mcts(700, 1);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let (policy, _q) = mcts.run(b, &mut rng);
        for c in 0..7 {
            assert!(
                policy[c] > 0.05,
                "col {} got policy {} - too low, expected roughly uniform",
                c,
                policy[c]
            );
        }
    }

    /// Batch sizes > 1 should produce the same search output as batch=1
    /// (modulo RNG differences within a batch — but with seeded RNG and
    /// identical config they should be very close).
    #[test]
    fn batched_matches_sequential() {
        let b = Board::new();
        let seed = 42u64;

        let mut mcts_seq = null_mcts(64, 1);
        let mut rng_seq = rand::rngs::StdRng::seed_from_u64(seed);
        let (p_seq, _) = mcts_seq.run_with_batch(b, 1, &mut rng_seq);

        let mut mcts_batch = null_mcts(64, 16);
        let mut rng_batch = rand::rngs::StdRng::seed_from_u64(seed);
        let (p_batch, _) = mcts_batch.run_with_batch(b, 16, &mut rng_batch);

        // With null network and the same seed, batched and sequential
        // should agree on the argmax (the MCTS picks the same columns).
        let argmax_seq = p_seq
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let argmax_batch = p_batch
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            argmax_seq, argmax_batch,
            "argmax columns differ between batch=1 and batch=16 ({} vs {})",
            argmax_seq, argmax_batch
        );
    }
}