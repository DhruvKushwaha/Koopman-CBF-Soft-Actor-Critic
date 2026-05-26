# Robust Koopman CBF-RL

A framework for **safe reinforcement learning** via Koopman operator theory and Control Barrier Functions (CBFs). A data-driven Koopman model of the environment dynamics is fitted offline using Extended Dynamic Mode Decomposition (EDMD). The model, together with an empirical robustness margin derived from prediction residuals, is used to construct a discrete-time CBF constraint. At every time step a lightweight QP safety filter projects the nominal RL action onto the safe set before it is applied to the environment. The filter is differentiable with respect to the actor parameters, enabling a soft CBF penalty term in the actor loss.

Two RL backbones are provided:

- **KCBF-SAC** — off-policy, sample-efficient, suited to continuous action spaces.
- **KCBF-PPO** — on-policy, stable training, suited to tasks requiring tight policy gradient control.

---

## Table of Contents

1. [Theory overview](#1-theory-overview)
2. [Codebase structure](#2-codebase-structure)
3. [Installation](#3-installation)
4. [Environments](#4-environments)
5. [Safety barriers](#5-safety-barriers)
6. [End-to-end usage](#6-end-to-end-usage)
7. [Configuration reference](#7-configuration-reference)
8. [Baselines](#8-baselines)
9. [Evaluation and plotting](#9-evaluation-and-plotting)
10. [Performance criteria per environment](#10-performance-criteria-per-environment)
11. [Experiment sweep pipeline](#11-experiment-sweep-pipeline)
12. [Experimental results summary](#12-experimental-results-summary)
13. [Known limitations](#13-known-limitations)
14. [Testing](#14-testing)

---

## 1. Theory Overview

### 1.1 Koopman operator and EDMD

The Koopman operator lifts a nonlinear system into a (potentially infinite-dimensional) linear space where dynamics evolve linearly. We approximate it with a finite-dimensional dictionary of observables.

Given raw states `y_t ∈ ℝ^n` and controls `u_t ∈ ℝ^m`, the observable (lifting) map is:

```
z_t = φ(y_t) = [y_t, ψ_1(y_t), ..., ψ_K(y_t)]  ∈ ℝ^(n + K)
```

where `ψ_i` are Radial Basis Functions (RBF) with centers sampled from the data. The lifted dynamics are approximated as linear:

```
z_{t+1} ≈ A z_t + B u_t
```

Matrices `A` and `B` are fit by Extended Dynamic Mode Decomposition (EDMD) — ridge-regularised least squares on a dataset of `(z_t, u_t, z_{t+1})` triples:

```
[A | B] = argmin_{A,B}  ||Z' - Z A^T - U B^T||_F^2 + λ ||[A|B]||_F^2
```

### 1.2 Discrete-time Control Barrier Function (KCBF)

A barrier function `h_K : ℝ^z_dim → ℝ` defines the safe set `S = {z : h_K(z) ≥ 0}`. We use linear barriers in the lifted space:

```
h_K(z) = c^T z + d
```

The discrete-time CBF condition requires that `h_K` does not decrease faster than a factor `(1 - η)` per step:

```
h_K(z_{t+1}) ≥ (1 - η) h_K(z_t)   with η ∈ (0, 1)
```

A **larger η** (e.g. 0.9) corresponds to a **looser** constraint that allows the barrier to decrease quickly toward zero. A **smaller η** (e.g. 0.3) enforces a **tighter** invariance requirement that keeps the system further inside the safe set but risks over-constraining the policy. For contact-rich locomotion environments where the Koopman model has large prediction error, η=0.9 is recommended; over-tightening (η ≤ 0.7) can cause policy collapse (§12).

Substituting the Koopman prediction:

```
c^T (A z_t + B u_t) + d ≥ (1 - η)(c^T z_t + d)
```

Rearranging into a linear constraint on `u_t`:

```
(B^T c)^T u_t  ≥  (1 - η) h_K(z_t) - c^T(A z_t) - d
```

### 1.3 Robust margin ρ

The Koopman model is an approximation; prediction residuals `r_t = z_{t+1} - (A z_t + B u_t)` are non-zero. We project residuals along the barrier direction to measure their impact:

```
δ_t = |c^T r_t|
```

The robust margin `ρ` is the empirical α-quantile (default α = 0.95) of `{δ_t}`:

```
ρ = Quantile_α ({δ_t})
```

The tightened CBF condition used in the QP filter adds `ρ` to the right-hand side, providing a probabilistic safety guarantee even under model error:

```
(B^T c)^T u  ≥  (1 - η) h_K(z) + ρ - c^T(A z) - d
```

**ρ as a pre-training diagnostic.** Before running RL training, inspect ρ from the Koopman residuals file. Empirical values across benchmarks:

| Environment | ρ (95th pct) | KCBF effectiveness |
|---|---|---|
| CartPole (stab/track) | ~9×10⁻⁴ | Zero violations achievable |
| Quadrotor 2D (stab/track) | ~1.3×10⁻³ | <0.5% violations with composite barrier |
| Safety Walker | 0.698 | Filter marginal; SAC+Lagrangian may be safer |
| Safety HalfCheetah | 1.782 | Filter largely inactive; contact dynamics dominate |

When ρ ≥ 0.5, the robust CBF constraint either becomes trivially satisfied (when `h_K < 0`) or infeasible under actuator limits, degrading the safety certificate. Consider richer Koopman dictionaries or multi-step CBF extensions (§13).

A **cluster-based margin** mode is also available (`margin_mode: cluster`), which fits k-means clusters in lifted space and uses per-cluster quantiles — tighter in well-visited regions, more conservative elsewhere.

### 1.4 QP safety filter

At every step the nominal action `u_nom` from the RL agent is projected onto the CBF constraint by solving:

```
min_{u, ξ}   ||u - u_nom||^2 + λ_slack ξ^2
  s.t.   (B^T c)^T u + ξ  ≥  b_cbf
         u_min ≤ u ≤ u_max
         ξ ≥ 0
```

where `b_cbf = (1-η) h_K(z) + ρ - c^T(Az) - d` and `ξ` is a slack variable that prevents infeasibility (penalised heavily by `λ_slack`). The QP is solved with OSQP via `qpsolvers`.

> **Slack rate monitoring:** When `ξ > 0` at a step, the filter no longer enforces a formal CBF certificate at that step. Monitor `slack_rate` in the CSV log — environments where `slack_rate > 0.05` do not reliably maintain the safety certificate.

### 1.5 Actor CBF penalty

To encourage the nominal policy to produce actions that require less correction, a soft differentiable penalty is added to the actor loss:

```
L_CBF = λ_h · E[ max(0,  b_cbf - (B^T c)^T u_actor)^2 ]
```

For **SAC** this uses `dist.rsample()` (reparameterised) rescaled to the action space. For **PPO** it uses a fresh `dist.rsample()` inside the minibatch update (not the stored roll-out action, which has no gradient).

---

## 2. Codebase Structure

```
robust_koopman_cbf_rl/
│
├── koopman/
│   ├── observables.py      # RBFObservables: lift y → z = [y, rbf(y)]
│   ├── dataset.py          # KoopmanDataset: store (y, u, y') triples
│   ├── fit_edmd.py         # fit_edmd(): ridge-regularised EDMD
│   ├── model.py            # KoopmanModel: lift, predict, save, load
│   ├── residuals.py        # compute_residuals(), compute_robust_margin()
│   └── validate.py         # prediction-error diagnostics
│
├── cbf/
│   ├── barrier_base.py         # SafetyConstraint ABC
│   ├── cartpole_barriers.py    # CartPolePositionBarrier, CartPoleAngleBarrier
│   ├── quadrotor_barriers.py   # Quadrotor2DAltitudeBarrier,
│   │                           # Quadrotor2DCompositeAltitudeBarrier (RD-2 fix),
│   │                           # Quadrotor2DPitchBarrier,
│   │                           # Quadrotor3DPositionBarrier
│   ├── velocity_barriers.py    # VelocityNormBarrier (safety-gymnasium)
│   ├── robust_margin.py        # RobustMargin: global or cluster-based ρ
│   ├── factory.py              # build_barrier() — barrier from config string
│   ├── qp_filter.py            # KCBFQPFilter: QP projection + CBF penalty terms
│   └── diagnostics.py          # DiagnosticsBuffer: per-episode h_value, cbf_gap, etc.
│
├── agents/
│   ├── sac.py              # KCBFSACAgent: GaussianActor + twin Q-critics
│   ├── ppo.py              # KCBFPPOAgent: ActorCritic (shared backbone)
│   ├── replay_buffer.py    # KCBFReplayBuffer (off-policy)
│   └── rollout_buffer.py   # KCBFRolloutBuffer + GAE (on-policy)
│
├── train/
│   ├── collect_koopman_data.py # Random rollouts for EDMD dataset
│   ├── train_koopman.py        # Stage 1: EDMD fit + residual margin
│   ├── train_sac_kcbf.py       # Stage 2a: KCBF-SAC training loop
│   ├── train_ppo_kcbf.py       # Stage 2b: KCBF-PPO training loop
│   └── evaluate.py             # evaluate(): N-episode rollout → metrics
│
├── envs/
│   ├── safe_control_gym_wrapper.py   # SCG → gymnasium API + augmented info
│   └── safety_gymnasium_wrapper.py   # Safety-Gymnasium velocity-constrained envs
│
├── baselines/
│   ├── lqr.py              # Linear Quadratic Regulator
│   ├── pid.py              # PID controller
│   ├── penalty_rl.py       # Reward-penalty baseline (r' = r - λ·cost)
│   ├── lagrangian_rl.py    # Lagrangian dual variable update
│   └── physical_cbf_qp.py  # CBF-QP with known linearised dynamics
│
├── plots/
│   ├── plot_returns.py         # Mean ± std return curves from CSV logs
│   ├── plot_violations.py      # Constraint violation rate curves
│   ├── plot_intervention_rate.py # QP filter intervention rate curves
│   ├── plot_training.py        # 4-panel training dashboard
│   └── plot_trajectory.py      # collect_trajectory() + per-state-dim plot
│
├── eval/
│   ├── run_eval.py         # CLI: load checkpoint → run eval → save eval_results.csv
│   └── compare_models.py   # CLI + function: bar-chart comparison across models
│
├── utils/
│   ├── config.py           # Dataclass configs (KoopmanCfg, SACCfg, PPOCfg, EnvCfg)
│   ├── logger.py           # CSVLogger, dump_json
│   ├── metrics.py          # compute_episode_metrics()
│   └── seeding.py          # set_seed()
│
├── configs/                # YAML config files (see §7)
└── tests/                  # pytest unit tests

experiments/
├── run_sweep.py            # Parallel baseline sweep (SAC/PPO/LQR/Penalty/Lagrangian)
├── run_kcbf_sweep.py       # Parallel KCBF-SAC sweep over all envs × seeds
├── compare_all.py          # Aggregate results + all paper figures
└── ablation_cbf_params.py  # η × λ_slack ablation grid
```

---

## 3. Installation

### Prerequisites

- Python 3.10+
- [safe-control-gym](https://github.com/utiasDSL/safe-control-gym) — for CartPole and Quadrotor environments
- [safety-gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) — for HalfCheetah/Walker velocity tasks

### Install

```bash
# Clone this repo
git clone <repo-url>
cd Koopman-CBF/V1

# Install the package (editable)
pip install -e .

# Install QP solver (required for the safety filter)
pip install qpsolvers[osqp]

# Install safe-control-gym (follow upstream instructions)
cd safe-control-gym && pip install -e . && cd ..

# Install safety-gymnasium (optional, for locomotion envs)
pip install safety-gymnasium
```

### Conda environment

```bash
conda create -n safe-control python=3.10
conda activate safe-control
pip install -e . qpsolvers[osqp] torch numpy pandas matplotlib scipy pyyaml pytest
```

---

## 4. Environments

### 4.1 safe-control-gym (SCG)

SCG provides physics-accurate classical control environments with task and constraint configuration. The wrapper `SafeControlGymWrapper` converts SCG's 4-tuple step API to the gymnasium 5-tuple API and augments the `info` dict with:

| Key | Description |
|---|---|
| `raw_state` | Full physical state from `env.state` |
| `reference_state` | Goal/reference state (`env.X_GOAL`) |
| `tracking_error` | `raw_state - reference_state` |
| `cost` | `1.0` if any constraint violated, else `0.0` |

#### CartPole — Stabilization

Config: `configs/env_cartpole_stab.yaml`

| Parameter | Value |
|---|---|
| State | `[x, ẋ, θ, θ̇]` — cart position, velocity, pole angle, angular velocity |
| Action | Force `F ∈ [-10, 10]` N |
| Task | Stabilize at `(x, θ) = (0, 0)` |
| Control freq | 50 Hz (one physics step = 20 ms) |
| Episode length | 5 s (250 steps) |
| Init randomization | `x ∈ [-1.5, 1.5]`, `θ ∈ [-0.1, 0.1]` rad |
| SCG constraints | `|x| ≤ 2.0` m, `|θ| ≤ 0.16` rad |
| CBF barrier | `CartPolePositionBarrier(x_max=2.2)` — two-sided; 20 cm margin inside episode boundary |
| Koopman config | `koopman.yaml` (n_rbf=64, collection_steps=50k) |

#### CartPole — Trajectory Tracking

Config: `configs/env_cartpole_track.yaml`

Same as stabilization but with `task: traj_tracking` and `obs_goal_horizon: 1` (observation doubles to `[state, reference]`). The reference is a 2-cycle circle trajectory in the `zx` plane.

#### Quadrotor 2D — Stabilization

Config: `configs/env_quadrotor2d_stab.yaml`

| Parameter | Value |
|---|---|
| State | `[x, ẋ, z, ż, θ, θ̇]` — horizontal pos/vel, altitude/vel, pitch/rate |
| Action | 2 motor thrusts `∈ [0.06, 0.29]` N each (or `[-1,1]` normalised) |
| Task | Stabilize at `(x, z) = (0, 1)` m |
| Control freq | 50 Hz; physics freq 1000 Hz (20 substeps) |
| Mass / inertia | M = 0.027 kg, Iyy = 1.4×10⁻⁵ kg·m² (Crazyflie 2.x) |
| CBF barrier | `Quadrotor2DCompositeAltitudeBarrier(z_min=0.2, z_max=1.8)` (see §5.2) |
| Koopman config | `koopman_quadrotor.yaml` (n_rbf=32, collection_steps=10k) |

> **Important:** `normalized_rl_action_space: True` is set so the actor's tanh output maps cleanly to the physical thrust range. Without normalization, the actor saturates at ≈ 0.29 N and cannot decrease thrust — training diverges.

> **Relative degree:** The altitude constraint has **relative degree 2** w.r.t. vertical thrust: the standard `Quadrotor2DAltitudeBarrier` gives `c^T B ≈ 0`, making the QP degenerate. Use `Quadrotor2DCompositeAltitudeBarrier` instead (see §5.2).

#### Quadrotor 2D — Tracking

Config: `configs/env_quadrotor2d_track.yaml` — figure-8 reference trajectory.

#### Quadrotor 3D — Stabilization / Tracking

Configs: `configs/env_quadrotor3d_stab.yaml`, `configs/env_quadrotor3d_track.yaml`

| Parameter | Value |
|---|---|
| State | 12D: `[x, ẋ, y, ẏ, z, ż, φ, φ̇, θ, θ̇, ψ, ψ̇]` |
| Action | 4 motor thrusts `∈ [0.029, 0.148]` N each |
| Mass / inertia | M = 0.027 kg, Ixx = Iyy = 1.4×10⁻⁵, Izz = 2.17×10⁻⁵ kg·m² |

### 4.2 Safety-Gymnasium

Velocity-constrained locomotion tasks from [safety-gymnasium](https://github.com/PKU-Alignment/safety-gymnasium). The wrapper reads `info["cost"]` natively — no additional constraint setup needed.

| Config file | Environment ID | Constraint | Koopman config |
|---|---|---|---|
| `env_safety_halfcheetah.yaml` | `SafetyHalfCheetahVelocity-v1` | forward velocity ≤ 2.0 m/s | `koopman_walker_tuned.yaml` (n_rbf=256) |
| `env_safety_walker.yaml` | `SafetyWalker2dVelocity-v1` | forward velocity ≤ 2.0 m/s | `koopman_walker_tuned.yaml` (n_rbf=256) |

The `velocity_limit` field in the YAML controls which CBF barrier is active (`VelocityNormBarrier`).

> **Note on locomotion ρ:** Even with n_rbf=256, the projected residual margin ρ ≈ 0.7–1.8 for these environments due to contact dynamics producing non-smooth velocity transitions. The KCBF filter is structurally limited here; see §13.

---

## 5. Safety Barriers

All barriers inherit from `SafetyConstraint` (`cbf/barrier_base.py`) and implement:

- `value(raw_state, info) → float` — barrier value `h(y)` at the current raw state
- `lifted_barrier_coeffs(z_dim, ...) → (c, d)` — linear coefficients so `h_K(z) = c^T z + d`

Use `build_barrier()` from `cbf/factory.py` to construct barriers from config strings.

### 5.1 CartPole barriers

| Class | Signature | Formula |
|---|---|---|
| `CartPolePositionBarrier` | `(x_max=2.2, side='right')` | `h = x_max - x` (right) or `h = x_max + x` (left) |
| `CartPoleAngleBarrier` | `(theta_max=0.14, side='right')` | `h = θ_max - θ` (right) or `h = θ_max + θ` (left) |

Both appear as linear maps on the first (identity) part of the lifted state `z`, with the RBF portion zeroed out. Two barriers are used simultaneously — one per side — so that `h_+(z) = x_max - x` and `h_-(z) = x_max + x`.

### 5.2 Quadrotor barriers

| Class | Signature | Notes |
|---|---|---|
| `Quadrotor2DAltitudeBarrier` | `(z_min, z_max, side='upper')` | Naïve altitude barrier; has **relative degree 2** — use composite version for tracking. |
| `Quadrotor2DCompositeAltitudeBarrier` | `(z_min, z_max=2.0, alpha=1.0, beta=0.5)` | `h = α(z − z_min) + β·ż` — converts RD-2 altitude to RD-1; **recommended for all Quadrotor tasks**. |
| `Quadrotor2DPitchBarrier` | `(theta_max, side='right')` | `h = θ_max - θ` |
| `Quadrotor3DPositionBarrier` | `(axis_index, lo, hi, side='upper')` | `h = hi - x[axis]` or `h = x[axis] - lo` |

**Why the composite barrier matters.** For `Quadrotor2DAltitudeBarrier`, the standard CBF-QP gives `c^T B ≈ 0` because altitude is RD-2 w.r.t. vertical thrust (thrust enters acceleration `z̈`, not velocity). The QP constraint becomes:

```
(B^T c)^T u  ≥  b_cbf     with B^T c ≈ 0
```

This is degenerate — any `u` trivially satisfies it, so the filter never corrects unsafe actions. In experiments, this caused the effective filter control authority `‖a_cbf‖` to collapse from `2.21×10⁻³` to `5.47×10⁻⁵` (40× degradation), resulting in 12.5% violation rate on Quadrotor tracking.

`Quadrotor2DCompositeAltitudeBarrier` resolves this by combining altitude and vertical velocity:

```
h(z) = α (z − z_min) + β ż
```

Since `ż` appears in vertical dynamics as `z̈ = (F_T cos θ)/m − g` (directly influenced by thrust), `c^T B ≠ 0` and the QP can produce meaningful corrections. With this barrier, Quadrotor 2D tracking violations dropped from 12.5% to 0.4%.

### 5.3 Velocity barrier (Safety-Gymnasium)

```
VelocityNormBarrier(v_max, vel_indices)
h(y) = v_max² - Σᵢ vᵢ²
```

This is **quadratic** in velocity, so it cannot be represented as `c^T z` using only identity features. It requires the observable dictionary to include `v_i²` as extra quadratic features (`extra_quadratic_indices`). The `lifted_barrier_coeffs()` call returns a third element `lift_extra = {"quadratic_indices": [...]}` that signals the Koopman model to augment the dictionary accordingly.

### 5.4 Conservative vs environment bounds

| Barrier | CBF bound | Environment bound | Margin |
|---|---|---|---|
| CartPole position | ±2.2 m | ±2.4 m (episode boundary) | 0.2 m |
| CartPole angle | ±0.14 rad | ±0.16 rad (SCG constraint) | 0.02 rad |
| Quadrotor altitude | [0.2, 1.8] m | [-0.05, 2.0] m (obs space) | 0.25 m / 0.2 m |

---

## 6. End-to-End Usage

Training consists of two stages: **(1) fit a Koopman model** offline from random rollouts, **(2) train the RL agent** with the QP filter active.

### Stage 1 — Koopman model training

```bash
python -m robust_koopman_cbf_rl.train.train_koopman \
  --env_cfg     robust_koopman_cbf_rl/configs/env_cartpole_stab.yaml \
  --koopman_cfg robust_koopman_cbf_rl/configs/koopman.yaml \
  --out         logs/models/cartpole_stab/koopman.npz
```

This:
1. Collects `collection_steps` random transitions from the environment.
2. Fits RBF observables (centers chosen from data) and solves the EDMD regression.
3. Computes per-step residuals and saves the `α`-quantile robust margin `ρ`.
4. Writes `koopman.npz` (model) and `koopman_residuals.npz` (residuals + ρ).

**Output files:**

| File | Contents |
|---|---|
| `koopman.npz` | `A`, `B`, RBF centers, `dim_y`, `n_rbf`, `bandwidth` |
| `koopman_residuals.npz` | `deltas` (per-step projected residuals), `rho`, `alpha` |

**Check ρ before training.** If `rho > 0.5`, the filter will likely be ineffective. Inspect via:

```python
import numpy as np
res = np.load("logs/models/cartpole_stab/koopman_residuals.npz", allow_pickle=True)
print(f"rho = {res['rho']:.4f},  deltas max = {res['deltas'].max():.4f}")
```

### Stage 2a — KCBF-SAC training

```bash
python -m robust_koopman_cbf_rl.train.train_sac_kcbf \
  --env_cfg     robust_koopman_cbf_rl/configs/env_cartpole_stab.yaml \
  --sac_cfg     robust_koopman_cbf_rl/configs/sac_kcbf_cartpole.yaml \
  --model       logs/models/cartpole_stab/koopman.npz \
  --log_dir     logs/kcbf_sweep/cartpole_stab/kcbf_sac/seed_1 \
  --seed        1 \
  --checkpoint_every 50000
```

**Output files:**

| File | Contents |
|---|---|
| `sac_kcbf.csv` | Per-episode log: `step, ep_return, ep_cost, h_min, intervention_rate, slack_rate, cbf_gap_mean, ...` |
| `sac_kcbf_step{N}.pt` | Periodic agent checkpoint (every `checkpoint_every` steps) |
| `sac_kcbf_final.pt` | Final agent checkpoint |
| `eval_final.csv` | 10-episode post-training evaluation |

### Stage 2b — KCBF-PPO training

```bash
python -m robust_koopman_cbf_rl.train.train_ppo_kcbf \
  --env_cfg     robust_koopman_cbf_rl/configs/env_cartpole_stab.yaml \
  --ppo_cfg     robust_koopman_cbf_rl/configs/ppo_kcbf_cartpole.yaml \
  --model       logs/models/cartpole_stab/koopman.npz \
  --log_dir     logs/cartpole/ppo_run1 \
  --checkpoint_every 50000
```

Same output structure with `ppo_kcbf_*` prefixes.

### Standalone evaluation

```bash
python -m robust_koopman_cbf_rl.eval.run_eval \
  --env_cfg    robust_koopman_cbf_rl/configs/env_cartpole_stab.yaml \
  --agent_type sac \
  --agent_ckpt logs/cartpole/sac_run1/sac_kcbf_final.pt \
  --model      logs/models/cartpole_stab/koopman.npz \
  --log_dir    logs/cartpole/sac_eval \
  --n_episodes 20
```

Produces `logs/cartpole/sac_eval/eval_results.csv` with columns: `return`, `total_cost`, `violation_rate`, `episode_violation`, `min_h_value`, plus CBF diagnostics.

### Programmatic API

```python
from robust_koopman_cbf_rl.koopman.model import KoopmanModel
from robust_koopman_cbf_rl.cbf.quadrotor_barriers import Quadrotor2DCompositeAltitudeBarrier
from robust_koopman_cbf_rl.cbf.robust_margin import RobustMargin
from robust_koopman_cbf_rl.cbf.qp_filter import KCBFQPFilter
from robust_koopman_cbf_rl.agents.sac import KCBFSACAgent

# Load pre-trained Koopman model
model = KoopmanModel.load("logs/models/quadrotor2d_track/koopman.npz")

# Build composite altitude barrier (required for Quadrotor 2D)
barrier = Quadrotor2DCompositeAltitudeBarrier(z_min=0.2, z_max=1.8, alpha=1.0, beta=0.5)
rm = RobustMargin(alpha=0.95, mode="global")
rm.update(np.load("logs/models/quadrotor2d_track/koopman_residuals.npz")["deltas"])
flt = KCBFQPFilter(model, barrier, rm, eta=0.9, lam_slack=1e4,
                   u_min=env.action_space.low, u_max=env.action_space.high)

# Load saved agent
agent = KCBFSACAgent.load("logs/kcbf_sweep/quadrotor2d_track/kcbf_sac/seed_1/sac_kcbf_final.pt",
                          koopman_model=model, qp_filter=flt)

# Single step
obs, info = env.reset()
y = info["raw_state"]
z = model.lift(y[:model.observables.dim_y])
u_safe, u_nom, diag = agent.select_action(obs)
# diag keys: h_value, cbf_lhs, cbf_rhs, cbf_gap, slack, correction_norm, intervention, rho

# QP filter directly (e.g. with a baseline policy)
u_nom = lqr_controller(obs)
u_safe, diag = flt.project(z, u_nom)
```

---

## 7. Configuration Reference

### Koopman — per-environment configs

Different environments require very different Koopman dictionaries. Use the appropriate config, not the generic default:

| Config | Recommended for | n_rbf | bandwidth | ridge | collection_steps |
|---|---|---|---|---|---|
| `koopman.yaml` | CartPole (generic default) | 64 | 1.0 | 1e-6 | 50 000 |
| `koopman_quadrotor.yaml` | Quadrotor 2D | 32 | auto (data-driven) | 0.01 | 10 000 |
| `koopman_walker_tuned.yaml` | Walker / HalfCheetah | 256 | auto | 1e-6 | 200 000 |

Additional Koopman parameters (from `koopman.yaml`):

| Key | Default | Description |
|---|---|---|
| `alpha` | 0.95 | Quantile for robust margin ρ |
| `seed` | 0 | RNG seed for center selection and data collection |
| `margin_mode` | `global` | `global` or `cluster` (k-means per-region margin) |
| `n_clusters` | 8 | Number of clusters when `margin_mode: cluster` |
| `residual_dim` | 0 | Lifted-state index for residual projection (must match barrier's active state dim) |

> **`residual_dim` must match the barrier**: for `CartPolePositionBarrier` the active state is `x` at index 0 → `residual_dim: 0`. For `Quadrotor2DCompositeAltitudeBarrier` the active states are `z` (index 2) and `ż` (index 3) → set `residual_dim` to the composite barrier's primary axis.

### SAC — per-environment configs

| Config | Env | total_steps | eta | lam_slack |
|---|---|---|---|---|
| `sac_kcbf_cartpole.yaml` | CartPole stab/track | 150 000 | 0.9 | 10 000 |
| `sac_kcbf_quadrotor.yaml` | Quadrotor 2D stab/track | 500 000 | 0.9 | 10 000 |
| `sac_kcbf_safety_gym.yaml` | Walker / HalfCheetah | 1 000 000 | 0.9 | 1 000 |

Common SAC parameters:

| Key | Value | Description |
|---|---|---|
| `warmup_steps` | 1 000 | Random exploration before learning starts |
| `batch_size` | 256 | Minibatch size for critic/actor updates |
| `buffer_capacity` | 1 000 000 | Replay buffer size |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.005 | Soft target network update rate |
| `lr` | 3e-4 | Adam learning rate |
| `alpha` | 0.2 | SAC entropy temperature |
| `lam_h` | 1.0 | CBF penalty weight in actor loss |

> **η tuning advice:** `η=0.9` is recommended across all environments tested. Lower η (tighter CBF) does not improve safety when ρ is large, and risks policy collapse via excessive QP intervention (see §12 ablation results). Higher η (η → 1) relaxes the constraint too much and increases violation rate.

### PPO (`configs/ppo_kcbf.yaml`)

| Key | Default | Description |
|---|---|---|
| `total_steps` | 1 000 000 | Total environment steps |
| `rollout_len` | 2 048 | Steps per rollout before update |
| `epochs` | 10 | Gradient epochs per rollout |
| `minibatch` | 64 | Minibatch size |
| `lr` | 3e-4 | Adam learning rate |
| `clip` | 0.2 | PPO clipping ratio ε |
| `c_v` | 0.5 | Value loss coefficient |
| `c_e` | 0.01 | Entropy bonus coefficient |
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.95 | GAE λ |
| `lam_h` | 1.0 | CBF penalty weight in actor loss |
| `eta` | 0.9 | CBF decay rate |
| `lam_slack` | 1 000 | QP slack penalty |

### Environment (`configs/env_*.yaml`)

| Key | Description |
|---|---|
| `kind` | `safe_control_gym` or `safety_gymnasium` |
| `env_id` | SCG: `cartpole`, `quadrotor`; safety-gym: `SafetyHalfCheetahVelocity-v1`, `SafetyWalker2dVelocity-v1` |
| `task_config` | Full SCG task dict (forwarded to `scg_make()`) |
| `task_config.task` | `stabilization` or `traj_tracking` |
| `task_config.obs_goal_horizon` | `0` (stabilization) or `1` (tracking — doubles obs dim) |
| `task_config.ctrl_freq` | Control frequency in Hz (50 for CartPole and Quadrotor) |
| `task_config.pyb_freq` | Physics frequency in Hz (**must be 1000 for Quadrotor** — 20 substeps) |
| `task_config.normalized_rl_action_space` | `True` for Quadrotor (maps action to [-1,1]) |
| `velocity_limit` | Velocity constraint for safety-gymnasium envs |
| `seed` | Environment seed |

---

## 8. Baselines

All baselines use the same gymnasium-compatible environment interface.

| Baseline | File | Description |
|---|---|---|
| **LQR** | `baselines/lqr.py` | Linear Quadratic Regulator using linearised dynamics |
| **PID** | `baselines/pid.py` | Proportional-Integral-Derivative controller |
| **Reward Penalty** | `baselines/penalty_rl.py` | Standard RL with cost term added to reward: `r' = r - λ·cost` |
| **Lagrangian RL** | `baselines/lagrangian_rl.py` | Dual variable update: `λ ← max(0, λ + lr·(mean_cost - budget))` |
| **Physical CBF-QP** | `baselines/physical_cbf_qp.py` | CBF-QP with known (linearised) dynamics instead of Koopman model |

The `LagrangianDual` class can be used as a mixin with any RL agent:

```python
from robust_koopman_cbf_rl.baselines.lagrangian_rl import LagrangianDual
dual = LagrangianDual(init_value=0.0, lr=0.01, budget=0.0)
# After each episode:
lam = dual.update(mean_cost=ep_cost)
# Scale the constraint penalty in the loss by lam
```

**Note on SAC+Lagrangian for Walker/HalfCheetah:** In paper experiments, SAC+Lagrangian achieves lower violation rates than KCBF-SAC on these environments (3.3% vs 37% on Walker) because it does not depend on EDMD model accuracy. When ρ is large (contact-rich environments), consider SAC+Lagrangian as a practical alternative.

---

## 9. Evaluation and Plotting

### Training dashboard

```python
from robust_koopman_cbf_rl.plots.plot_training import plot_training

plot_training(
    csv_paths=["logs/sac_run1/sac_kcbf.csv", "logs/ppo_run1/ppo_kcbf.csv"],
    labels=["SAC", "PPO"],
    out_path="logs/training_dashboard.png",
    smooth=10,   # rolling window for smoothing
)
```

Produces a 2×2 panel figure: **Episode Return**, **Episode Cost**, **Mean CBF Gap**, **Intervention Rate**.

### Return / violation / intervention curves

```python
from robust_koopman_cbf_rl.plots.plot_returns import plot_returns
from robust_koopman_cbf_rl.plots.plot_violations import plot_violations
from robust_koopman_cbf_rl.plots.plot_intervention_rate import plot_intervention_rate

plot_returns(["logs/sac_run1/sac_kcbf.csv"], ["SAC"], "logs/returns.png")
plot_violations(["logs/sac_run1/sac_kcbf.csv"], ["SAC"], "logs/violations.png")
plot_intervention_rate(["logs/sac_run1/sac_kcbf.csv"], ["SAC"], "logs/intervention.png")
```

### Trajectory plot

```python
from robust_koopman_cbf_rl.plots.plot_trajectory import collect_trajectory, plot_trajectory

traj = collect_trajectory(env, agent, model, flt, seed=42)
plot_trajectory(
    traj,
    state_names=["x", "ẋ", "θ", "θ̇"],
    ref_traj=None,        # provide (T, dim_y) array for tracking tasks
    out_path="logs/trajectory.png",
)
```

Each state dimension is plotted in its own subplot (reference overlaid as dashed line when provided). The bottom panel shows `h(z)` with a red dashed line at 0 marking the safety boundary.

### Multi-model comparison

```bash
python -m robust_koopman_cbf_rl.eval.compare_models \
  --csvs  logs/sac_eval/eval_results.csv logs/ppo_eval/eval_results.csv \
  --labels SAC PPO \
  --out   logs/comparison.png
```

Produces 4 bar charts (mean ± std): **Episode Return**, **Episode Cost**, **Violation Rate**, **Min h(z)**.

### Diagnostics

The `DiagnosticsBuffer` accumulates per-step QP filter diagnostics and computes episode summaries. The CSV logger writes these columns automatically:

| Column | Description |
|---|---|
| `intervention_rate` | Fraction of steps where QP modified the action |
| `slack_rate` | Fraction of steps with non-zero slack variable |
| `slack_mean` | Mean slack value |
| `cbf_gap_min` | Minimum `a^T u + ξ - b` over episode (negative = violation) |
| `cbf_gap_mean` | Mean CBF gap |
| `h_min` | Minimum barrier value `h_K(z)` per episode (negative = unsafe region entered) |

---

## 10. Performance Criteria per Environment

Two families of metrics are tracked:

- **Safety** — non-negotiable; should hold on every evaluation episode.
- **Task performance** — quantifies reward optimality and filter efficiency.

### How to read the metrics

| Metric | Source | Target direction |
|---|---|---|
| `episode_violation` | SCG constraint checker (coarser bounds) | = 0.0 (no episode violated) |
| `min_h_value` | KCBF barrier (tighter bounds) | > 0.0 (barrier never breached) |
| `violation_rate` | Fraction of steps with `h_value < 0` | = 0.0 |
| `return` | Sum of rewards over episode | Maximise, env-specific target below |
| `intervention_rate` | Fraction of steps QP modified the action | Minimise (lower = policy learned safe naturally) |
| `cbf_gap_mean` | Mean `a^T u + ξ − b` (≥ 0 means constraint met) | > 0.0 at all times |
| `slack_rate` | Fraction of steps where QP required slack | = 0.0 (any slack degrades certificate) |

> **Invariant:** `episode_violation = 0` and `min_h_value > 0` must both hold simultaneously. `min_h_value > 0` is the stronger claim — the CBF bound is 10–25 cm inside the SCG constraint boundary.

---

### 10.1 CartPole — Stabilization

**Episode structure:** 5 s × 50 Hz = **250 steps max**. Reward = +1 per survived step. Early termination if `|x| > 2.4 m` or `|θ| > 0.4 rad`. Random initialisation covers `x ∈ [−1.5, 1.5]` m.

| Criterion | Acceptable | Good | Excellent | **Achieved (KCBF-SAC)** |
|---|---|---|---|---|
| `return` (avg) | ≥ 150 | ≥ 200 | ≥ 235 | **88.6 ± 0.3** *(normalized, ~222/250)* |
| `violation_rate` | < 0.05 | < 0.01 | = 0.000 | **0.000 ± 0.000** |
| `min_h_value` | > 0 | > 0 | > 0.05 m | **1.94 m** |
| `intervention_rate` | < 0.30 | < 0.15 | < 0.05 | **0.000** |
| `cbf_gap_mean` | > 0 | > 0.02 | > 0.05 | > 0 |

**Notes:**
- KCBF-SAC matches unconstrained SAC return (88.5 ± 0.6) with zero violations — no performance-safety tradeoff.
- SAC+Lagrangian fails to converge reliably (34.0 ± 33.4 return) due to oscillating dual variables.
- The CBF bound `x_max = 2.2 m` sits 20 cm inside the episode boundary at 2.4 m.

---

### 10.2 CartPole — Trajectory Tracking

**Episode structure:** 5 s × 50 Hz = **250 steps max**. Observation doubles to `[state (4D), reference (4D)]`. Reference is a 2-cycle circle trajectory.

| Criterion | Acceptable | Good | Excellent | **Achieved (KCBF-SAC)** |
|---|---|---|---|---|
| `return` (avg) | ≥ 100 | ≥ 175 | ≥ 220 | **97.0 ± 1.5** |
| `violation_rate` | < 0.05 | < 0.01 | = 0.000 | **0.000 ± 0.000** |
| `min_h_value` | > 0 | > 0 | > 0.05 m | **1.81 m** |
| `intervention_rate` | < 0.35 | < 0.20 | < 0.08 | **0.0002** |

**Notes:**
- Exceeds unconstrained SAC return (95.6 ± 0.5) while achieving zero violations — tracking reference keeps the cart away from the barrier boundary.
- Intervention rate ≈ 0.02% indicates the actor has fully internalized the constraint.

---

### 10.3 Quadrotor 2D — Stabilization

**Episode structure:** 6 s × 50 Hz = **300 steps max**. Goal: `(x, z) = (0, 1)` m.

| Criterion | Acceptable | Good | Excellent | **Achieved (KCBF-SAC)** |
|---|---|---|---|---|
| `return` (avg) | ≥ −50 | ≥ 0 | ≥ 50 | **195.5 ± 1.6** |
| `violation_rate` | < 0.05 | < 0.01 | < 0.005 | **0.004 ± 0.001** |
| `min_h_value` (altitude) | > 0 | > 0 | > 0.10 m | **0.364 m** |
| `intervention_rate` | < 0.40 | < 0.20 | < 0.10 | **0.006** |

**Notes:**
- All methods achieve comparable return (~195–196); safety differences are small but KCBF-SAC (0.41%) slightly improves over SAC (0.64%).
- Must use `Quadrotor2DCompositeAltitudeBarrier`; the naïve `Quadrotor2DAltitudeBarrier` produces a degenerate QP.

---

### 10.4 Quadrotor 2D — Tracking

**Episode structure:** 6 s × 50 Hz = **300 steps max**. Reference is a figure-8.

| Criterion | Acceptable | Good | Excellent | **Achieved (KCBF-SAC)** |
|---|---|---|---|---|
| `return` (avg) | ≥ −100 | ≥ −20 | ≥ 30 | **170.6 ± 32.1** |
| `violation_rate` | < 0.05 | < 0.01 | < 0.005 | **0.004 ± 0.001** |
| `min_h_value` (altitude) | > 0 | > 0 | > 0.05 m | **0.239 m** |

**Notes:**
- **96.8% violation reduction** vs unconstrained SAC (12.55% → 0.40%). Key result from composite barrier fix.
- Return variance is high (std=32.1) due to episodic slack activations during aggressive figure-8 maneuvers. When slack is activated, the certificate is formally weakened but the average violation rate remains below 0.4%.
- ρ = 1.3×10⁻³ confirms EDMD captures quadrotor dynamics well once the correct barrier is used.

---

### 10.5 Quadrotor 3D — Stabilization / Tracking

**Episode structure:** 6 s × 50 Hz = **300 steps max**. 12D state, 4 motors.

| Criterion | Acceptable | Good | Excellent |
|---|---|---|---|
| `return` (avg) | ≥ −150 | ≥ −50 | ≥ 0 |
| Episode completion rate | ≥ 0.4 | ≥ 0.65 | ≥ 0.85 |
| `violation_rate` | < 0.10 | < 0.05 | < 0.01 |
| `min_h_value` (altitude) | > 0 | > 0 | > 0.05 m |
| `intervention_rate` | < 0.50 | < 0.30 | < 0.15 |

*Note: 3D Quadrotor experiments were not included in the paper submission. Results pending.*

---

### 10.6 SafetyHalfCheetah — Velocity Constrained

**Episode structure:** 1000 steps. Reward = forward velocity. Cost = 1 per step exceeding `velocity_limit = 2.0 m/s`.

| Criterion | Acceptable | Good | Excellent | **Achieved (KCBF-SAC)** |
|---|---|---|---|---|
| `return` (avg, 1000 steps) | ≥ 1500 | ≥ 2500 | ≥ 3500 | **2290 ± 22** |
| `violation_rate` | < 0.50 | < 0.20 | < 0.10 | **0.828 ± 0.008** |
| `min_h_value` | > 0 | > 0 | > 0.1 | **-18.82** *(unsafe)* |

**Notes:**
- ρ = 1.782 — contact dynamics produce non-smooth velocity transitions that linear EDMD cannot capture. The filter is structurally inactive for most of training.
- KCBF-SAC achieves modest violation reduction (97.9% → 82.8%) at high return cost (7741 → 2290, −70%).
- `min_h_value = -18.82` confirms the velocity constraint is deeply violated; the CBF certificate is not maintained.
- **Recommendation:** Use SAC+Lagrangian for this environment until a multi-step or deep Koopman extension is available.

---

### 10.7 SafetyWalker2d — Velocity Constrained

**Episode structure:** 1000 steps. Velocity constraint: `v_x ≤ 2.0 m/s`.

| Criterion | Acceptable | Good | Excellent | **Achieved (KCBF-SAC)** | **Best baseline** |
|---|---|---|---|---|---|
| `return` (avg, 1000 steps) | ≥ 800 | ≥ 1500 | ≥ 2200 | **2607 ± 202** | SAC: 3803 ± 292 |
| `violation_rate` | < 0.20 | < 0.10 | < 0.05 | **0.370 ± 0.226** | SAC+Lag: **0.033 ± 0.015** |
| `min_h_value` | > 0 | > 0 | > 0.1 | **-7.81** *(unsafe)* | — |
| `intervention_rate` | < 0.25 | < 0.12 | < 0.05 | 0.039 | — |

**Notes:**
- ρ = 0.698 — filter is marginal. KCBF-SAC reduces violations from 77.8% (SAC) to 37.0%, but SAC+Lagrangian achieves 3.3% with comparable return at much lower variance.
- High `violation_rate` variance (std=0.226) indicates seeds produce qualitatively different policies; some seeds have near-zero violations while others fail entirely.
- `min_h_value = -7.81` confirms the velocity barrier is regularly breached despite QP filter.
- **Recommendation:** Use SAC+Lagrangian for Walker when safety is the primary objective. KCBF-SAC is preferable when interpretable filter diagnostics are needed.

---

### Summary table

| Environment | Max steps | **KCBF-SAC return** | **KCBF-SAC viol. rate** | Best baseline (safety) |
|---|---|---|---|---|
| CartPole Stabilization | 250 | 88.6 ± 0.3 | **0.000** | SAC: 0.230 |
| CartPole Tracking | 250 | 97.0 ± 1.5 | **0.000** | SAC+Pen: 0.183 |
| Quadrotor 2D Stab | 300 | 195.5 ± 1.6 | **0.004** | SAC+Pen: 0.003 |
| Quadrotor 2D Track | 300 | 170.6 ± 32.1 | **0.004** | SAC+Pen: 0.007 |
| Safety HalfCheetah | 1000 | 2290 ± 22 | 0.828 | SAC+Lag: 0.977 *(all fail)* |
| Safety Walker | 1000 | 2607 ± 202 | 0.370 | **SAC+Lag: 0.033** |

---

## 11. Experiment Sweep Pipeline

All paper experiments are reproducible via four scripts in `experiments/`:

### Step 1 — Fit Koopman models

```bash
# Fit one model per environment (run once per env)
for ENV in cartpole_stab cartpole_track quadrotor2d_stab quadrotor2d_track; do
  python -m robust_koopman_cbf_rl.train.train_koopman \
    --env_cfg  robust_koopman_cbf_rl/configs/env_${ENV}.yaml \
    --koopman_cfg robust_koopman_cbf_rl/configs/koopman.yaml \
    --out logs/models/${ENV}/koopman.npz
done

for ENV in safety_walker safety_halfcheetah; do
  python -m robust_koopman_cbf_rl.train.train_koopman \
    --env_cfg  robust_koopman_cbf_rl/configs/env_${ENV}.yaml \
    --koopman_cfg robust_koopman_cbf_rl/configs/koopman_walker_tuned.yaml \
    --out logs/models/${ENV}/koopman.npz
done
```

### Step 2 — Baseline sweep (SAC, PPO, Penalty, Lagrangian, LQR)

```bash
python experiments/run_sweep.py \
  --log_root  logs/baseline_sweep \
  --seeds 1 2 3 \
  --workers 4
```

CLI options:
- `--envs cartpole_stab cartpole_track ...` — subset of environments
- `--baselines sac sac_lagrangian ...` — subset of methods
- `--total_steps 150000` — override training budget
- `--skip_classical` — skip LQR/PID

### Step 3 — KCBF-SAC sweep

```bash
python experiments/run_kcbf_sweep.py \
  --model_dir logs/models \
  --log_root  logs/kcbf_sweep \
  --seeds 1 2 3 \
  --workers 4
```

CLI options:
- `--envs` / `--variants kcbf_sac kcbf_ppo` — subset filters
- `--total_steps` — override training budget
- `--skip_koopman` — if models already exist

### Step 4 — Aggregate results and generate all figures

```bash
python experiments/compare_all.py \
  --baseline_root logs/baseline_sweep \
  --kcbf_root     logs/kcbf_sweep \
  --out_dir       logs/final_results \
  --skip_training_curves
```

Produces in `logs/final_results/`:
- `aggregate_all_summary.csv` — one row per (env, method, seed)
- `latex_table.txt` — LaTeX tabular for paper
- `scg_methods_comparison.png` — bar charts for safe-control-gym envs
- `safetygym_methods_comparison.png` — bar charts for Safety Gymnasium envs
- `pareto_safety_efficiency.png` — return vs violation Pareto scatter
- `kcbf_diagnostics.png` — intervention rate, slack rate, h_min per env

### Step 5 — CBF parameter ablation (η × λ_slack)

```bash
python experiments/ablation_cbf_params.py \
  --envs safety_walker \
  --eta 0.5 0.7 0.9 \
  --lam_slack 1000 5000 10000 \
  --model_dirs safety_walker=logs/models/safety_walker \
  --log_root logs/ablation \
  --total_steps 300000 \
  --workers 4
```

For 1M convergence runs of specific configs:

```bash
python -m robust_koopman_cbf_rl.train.train_sac_kcbf \
  --env_cfg  robust_koopman_cbf_rl/configs/env_safety_walker.yaml \
  --sac_cfg  robust_koopman_cbf_rl/configs/sac_kcbf_safety_gym.yaml \
  --model    logs/models/safety_walker/koopman.npz \
  --log_dir  logs/ablation_1M/safety_walker/eta0.7_lam10000 \
  --eta 0.7 --lam_slack 10000 --total_steps 1000000
```

---

## 12. Experimental Results Summary

Results from 3 seeds × 6 environments. All metrics are mean ± std over seeds; evaluation uses 10 episodes per checkpoint.

### CartPole (zero-violation)

| Method | Return ↑ | Violation Rate ↓ | h_min |
|---|---|---|---|
| **KCBF-SAC** | **88.6 ± 0.3** | **0.000 ± 0.000** | **1.94** |
| SAC | 88.5 ± 0.6 | 0.230 ± 0.017 | — |
| SAC+Lagrangian | 34.0 ± 33.4 | 0.204 ± 0.002 | — |
| SAC+Penalty | 54.4 ± 38.1 | 0.218 ± 0.055 | — |
| LQR | 5.5 | 0.164 | — |

KCBF-SAC matches unconstrained SAC return with zero violations. Lagrangian and Penalty methods fail due to oscillating dual variables or insufficient reward shaping. Projected residual ρ = 9×10⁻⁴ — EDMD captures CartPole dynamics nearly exactly.

### Quadrotor 2D Tracking (96.8% violation reduction)

| Method | Return ↑ | Violation Rate ↓ |
|---|---|---|
| **KCBF-SAC** | 170.6 ± 32.1 | **0.004 ± 0.001** |
| SAC | 195.8 ± 0.2 | 0.125 ± 0.014 |
| SAC+Lagrangian | 194.3 ± 0.7 | 0.015 ± 0.004 |
| SAC+Penalty | 192.7 ± 1.1 | 0.007 ± 0.001 |

Key finding: the `Quadrotor2DCompositeAltitudeBarrier` is required. The naïve `Quadrotor2DAltitudeBarrier` produces a degenerate QP with 40× lower filter control authority. KCBF-SAC return variance (std=32.1) reflects episodic slack activations during aggressive figure-8 tracking.

### Walker Ablation (η × λ at 1M steps, single seed)

| Configuration | Return ↑ | Violation Rate ↓ | Intervention Rate | Outcome |
|---|---|---|---|---|
| η=0.9 (default, 3 seeds) | 2607 ± 202 | 0.370 ± 0.226 | 3.9% | Stable |
| η=0.7, λ=10k (1M) | ~26 | 6.7% | **85%** | **Policy collapse** |
| η=0.5, λ=10k (1M) | 3405 | 84.5% | 6.2% | **Infeasibility — no filter** |
| SAC+Lagrangian (3 seeds) | 2801 ± 9 | **0.033 ± 0.015** | — | Best safety |

- **η=0.7**: Tight CBF causes QP to override actor 85% of steps → actor cannot learn → bimodal policy (some episodes safe, others catastrophic), mean return collapses to ≈26.
- **η=0.5**: Even tighter constraint → frequently infeasible under actuator limits → slack dominates → effective filter intervention drops to 6.2% → policy learns freely but unsafely.
- **Lesson**: With large ρ (Walker: 0.698), η=0.9 is the only stable operating point among those tested. Do not lower η to try to improve safety when ρ is large.

---

## 13. Known Limitations

### 1. Relative degree requirement

The CBF-QP requires the lifted barrier to have relative degree 1 (i.e., `c^T B ≠ 0`). For constraints where the safety-relevant state is indirectly actuated (e.g., altitude via thrust in quadrotor dynamics), the naïve lifted barrier has `c^T B ≈ 0` and the filter is degenerate.

**Fix:** Use `Quadrotor2DCompositeAltitudeBarrier(z_min, z_max, alpha, beta)` which encodes `h = α(z − z_min) + β·ż`. This is a case-by-case solution; a general high-order KCBF extension is future work.

### 2. Large prediction error (ρ) disables the filter

When ρ ≥ typical `h_K(z)`, the robust CBF constraint either:
- Becomes **trivially satisfied** when `h_K(z) < 0` (already unsafe): any control satisfies it.
- Becomes **infeasible** when `h_K(z) > 0`: the constraint requires the next h to exceed ρ, which actuator limits may prohibit.

Environments with contact dynamics (HalfCheetah: ρ=1.782, Walker: ρ=0.698) are fundamentally difficult for first-order linear EDMD. Compute ρ before training to decide whether KCBF is appropriate.

### 3. Policy collapse under over-constraint

Tight CBF parameters (η=0.7 on Walker) cause the QP to dominate actor updates, preventing learning. This is absent from Lagrangian methods, which enforce constraints through a differentiable objective rather than hard action substitution.

### 4. One-step certificate only

The CBF condition is enforced step-wise. Trajectories can approach the unsafe region across many steps even if each individual step nominally satisfies the constraint.

### 5. Slack degrades the safety certificate

When `ξ > 0` (the QP is infeasible without slack), the filter no longer enforces a formal CBF condition at that step. Monitor `slack_rate` in training logs. A slack rate above ~5% suggests the filter is frequently over-ridden, and the certificate is unreliable.

### 6. Static, hand-specified barrier functions

Barriers are designed per environment. For new environments, a barrier must be selected and its relative degree verified manually. Joint learning of the Koopman lifting and barrier certificate is an active research direction.

---

## 14. Testing

```bash
# Fast unit tests (no environment required)
pytest robust_koopman_cbf_rl/tests/ \
  --ignore=robust_koopman_cbf_rl/tests/test_safe_control_gym_wrapper.py \
  --ignore=robust_koopman_cbf_rl/tests/test_safety_gymnasium_wrapper.py \
  --ignore=robust_koopman_cbf_rl/tests/test_train_smoke.py \
  --ignore=robust_koopman_cbf_rl/tests/test_integration_cartpole.py \
  -k "not test_physical_cbf_filter" \
  -v

# Full suite (requires safe-control-gym and qpsolvers)
pytest robust_koopman_cbf_rl/tests/ -v
```

| Test file | What it covers |
|---|---|
| `test_barriers.py` | Barrier values and lifted coefficients for all barrier types, including `Quadrotor2DCompositeAltitudeBarrier` |
| `test_observables.py` | RBF lifting, z_dim calculation, center fitting |
| `test_fit_edmd.py` | EDMD regression correctness |
| `test_koopman_model.py` | Predict, save, load round-trip |
| `test_residuals.py` | Residual projection and quantile margin |
| `test_robust_margin.py` | Global and cluster margin modes |
| `test_qp_filter.py` | QP pass-through on safe input, correction on unsafe input *(requires qpsolvers)* |
| `test_sac_agent.py` | SAC select_action, actor/critic structure |
| `test_ppo_agent.py` | PPO select_action, update statistics |
| `test_checkpoint.py` | SAC/PPO save and load round-trip |
| `test_plots.py` | Plot functions write valid output files |
| `test_replay_buffer.py` | SAC replay buffer add/sample |
| `test_rollout_buffer.py` | PPO rollout buffer + GAE computation |
| `test_metrics.py` | Episode metric computation |
| `test_baselines.py` | Reward penalty, Lagrangian dual, physical CBF |
| `test_safe_control_gym_wrapper.py` | SCG wrapper (requires safe-control-gym) |
| `test_integration_cartpole.py` | End-to-end CartPole loop (requires safe-control-gym + qpsolvers) |
