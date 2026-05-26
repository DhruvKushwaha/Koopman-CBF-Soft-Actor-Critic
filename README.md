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
11. [Testing](#11-testing)

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
│   ├── quadrotor_barriers.py   # Quadrotor2DAltitudeBarrier, Quadrotor2DPitchBarrier,
│   │                           # Quadrotor3DPositionBarrier
│   ├── velocity_barriers.py    # VelocityNormBarrier (safety-gymnasium)
│   ├── robust_margin.py        # RobustMargin: global or cluster-based ρ
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
│   ├── plot_training.py        # 4-panel training dashboard (return, cost, CBF gap, intervention)
│   └── plot_trajectory.py      # collect_trajectory() + per-state-dim plot vs reference
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
| CBF barrier | `CartPolePositionBarrier(x_max=2.2)` — 10 cm margin inside episode boundary |

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
| CBF barrier | `Quadrotor2DAltitudeBarrier(z_min=0.2, z_max=1.8)` |

> **Important:** `normalized_rl_action_space: True` is set so the actor's tanh output maps cleanly to the physical thrust range. Without normalization, the actor saturates at ≈ 0.29 N and cannot decrease thrust — training diverges.

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

| Config file | Environment ID | Constraint |
|---|---|---|
| `env_safety_halfcheetah.yaml` | `SafetyHalfCheetahVelocity-v1` | velocity ≤ 2.0 m/s |
| `env_safety_walker.yaml` | `SafetyWalker2dVelocity-v1` | velocity ≤ 2.0 m/s |

The `velocity_limit` field in the YAML controls which CBF barrier is active (`VelocityNormBarrier`).

---

## 5. Safety Barriers

All barriers inherit from `SafetyConstraint` (`cbf/barrier_base.py`) and implement:

- `value(raw_state, info) → float` — barrier value `h(y)` at the current raw state
- `lifted_barrier_coeffs(z_dim, ...) → (c, d)` — linear coefficients so `h_K(z) = c^T z + d`

### 5.1 CartPole barriers

| Class | Formula | Default |
|---|---|---|
| `CartPolePositionBarrier(x_max, side)` | `h = x_max - x` (right) or `h = x_max + x` (left) | `x_max=2.2` |
| `CartPoleAngleBarrier(theta_max, side)` | `h = θ_max - θ` (right) or `h = θ_max + θ` (left) | `theta_max=0.14` rad |

Both appear as linear maps on the first (identity) part of the lifted state `z`, with the RBF portion zeroed out.

### 5.2 Quadrotor barriers

| Class | Formula |
|---|---|
| `Quadrotor2DAltitudeBarrier(z_min, z_max, side)` | `h = z_max - z` (upper) or `h = z - z_min` (lower) |
| `Quadrotor2DPitchBarrier(theta_max, side)` | `h = θ_max - θ` |
| `Quadrotor3DPositionBarrier(axis_index, lo, hi, side)` | `h = hi - x[axis]` (upper) or `h = x[axis] - lo` (lower) |

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
  --env_cfg  robust_koopman_cbf_rl/configs/env_cartpole_stab.yaml \
  --koopman_cfg robust_koopman_cbf_rl/configs/koopman.yaml \
  --out logs/cartpole/koopman.npz
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

### Stage 2a — KCBF-SAC training

```bash
python -m robust_koopman_cbf_rl.train.train_sac_kcbf \
  --env_cfg     robust_koopman_cbf_rl/configs/env_cartpole_stab.yaml \
  --sac_cfg     robust_koopman_cbf_rl/configs/sac_kcbf.yaml \
  --model       logs/cartpole/koopman.npz \
  --log_dir     logs/cartpole/sac_run1 \
  --checkpoint_every 50000
```

**Output files:**

| File | Contents |
|---|---|
| `sac_kcbf.csv` | Per-episode log: `step, ep_return, ep_cost, intervention_rate, cbf_gap_mean, ...` |
| `sac_kcbf_step{N}.pt` | Periodic agent checkpoint (every `checkpoint_every` steps) |
| `sac_kcbf_final.pt` | Final agent checkpoint |
| `eval_final.csv` | 10-episode post-training evaluation |

### Stage 2b — KCBF-PPO training

```bash
python -m robust_koopman_cbf_rl.train.train_ppo_kcbf \
  --env_cfg     robust_koopman_cbf_rl/configs/env_cartpole_stab.yaml \
  --ppo_cfg     robust_koopman_cbf_rl/configs/ppo_kcbf_cartpole.yaml \
  --model       logs/cartpole/koopman.npz \
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
  --model      logs/cartpole/koopman.npz \
  --log_dir    logs/cartpole/sac_eval \
  --n_episodes 20
```

Produces `logs/cartpole/sac_eval/eval_results.csv` with columns: `return`, `total_cost`, `violation_rate`, `episode_violation`, `min_h_value`, plus CBF diagnostics.

### Programmatic API

```python
from robust_koopman_cbf_rl.koopman.model import KoopmanModel
from robust_koopman_cbf_rl.cbf.cartpole_barriers import CartPolePositionBarrier
from robust_koopman_cbf_rl.cbf.robust_margin import RobustMargin
from robust_koopman_cbf_rl.cbf.qp_filter import KCBFQPFilter
from robust_koopman_cbf_rl.agents.sac import KCBFSACAgent

# Load pre-trained Koopman model
model = KoopmanModel.load("logs/cartpole/koopman.npz")

# Build safety filter
barrier = CartPolePositionBarrier(x_max=2.2, side="right")
rm = RobustMargin(alpha=0.95, mode="global")
rm.update(np.load("logs/cartpole/koopman_residuals.npz")["deltas"])
flt = KCBFQPFilter(model, barrier, rm, eta=0.5, lam_slack=1e3,
                   u_min=env.action_space.low, u_max=env.action_space.high)

# Load saved agent
agent = KCBFSACAgent.load("logs/cartpole/sac_run1/sac_kcbf_final.pt",
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

### Koopman (`configs/koopman.yaml`)

| Key | Default | Description |
|---|---|---|
| `n_rbf` | 64 | Number of RBF centers in the observable dictionary |
| `bandwidth` | 1.0 | RBF bandwidth σ (Gaussian kernel) |
| `ridge` | 1e-6 | L2 regularisation for EDMD regression |
| `alpha` | 0.95 | Quantile for robust margin ρ |
| `collection_steps` | 50 000 | Random steps for Koopman dataset |
| `seed` | 0 | RNG seed for center selection and data collection |
| `margin_mode` | `global` | `global` or `cluster` (k-means per-region margin) |
| `n_clusters` | 8 | Number of clusters when `margin_mode: cluster` |
| `residual_dim` | 0 | Lifted-state index for residual projection (must match barrier's active state dim) |

### SAC (`configs/sac_kcbf.yaml`)

| Key | Default | Description |
|---|---|---|
| `total_steps` | 200 000 | Total environment steps |
| `warmup_steps` | 1 000 | Random exploration before learning starts |
| `batch_size` | 256 | Minibatch size for critic/actor updates |
| `buffer_capacity` | 1 000 000 | Replay buffer size |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.005 | Soft target network update rate |
| `lr` | 3e-4 | Adam learning rate |
| `alpha` | 0.2 | SAC entropy temperature |
| `lam_h` | 1.0 | CBF penalty weight in actor loss |
| `eta` | 0.5 | CBF decay rate (in QP filter) |
| `lam_slack` | 1 000 | Slack variable penalty (QP infeasibility cost) |

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
| `eta` | 0.5 | CBF decay rate |
| `lam_slack` | 1 000 | QP slack penalty |

### Environment (`configs/env_*.yaml`)

| Key | Description |
|---|---|
| `kind` | `safe_control_gym` or `safety_gymnasium` |
| `env_id` | SCG: `cartpole`, `quadrotor`; safety-gym: `SafetyHalfCheetahVelocity-v1` etc. |
| `task_config` | Full SCG task dict (forwarded to `scg_make()`) |
| `task_config.task` | `stabilization` or `traj_tracking` |
| `task_config.obs_goal_horizon` | `0` (stabilization) or `1` (tracking — doubles obs dim) |
| `task_config.ctrl_freq` | Control frequency in Hz (50 for CartPole and Quadrotor) |
| `task_config.pyb_freq` | Physics frequency in Hz (**must be 1000 for Quadrotor** — 20 substeps) |
| `task_config.normalized_rl_action_space` | `True` for Quadrotor (maps action to [-1,1]) |
| `velocity_limit` | Velocity constraint for safety-gymnasium envs |
| `seed` | Environment seed |

> **`residual_dim` must match the barrier**: for `CartPolePositionBarrier` the active state is `x` at index 0 → `residual_dim: 0`. For `Quadrotor2DAltitudeBarrier` the active state is `z` at index 2 → `residual_dim: 2`.

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

Or from Python:

```python
from robust_koopman_cbf_rl.eval.compare_models import compare_models

compare_models(
    csv_paths=["logs/sac_eval/eval_results.csv", "logs/ppo_eval/eval_results.csv"],
    labels=["SAC", "PPO"],
    out_path="logs/comparison.png",
)
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
| `h_min` | Minimum barrier value `h_K(z)` (negative = unsafe) |

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

> **Invariant:** `episode_violation = 0` and `min_h_value > 0` must both hold simultaneously. `min_h_value > 0` is the stronger claim — the CBF bound is 10–25 cm inside the SCG constraint boundary.

---

### 10.1 CartPole — Stabilization

**Episode structure:** 5 s × 50 Hz = **250 steps max**. Reward = +1 per survived step (`rl_reward`). Early termination if `|x| > 2.4 m` or `|θ| > 0.4 rad`. Random initialisation covers `x ∈ [−1.5, 1.5]` m.

| Criterion | Acceptable | Good | Excellent |
|---|---|---|---|
| `return` (avg) | ≥ 150 | ≥ 200 | ≥ 235 |
| `episode_violation` | = 0.0 | = 0.0 | = 0.0 |
| `min_h_value` | > 0 | > 0 | > 0.05 m |
| `intervention_rate` | < 0.30 | < 0.15 | < 0.05 |
| `cbf_gap_mean` | > 0 | > 0.02 | > 0.05 |

**Notes:**
- Max return of 250 corresponds to a perfect stabiliser that never terminates early. Given random initialisation from up to ±1.5 m, achieving ≥ 200 consistently indicates a robust policy.
- `intervention_rate > 0.30` late in training typically means the policy has not yet learned to avoid the barrier proactively — the QP filter is compensating for a poor nominal policy.
- The CBF bound `x_max = 2.2 m` sits 20 cm inside the episode boundary at 2.4 m. `min_h_value > 0` is therefore a strictly stronger guarantee than `episode_violation = 0`.

---

### 10.2 CartPole — Trajectory Tracking

**Episode structure:** 5 s × 50 Hz = **250 steps max**. Observation doubles to `[state (4D), reference (4D)]` with `obs_goal_horizon: 1`. Reference is a 2-cycle circle trajectory; the reward is shaped by tracking error.

| Criterion | Acceptable | Good | Excellent |
|---|---|---|---|
| `return` (avg) | ≥ 100 | ≥ 175 | ≥ 220 |
| `episode_violation` | = 0.0 | = 0.0 | = 0.0 |
| `min_h_value` | > 0 | > 0 | > 0.05 m |
| x-position RMSE | < 0.50 m | < 0.25 m | < 0.10 m |
| `intervention_rate` | < 0.35 | < 0.20 | < 0.08 |

**Notes:**
- Tracking is harder than stabilisation: the reference visits `x ≈ ±1.0 m`, placing the cart close to the barrier boundary during parts of the trajectory.
- `x-position RMSE` is not logged automatically; compute from `info["tracking_error"]` or the `plot_trajectory` output.
- A RMSE < 0.10 m at `ctrl_freq = 50 Hz` is comparable to LQR performance on the linearised model.

---

### 10.3 Quadrotor 2D — Stabilization

**Episode structure:** 6 s × 50 Hz = **300 steps max**. Goal: `(x, z) = (0, 1)` m. The SCG rl_reward is a shaped negative quadratic (deviation from goal + control effort); typical range per episode is roughly −300 to +100 depending on proximity.

| Criterion | Acceptable | Good | Excellent |
|---|---|---|---|
| `return` (avg) | ≥ −50 | ≥ 0 | ≥ 50 |
| Episode completion rate | ≥ 0.6 | ≥ 0.8 | ≥ 0.95 |
| `episode_violation` | = 0.0 | = 0.0 | = 0.0 |
| `min_h_value` (altitude) | > 0 | > 0 | > 0.10 m |
| `intervention_rate` | < 0.40 | < 0.20 | < 0.10 |
| Steady-state altitude error | < 0.20 m | < 0.10 m | < 0.05 m |

**Notes:**
- "Episode completion rate" is the fraction of episodes that run all 300 steps without `done_on_out_of_bound`. Early termination typically means the drone drifted outside `|x| > 2.0 m` or `z > 2.0 m`.
- The altitude barrier `[0.2, 1.8]` m leaves 0.25 m margin to the ground plane and 0.2 m to the observation-space ceiling. `intervention_rate` will be higher than CartPole because the dual-motor Quadrotor 2D has a more aggressive barrier geometry.
- Negative returns are common during early training (the shaped reward penalises every meter of deviation from the 1 m goal altitude).

---

### 10.4 Quadrotor 2D — Trajectory Tracking

**Episode structure:** 6 s × 50 Hz = **300 steps max**. Reference is a figure-8. Observation doubles with `obs_goal_horizon: 1`.

| Criterion | Acceptable | Good | Excellent |
|---|---|---|---|
| `return` (avg) | ≥ −100 | ≥ −20 | ≥ 30 |
| Episode completion rate | ≥ 0.5 | ≥ 0.75 | ≥ 0.9 |
| `episode_violation` | = 0.0 | = 0.0 | = 0.0 |
| `min_h_value` (altitude) | > 0 | > 0 | > 0.05 m |
| x/z-position RMSE | < 0.40 m | < 0.20 m | < 0.08 m |

**Notes:**
- Figure-8 tracking requires coordinated horizontal and vertical motion. The barrier on altitude couples with the vertical trajectory component — the agent must plan ahead to avoid requiring large QP interventions near the altitude bounds.
- Tracking RMSE is evaluated on the `x` and `z` axes separately; the x-axis is typically the harder one as the figure-8 sweeps ±1.5 m.

---

### 10.5 Quadrotor 3D — Stabilization / Tracking

**Episode structure:** 6 s × 50 Hz = **300 steps max**. 12D state, 4 motors. Significantly harder than 2D due to coupling of roll/pitch/yaw and the 4 independent motor constraints.

| Criterion | Acceptable | Good | Excellent |
|---|---|---|---|
| `return` (avg) | ≥ −150 | ≥ −50 | ≥ 0 |
| Episode completion rate | ≥ 0.4 | ≥ 0.65 | ≥ 0.85 |
| `episode_violation` | = 0.0 | = 0.0 | = 0.0 |
| `min_h_value` (altitude) | > 0 | > 0 | > 0.05 m |
| 3D position RMSE | < 0.60 m | < 0.30 m | < 0.12 m |
| `intervention_rate` | < 0.50 | < 0.30 | < 0.15 |

**Notes:**
- The 3D Quadrotor is the most challenging environment. A completion rate ≥ 0.65 with zero violations is already a strong result for KCBF methods.
- The Koopman model accuracy degrades with dimensionality. If `cbf_gap_mean` is consistently low (< 0.01) combined with a high intervention rate, the residual margin `ρ` may be too large — consider re-fitting the Koopman model with more `collection_steps` or larger `n_rbf`.
- `min_h_value ≤ 0` on any evaluation episode is a hard failure for the KCBF framework, regardless of return.

---

### 10.6 SafetyHalfCheetah — Velocity Constrained

**Episode structure:** 1000 steps. Reward = forward velocity (higher is better). Cost = 1 per step where velocity exceeds `velocity_limit = 2.0 m/s`. No episode termination on constraint violation (`done_on_violation: False` in safety-gymnasium).

| Criterion | Acceptable | Good | Excellent |
|---|---|---|---|
| `return` (avg, 1000 steps) | ≥ 1500 | ≥ 2500 | ≥ 3500 |
| `total_cost` (avg) | = 0.0 | = 0.0 | = 0.0 |
| `episode_violation` | = 0.0 | = 0.0 | = 0.0 |
| `min_h_value` | > 0 | > 0 | > 0.1 |
| `intervention_rate` | < 0.30 | < 0.15 | < 0.05 |

**Notes:**
- Unconstrained SAC on HalfCheetah typically achieves ≥ 6000 return per episode. A `velocity_limit = 2.0 m/s` cap reduces the reachable return to roughly 2000–4000 (2.0 m/s × 1000 steps × per-step reward scaling). Achieving 3500 while maintaining zero violations is close to optimal for this constraint.
- `total_cost = 0.0` is the primary safety metric — every non-zero episode cost represents a step where the agent exceeded the velocity limit.
- If `return < 1500` with zero violations, the VelocityNormBarrier margin or the KCBF `η` is too conservative. Try reducing `eta` in the SAC/PPO config.

---

### 10.7 SafetyWalker2d — Velocity Constrained

**Episode structure:** 1000 steps. Same structure as HalfCheetah but with the bipedal Walker2d dynamics.

| Criterion | Acceptable | Good | Excellent |
|---|---|---|---|
| `return` (avg, 1000 steps) | ≥ 800 | ≥ 1500 | ≥ 2200 |
| `total_cost` (avg) | = 0.0 | = 0.0 | = 0.0 |
| `episode_violation` | = 0.0 | = 0.0 | = 0.0 |
| `min_h_value` | > 0 | > 0 | > 0.1 |
| `intervention_rate` | < 0.25 | < 0.12 | < 0.05 |

**Notes:**
- Walker2d is harder to stabilize than HalfCheetah; unconstrained SAC achieves ~3500–5000 return. The 2.0 m/s cap reduces the ceiling to roughly 1500–2500.
- The `VelocityNormBarrier` constraint is quadratic (`h = v_max² − Σvᵢ²`), so the Koopman observable dictionary must include velocity-squared features (`extra_quadratic_indices`). Verify the `z_dim` is consistent with the barrier's `lifted_barrier_coeffs` output.
- Lower `intervention_rate` targets compared to HalfCheetah reflect that Walker2d's walking gait produces more consistent velocities, leaving less need for reactive QP corrections.

---

### Summary table

| Environment | Max steps | Target return | Safety requirement | Ref intervention rate |
|---|---|---|---|---|
| CartPole Stabilization | 250 | ≥ 200 | `min_h_value > 0`, `episode_violation = 0` | < 15% |
| CartPole Tracking | 250 | ≥ 175 | `min_h_value > 0`, `episode_violation = 0` | < 20% |
| Quadrotor 2D Stab | 300 | ≥ 0 | `min_h_value > 0`, `episode_violation = 0` | < 20% |
| Quadrotor 2D Track | 300 | ≥ −20 | `min_h_value > 0`, `episode_violation = 0` | < 25% |
| Quadrotor 3D Stab | 300 | ≥ −50 | `min_h_value > 0`, `episode_violation = 0` | < 30% |
| Quadrotor 3D Track | 300 | ≥ −80 | `min_h_value > 0`, `episode_violation = 0` | < 35% |
| SafetyHalfCheetah | 1000 | ≥ 2500 | `total_cost = 0`, `episode_violation = 0` | < 15% |
| SafetyWalker2d | 1000 | ≥ 1500 | `total_cost = 0`, `episode_violation = 0` | < 12% |

---

## 11. Testing

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
| `test_barriers.py` | Barrier values and lifted coefficients for all barrier types |
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
