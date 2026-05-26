"""Collect per-step constraint diagnostics during evaluation rollouts.

Returns NaN-padded arrays (n_episodes × T_max) for every constraint signal,
suitable for nanmean / nanstd statistics and NPZ storage.

Handles:
  - KCBF agents: select_action returns (..., diag) with h_value, cbf_gap, intervention
  - Baseline agents (NullFilter): diag carries NaN h_value — still records cost/reward
"""
from __future__ import annotations
from pathlib import Path

import numpy as np


def eval_constraint_trace(
    env,
    agent,
    model,
    n_episodes: int = 10,
    out_path: str | None = None,
) -> dict:
    """Roll out agent for n_episodes and collect per-step constraint signals.

    Returns a dict of float64 arrays with shape (n_episodes, T_max),
    NaN-padded beyond each episode's true length:

        h_values        barrier value h(z_t)
        violations      1.0 where h(z_t) < 0, else 0.0 (NaN after episode end)
        cbf_gaps        a^T u + xi - b  (NaN for non-KCBF methods)
        slacks          xi  (NaN for non-KCBF methods)
        interventions   1.0 if ||u_safe - u_nom|| > eps  (NaN for non-KCBF)
        correction_norm ||u_safe - u_nom||  (NaN for non-KCBF)
        rewards         step reward r_t
        costs           step cost c_t from env info
        ep_lengths      (n_episodes,)  true episode length
    """
    episodes: list[dict] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=1000 + ep)
        y = info.get("raw_state", obs)
        z = model.lift(y[:model.observables.dim_y])

        h_vals, gaps, slacks, ints_, corr, rewards, costs = [], [], [], [], [], [], []
        done = False

        while not done:
            result = agent.select_action(obs)
            u_safe = result[0]
            diag: dict = result[-1] if isinstance(result, tuple) and len(result) > 1 else {}

            next_obs, r, term, trunc, info = env.step(u_safe)

            h_vals.append(float(diag.get("h_value", float("nan"))))
            gaps.append(float(diag.get("cbf_gap", float("nan"))))
            slacks.append(float(diag.get("slack", float("nan"))))
            ints_.append(float(diag.get("intervention", float("nan"))))
            corr.append(float(diag.get("correction_norm", float("nan"))))
            rewards.append(float(r))
            costs.append(float(info.get("cost", 0.0)))

            obs = next_obs
            y = info.get("raw_state", obs)
            z = model.lift(y[:model.observables.dim_y])
            done = bool(term) or bool(trunc)

        episodes.append({k: np.array(v, dtype=np.float64) for k, v in {
            "h_values": h_vals, "cbf_gaps": gaps, "slacks": slacks,
            "interventions": ints_, "correction_norm": corr,
            "rewards": rewards, "costs": costs,
        }.items()})

    T_max = max(len(e["h_values"]) for e in episodes)

    def _pad(arr: np.ndarray) -> np.ndarray:
        out = np.full(T_max, float("nan"))
        out[:len(arr)] = arr
        return out

    def _stack(key: str) -> np.ndarray:
        return np.stack([_pad(e[key]) for e in episodes])

    h_mat = _stack("h_values")
    viol_mat = (h_mat < 0.0).astype(np.float64)
    viol_mat[np.isnan(h_mat)] = float("nan")

    traces = {
        "h_values": h_mat,
        "violations": viol_mat,
        "cbf_gaps": _stack("cbf_gaps"),
        "slacks": _stack("slacks"),
        "interventions": _stack("interventions"),
        "correction_norm": _stack("correction_norm"),
        "rewards": _stack("rewards"),
        "costs": _stack("costs"),
        "ep_lengths": np.array([len(e["h_values"]) for e in episodes], dtype=np.int32),
    }

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **traces)

    return traces


def summary_from_traces(traces: dict) -> dict:
    """Per-episode scalars for quick comparison tables."""
    h = traces["h_values"]
    v = traces["violations"]
    return {
        "mean_return": float(np.nansum(traces["rewards"], axis=1).mean()),
        "mean_cost": float(np.nansum(traces["costs"], axis=1).mean()),
        "violation_rate": float(np.nanmean(v)),
        "episode_violation_rate": float(np.any(v == 1.0, axis=1).mean()),
        "min_h_mean": float(np.nanmin(h, axis=1).mean()),
        "min_h_std": float(np.nanmin(h, axis=1).std()),
        "mean_h_mean": float(np.nanmean(h, axis=1).mean()),
        "intervention_rate": float(np.nanmean(traces["interventions"])),
        "cbf_gap_mean": float(np.nanmean(traces["cbf_gaps"])),
        "mean_ep_length": float(traces["ep_lengths"].mean()),
    }
