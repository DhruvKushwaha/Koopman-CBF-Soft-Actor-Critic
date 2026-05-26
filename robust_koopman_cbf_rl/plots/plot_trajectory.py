"""Collect a single rollout trajectory and plot state dims vs reference."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect_trajectory(env, agent, model, qp_filter, seed: int = 42) -> dict:
    """Run one episode and return states, actions, h_values arrays."""
    obs, info = env.reset(seed=seed)
    y = info.get("raw_state", obs)
    z = model.lift(y[:model.observables.dim_y])
    states, actions, h_vals = [], [], []
    done = False
    while not done:
        states.append(y.copy())
        if hasattr(agent, "select_action"):
            result = agent.select_action(obs)
            u_safe, diag = result[0], result[-1]
        else:
            u_safe, diag = qp_filter.project(z, np.zeros(env.action_space.shape))
        actions.append(u_safe.copy())
        h_vals.append(float(diag["h_value"]))
        obs, _, term, trunc, info = env.step(u_safe)
        y = info.get("raw_state", obs)
        z = model.lift(y[:model.observables.dim_y])
        done = bool(term) or bool(trunc)
    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "h_values": np.array(h_vals),
    }


def plot_trajectory(traj: dict, state_names=None, ref_traj=None,
                    out_path: str = "trajectory.pdf"):
    """Plot each state dimension over time plus the CBF value h(z).

    Args:
        traj: dict from collect_trajectory with keys 'states', 'actions', 'h_values'.
        state_names: list of str labels for state dimensions (optional).
        ref_traj: (T, dim_y) reference trajectory to overlay (optional, for tracking tasks).
        out_path: output file path.
    """
    states = np.asarray(traj["states"])    # (T, dim_y)
    h_values = np.asarray(traj["h_values"])  # (T,)
    T, dim_y = states.shape
    n_panels = dim_y + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2 * n_panels), sharex=True)
    t = np.arange(T)
    for i in range(dim_y):
        ax = axes[i]
        label = state_names[i] if state_names else f"state[{i}]"
        ax.plot(t, states[:, i], label="agent")
        if ref_traj is not None and i < ref_traj.shape[1]:
            ax.plot(t[:ref_traj.shape[0]], ref_traj[:, i], "--", label="ref", alpha=0.7)
        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=7)
    ax_h = axes[-1]
    ax_h.plot(t, h_values)
    ax_h.axhline(0, color="r", linestyle="--", linewidth=0.8)
    ax_h.set_ylabel("h(z)")
    ax_h.set_xlabel("step")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
