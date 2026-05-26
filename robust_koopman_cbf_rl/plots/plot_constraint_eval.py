"""6-panel constraint evaluation dashboard.

Panels:
  1  h(z) trajectory        — mean ± std band across episodes, all methods
  2  Cumulative violation    — P(h(t') < 0 for any t' ≤ t), mean across episodes
  3  Episode violation rate  — bar chart per method (% episodes with ≥1 violation)
  4  min h(z) distribution  — violin per method (positive = always safe)
  5  CBF gap trajectory      — mean ± std (KCBF methods; NaN-skipped for others)
  6  Intervention rate       — mean ± std over time (KCBF methods only)

Input: traces_dict = {label: traces} where traces is the dict returned by
       eval_constraint_trace() or loaded from an .npz file.

Usage:
    from robust_koopman_cbf_rl.plots.plot_constraint_eval import plot_constraint_eval
    plot_constraint_eval({"KCBF-SAC": traces_kcbf, "SAC": traces_sac}, "out.png")
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_CMAP = plt.get_cmap("tab10")

_KCBF_METRICS = {"cbf_gaps", "interventions"}


def _color(i: int) -> str:
    return _CMAP(i % 10)


def _mean_std(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) along axis=0, ignoring NaN."""
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)


def _has_data(mat: np.ndarray) -> bool:
    """True if at least one non-NaN value exists."""
    return bool(np.any(~np.isnan(mat)))


def plot_constraint_eval(
    traces_dict: dict[str, dict],
    out_path: str,
    h_threshold: float = 0.0,
    smooth: int = 5,
    figsize: tuple = (18, 10),
) -> None:
    """Generate 6-panel constraint evaluation dashboard.

    Args:
        traces_dict: {label: traces} from eval_constraint_trace().
        out_path: Output PNG path.
        h_threshold: Safety boundary value (default 0 — barrier must be ≥ 0).
        smooth: Rolling-average window for trajectory panels.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    ax_h, ax_cumviol, ax_vbar = axes[0]
    ax_violin, ax_gap, ax_int = axes[1]

    labels = list(traces_dict.keys())
    all_traces = list(traces_dict.values())

    # ── Panel 1: h(z) trajectory ─────────────────────────────────────────
    ax_h.axhline(h_threshold, color="red", linestyle="--", linewidth=1.2,
                 alpha=0.7, label="h = 0 (safety boundary)")
    for i, (lbl, tr) in enumerate(traces_dict.items()):
        h = tr["h_values"]
        if not _has_data(h):
            continue
        mu, sigma = _mean_std(h)
        steps = np.arange(len(mu))
        w = min(smooth, len(mu))
        mu_s = np.convolve(mu, np.ones(w) / w, mode="same")
        sig_s = np.convolve(sigma, np.ones(w) / w, mode="same")
        c = _color(i)
        ax_h.plot(steps, mu_s, color=c, label=lbl, linewidth=1.8)
        ax_h.fill_between(steps, mu_s - sig_s, mu_s + sig_s, alpha=0.18, color=c)
    ax_h.set_xlabel("Step")
    ax_h.set_ylabel("h(z)")
    ax_h.set_title("Barrier Value h(z) over Episode")
    ax_h.legend(fontsize=8)
    ax_h.grid(axis="y", linestyle=":", alpha=0.4)

    # ── Panel 2: Cumulative violation probability ─────────────────────────
    for i, (lbl, tr) in enumerate(traces_dict.items()):
        v = tr["violations"]
        if not _has_data(v):
            continue
        ep_viol_any = np.zeros(v.shape[1])
        for t in range(v.shape[1]):
            ep_viol_any[t] = np.nanmean(np.nanmax(v[:, :t + 1], axis=1))
        steps = np.arange(v.shape[1])
        ax_cumviol.plot(steps, ep_viol_any, color=_color(i), label=lbl, linewidth=1.8)
    ax_cumviol.set_xlabel("Step")
    ax_cumviol.set_ylabel("P(violated by step t)")
    ax_cumviol.set_title("Cumulative Violation Probability")
    ax_cumviol.set_ylim(-0.02, 1.02)
    ax_cumviol.legend(fontsize=8)
    ax_cumviol.grid(axis="y", linestyle=":", alpha=0.4)

    # ── Panel 3: Episode violation rate bar ──────────────────────────────
    vio_means, vio_stds = [], []
    for tr in all_traces:
        v = tr["violations"]
        ep_any = np.any(v == 1.0, axis=1).astype(float)  # (n_ep,)
        vio_means.append(float(ep_any.mean()) * 100.0)
        vio_stds.append(float(ep_any.std()) * 100.0)
    x = np.arange(len(labels))
    colors = [_color(i) for i in range(len(labels))]
    ax_vbar.bar(x, vio_means, yerr=vio_stds, capsize=5, color=colors,
                error_kw={"elinewidth": 1.5}, edgecolor="gray", linewidth=0.5)
    ax_vbar.set_xticks(x)
    ax_vbar.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax_vbar.set_ylabel("Episode Violation Rate (%)")
    ax_vbar.set_title("% Episodes with ≥1 Safety Violation")
    ax_vbar.set_ylim(0, max(vio_means) * 1.3 + 1)
    ax_vbar.grid(axis="y", linestyle=":", alpha=0.4)

    # ── Panel 4: min h(z) violin / box ───────────────────────────────────
    min_h_data = []
    valid_labels_violin = []
    valid_colors_violin = []
    for i, (lbl, tr) in enumerate(traces_dict.items()):
        h = tr["h_values"]
        if not _has_data(h):
            min_h_data.append([float("nan")])
        else:
            min_h_data.append(list(np.nanmin(h, axis=1)))
        valid_labels_violin.append(lbl)
        valid_colors_violin.append(_color(i))

    parts = ax_violin.violinplot(min_h_data, positions=list(range(len(valid_labels_violin))),
                                  showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], valid_colors_violin):
        pc.set_facecolor(col)
        pc.set_alpha(0.5)
    ax_violin.axhline(h_threshold, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_violin.set_xticks(range(len(valid_labels_violin)))
    ax_violin.set_xticklabels(valid_labels_violin, rotation=30, ha="right", fontsize=9)
    ax_violin.set_ylabel("min h(z)")
    ax_violin.set_title("Min Barrier Value per Episode\n(above dashed line = safe)")
    ax_violin.grid(axis="y", linestyle=":", alpha=0.4)

    # ── Panel 5: CBF gap trajectory ───────────────────────────────────────
    any_gap_plotted = False
    for i, (lbl, tr) in enumerate(traces_dict.items()):
        g = tr["cbf_gaps"]
        if not _has_data(g):
            continue
        mu, sigma = _mean_std(g)
        steps = np.arange(len(mu))
        w = min(smooth, len(mu))
        mu_s = np.convolve(mu, np.ones(w) / w, mode="same")
        sig_s = np.convolve(sigma, np.ones(w) / w, mode="same")
        c = _color(i)
        ax_gap.plot(steps, mu_s, color=c, label=lbl, linewidth=1.8)
        ax_gap.fill_between(steps, mu_s - sig_s, mu_s + sig_s, alpha=0.18, color=c)
        any_gap_plotted = True
    ax_gap.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_gap.set_xlabel("Step")
    ax_gap.set_ylabel("CBF Gap = a^T u + ξ − b")
    ax_gap.set_title("CBF Gap over Episode\n(gap > 0: constraint satisfied)")
    if any_gap_plotted:
        ax_gap.legend(fontsize=8)
    else:
        ax_gap.text(0.5, 0.5, "No KCBF data", transform=ax_gap.transAxes,
                    ha="center", va="center", color="gray")
    ax_gap.grid(axis="y", linestyle=":", alpha=0.4)

    # ── Panel 6: Intervention rate over time ──────────────────────────────
    any_int_plotted = False
    for i, (lbl, tr) in enumerate(traces_dict.items()):
        iv = tr["interventions"]
        if not _has_data(iv):
            continue
        mu, sigma = _mean_std(iv)
        steps = np.arange(len(mu))
        w = min(smooth, len(mu))
        mu_s = np.convolve(mu, np.ones(w) / w, mode="same")
        sig_s = np.convolve(sigma, np.ones(w) / w, mode="same")
        c = _color(i)
        ax_int.plot(steps, mu_s, color=c, label=lbl, linewidth=1.8)
        ax_int.fill_between(steps, np.clip(mu_s - sig_s, 0, 1),
                             np.clip(mu_s + sig_s, 0, 1), alpha=0.18, color=c)
        any_int_plotted = True
    ax_int.set_xlabel("Step")
    ax_int.set_ylabel("Fraction of Steps Corrected")
    ax_int.set_title("QP Intervention Rate over Episode")
    ax_int.set_ylim(-0.02, 1.02)
    if any_int_plotted:
        ax_int.legend(fontsize=8)
    else:
        ax_int.text(0.5, 0.5, "No KCBF data", transform=ax_int.transAxes,
                    ha="center", va="center", color="gray")
    ax_int.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("Constraint Evaluation Dashboard", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Constraint eval plot → {out_path}")
