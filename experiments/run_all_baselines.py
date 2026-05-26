"""Orchestration script: run all baselines sequentially, then generate comparison plots.

Usage:
    python -m experiments.run_all_baselines \\
      --env_cfg  robust_koopman_cbf_rl/configs/env_cartpole_stab.yaml \\
      --log_root logs/cartpole_stab \\
      --total_steps 200000
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import warnings

from robust_koopman_cbf_rl.utils.config import load_yaml, merge, EnvCfg, SACCfg, PPOCfg
from robust_koopman_cbf_rl.train.train_koopman import build_env


def _run_sac(env_cfg_path, sac_cfg_path, log_dir, total_steps):
    from robust_koopman_cbf_rl.train.train_sac import main
    main(env_cfg_path, sac_cfg_path, str(log_dir), total_steps=total_steps)


def _run_ppo(env_cfg_path, ppo_cfg_path, log_dir, total_steps):
    from robust_koopman_cbf_rl.train.train_ppo import main
    main(env_cfg_path, ppo_cfg_path, str(log_dir), total_steps=total_steps)


def _run_sac_penalty(env_cfg_path, sac_cfg_path, log_dir, total_steps):
    from robust_koopman_cbf_rl.train.train_sac_penalty import main
    main(env_cfg_path, sac_cfg_path, str(log_dir), total_steps=total_steps)


def _run_sac_lagrangian(env_cfg_path, sac_cfg_path, log_dir, total_steps):
    from robust_koopman_cbf_rl.train.train_sac_lagrangian import main
    main(env_cfg_path, sac_cfg_path, str(log_dir), total_steps=total_steps)


def _eval_lqr(env_cfg, log_dir):
    from robust_koopman_cbf_rl.baselines.lqr import LQRBaseline
    from robust_koopman_cbf_rl.train.evaluate_baseline import evaluate_baseline
    env = build_env(env_cfg)
    try:
        lqr = LQRBaseline(env)
        results = evaluate_baseline(env, lqr, n_episodes=10)
        log_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(log_dir / "eval_final.csv", index=False)
        print(f"    LQR eval saved to {log_dir / 'eval_final.csv'}")
    except Exception as exc:
        print(f"    LQR eval failed: {exc}")
    finally:
        env.close()


def _eval_pid(env_cfg, log_dir):
    from robust_koopman_cbf_rl.baselines.pid import PIDBaseline
    from robust_koopman_cbf_rl.train.evaluate_baseline import evaluate_baseline
    env = build_env(env_cfg)
    try:
        pid = PIDBaseline(env)
        results = evaluate_baseline(env, pid, n_episodes=10)
        log_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(log_dir / "eval_final.csv", index=False)
        print(f"    PID eval saved to {log_dir / 'eval_final.csv'}")
    except Exception as exc:
        print(f"    PID eval failed: {exc}")
    finally:
        env.close()


def main(env_cfg_path, log_root, total_steps=None, sac_cfg_path=None, ppo_cfg_path=None):
    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    env_cfg = merge(EnvCfg, load_yaml(env_cfg_path))

    # Resolve default cfg paths if not provided
    _pkg = Path(__file__).parent.parent / "robust_koopman_cbf_rl" / "configs"
    if sac_cfg_path is None:
        candidates = list(_pkg.glob("*sac*.yaml"))
        sac_cfg_path = str(candidates[0]) if candidates else env_cfg_path
    if ppo_cfg_path is None:
        candidates = list(_pkg.glob("*ppo*.yaml"))
        ppo_cfg_path = str(candidates[0]) if candidates else env_cfg_path

    baselines = [
        ("[1/6] Running SAC baseline...",       "sac_baseline",    _run_sac,           (env_cfg_path, sac_cfg_path)),
        ("[2/6] Running PPO baseline...",        "ppo_baseline",    _run_ppo,           (env_cfg_path, ppo_cfg_path)),
        ("[3/6] Running SAC+Penalty baseline...", "sac_penalty",    _run_sac_penalty,   (env_cfg_path, sac_cfg_path)),
        ("[4/6] Running SAC+Lagrangian baseline...", "sac_lagrangian", _run_sac_lagrangian, (env_cfg_path, sac_cfg_path)),
    ]

    for label, subdir, fn, args in baselines:
        print(label)
        log_dir = log_root / subdir
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            fn(*args, log_dir, total_steps)
        except Exception as exc:
            warnings.warn(f"Baseline {subdir} failed: {exc}")
            print(f"    FAILED: {exc}")

    print("[5/6] Running LQR eval (safe_control_gym only)...")
    print("Physical CBF baseline skipped: requires env-specific dynamics callable.")
    if env_cfg.kind == "safe_control_gym":
        try:
            _eval_lqr(env_cfg, log_root / "lqr")
        except Exception as exc:
            warnings.warn(f"LQR eval failed: {exc}")
            print(f"    FAILED: {exc}")
    else:
        print("    Skipped (env.kind != safe_control_gym).")

    print("[6/6] Running PID eval (safe_control_gym only)...")
    if env_cfg.kind == "safe_control_gym":
        try:
            _eval_pid(env_cfg, log_root / "pid")
        except Exception as exc:
            warnings.warn(f"PID eval failed: {exc}")
            print(f"    FAILED: {exc}")
    else:
        print("    Skipped (env.kind != safe_control_gym).")

    # Gather all available eval CSVs for comparison
    csv_paths = []
    labels = []
    _label_map = {
        "sac_baseline": "SAC",
        "ppo_baseline": "PPO",
        "sac_penalty": "SAC+Penalty",
        "sac_lagrangian": "SAC+Lagrangian",
        "lqr": "LQR",
        "pid": "PID",
        "sac_run1": "KCBF-SAC",
        "ppo_run1": "KCBF-PPO",
    }
    for subdir, label in _label_map.items():
        p = log_root / subdir / "eval_final.csv"
        if p.exists():
            csv_paths.append(str(p))
            labels.append(label)

    if len(csv_paths) >= 2:
        print(f"\nGenerating comparison plot from {len(csv_paths)} baselines...")
        out_plot = str(log_root / "comparison.png")
        try:
            from robust_koopman_cbf_rl.eval.compare_models import compare_models
            compare_models(csv_paths, labels, out_plot)
            print(f"Comparison saved → {out_plot}")
        except Exception as exc:
            warnings.warn(f"compare_models failed: {exc}")

        # Training curves
        train_csvs = []
        train_labels = []
        _train_map = {
            "sac_baseline": ("sac_baseline.csv", "SAC"),
            "ppo_baseline": ("ppo_baseline.csv", "PPO"),
            "sac_penalty": ("sac_penalty.csv", "SAC+Penalty"),
            "sac_lagrangian": ("sac_lagrangian.csv", "SAC+Lagrangian"),
            "sac_run1": ("sac_kcbf.csv", "KCBF-SAC"),
            "ppo_run1": ("ppo_kcbf.csv", "KCBF-PPO"),
        }
        for subdir, (csv_name, lbl) in _train_map.items():
            p = log_root / subdir / csv_name
            if p.exists():
                train_csvs.append(str(p))
                train_labels.append(lbl)

        if len(train_csvs) >= 1:
            print(f"Generating training curves from {len(train_csvs)} logs...")
            out_curves = str(log_root / "training_curves.png")
            try:
                from robust_koopman_cbf_rl.plots.plot_training import plot_training
                plot_training(train_csvs, train_labels, out_curves)
                print(f"Training curves saved → {out_curves}")
            except Exception as exc:
                warnings.warn(f"plot_training failed: {exc}")
    else:
        print("Not enough eval CSVs for comparison plot.")

    print("\nAll baselines complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run all baselines and generate comparison plots.")
    ap.add_argument("--env_cfg", required=True, help="Path to env config YAML")
    ap.add_argument("--log_root", required=True, help="Root directory for all logs")
    ap.add_argument("--total_steps", type=int, default=None, help="Override total training steps")
    ap.add_argument("--sac_cfg", default=None, help="Path to SAC config YAML")
    ap.add_argument("--ppo_cfg", default=None, help="Path to PPO config YAML")
    a = ap.parse_args()
    main(a.env_cfg, a.log_root, a.total_steps, a.sac_cfg, a.ppo_cfg)
