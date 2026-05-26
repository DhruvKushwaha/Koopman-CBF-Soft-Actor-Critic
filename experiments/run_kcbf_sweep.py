"""Multi-seed KCBF-SAC and KCBF-PPO sweep across all 4 environments.

Step 1 (once per env): Train the Koopman model and compute residuals.
Step 2 (per seed):     Run KCBF-SAC and KCBF-PPO using the shared Koopman model.

Koopman models are shared across seeds (physics doesn't change with seed).
KCBF-SAC/PPO policy training is seeded for reproducibility.

Output layout:
    {model_dir}/{env_name}/koopman.npz           — Koopman model
    {model_dir}/{env_name}/koopman_residuals.npz — residual quantiles
    {log_root}/{env_name}/kcbf_sac/seed_{s}/     — KCBF-SAC training logs
    {log_root}/{env_name}/kcbf_ppo/seed_{s}/     — KCBF-PPO training logs

Usage:
    # Full sweep: 4 envs × 2 KCBF variants × 3 seeds
    python experiments/run_kcbf_sweep.py \\
      --model_dir logs/models \\
      --log_root  logs/kcbf_sweep

    # CartPole only, skip Koopman training if models already exist
    python experiments/run_kcbf_sweep.py \\
      --model_dir logs/models \\
      --log_root  logs/kcbf_sweep \\
      --envs cartpole_stab cartpole_track \\
      --skip_koopman

    # Single env+seed for debugging
    python experiments/run_kcbf_sweep.py \\
      --model_dir logs/models \\
      --log_root  logs/debug \\
      --envs cartpole_stab --seeds 0 --variants kcbf_sac
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_CFG = _ROOT / "robust_koopman_cbf_rl" / "configs"

ENVS: dict[str, dict] = {
    "cartpole_stab": {
        "env_cfg":     str(_CFG / "env_cartpole_stab.yaml"),
        "sac_cfg":     str(_CFG / "sac_kcbf_cartpole.yaml"),
        "ppo_cfg":     str(_CFG / "ppo_kcbf_cartpole.yaml"),
        "koopman_cfg": str(_CFG / "koopman.yaml"),
    },
    "cartpole_track": {
        "env_cfg":     str(_CFG / "env_cartpole_track.yaml"),
        "sac_cfg":     str(_CFG / "sac_kcbf_cartpole.yaml"),
        "ppo_cfg":     str(_CFG / "ppo_kcbf_cartpole.yaml"),
        "koopman_cfg": str(_CFG / "koopman.yaml"),
    },
    "safety_halfcheetah": {
        "env_cfg":     str(_CFG / "env_safety_halfcheetah.yaml"),
        "sac_cfg":     str(_CFG / "sac_kcbf_safety_gym.yaml"),
        "ppo_cfg":     str(_CFG / "ppo_kcbf_safety_gym.yaml"),
        "koopman_cfg": str(_CFG / "koopman.yaml"),
    },
    "safety_walker": {
        "env_cfg":     str(_CFG / "env_safety_walker.yaml"),
        "sac_cfg":     str(_CFG / "sac_kcbf_safety_gym.yaml"),
        "ppo_cfg":     str(_CFG / "ppo_kcbf_safety_gym.yaml"),
        "koopman_cfg": str(_CFG / "koopman.yaml"),
    },
    "quadrotor2d_stab": {
        "env_cfg":     str(_CFG / "env_quadrotor2d_stab.yaml"),
        "sac_cfg":     str(_CFG / "sac_kcbf_quadrotor.yaml"),
        "ppo_cfg":     str(_CFG / "ppo_kcbf_quadrotor.yaml"),
        "koopman_cfg": str(_CFG / "koopman_quadrotor.yaml"),
    },
    "quadrotor2d_track": {
        "env_cfg":     str(_CFG / "env_quadrotor2d_track.yaml"),
        "sac_cfg":     str(_CFG / "sac_kcbf_quadrotor.yaml"),
        "ppo_cfg":     str(_CFG / "ppo_kcbf_quadrotor.yaml"),
        "koopman_cfg": str(_CFG / "koopman_quadrotor.yaml"),
    },
}

VARIANTS: dict[str, dict] = {
    "kcbf_sac": {
        "module":   "robust_koopman_cbf_rl.train.train_sac_kcbf",
        "cfg_flag": "--sac_cfg",
        "cfg_key":  "sac_cfg",
    },
    "kcbf_ppo": {
        "module":   "robust_koopman_cbf_rl.train.train_ppo_kcbf",
        "cfg_flag": "--ppo_cfg",
        "cfg_key":  "ppo_cfg",
    },
}


def _train_koopman(env_name: str, model_dir: Path, python: str) -> tuple[str, str]:
    """Train and save Koopman model for env_name. Skip if already exists."""
    env = ENVS[env_name]
    out_path = model_dir / env_name / "koopman.npz"
    if out_path.exists():
        print(f"[SKIP] Koopman {env_name} (already at {out_path})")
        return f"koopman/{env_name}", "skipped"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = out_path.parent / "koopman_train.log"
    cmd = [python, "-m", "robust_koopman_cbf_rl.train.train_koopman",
           "--env_cfg", env["env_cfg"],
           "--koopman_cfg", env["koopman_cfg"],
           "--out", str(out_path)]
    print(f"[KOOPMAN] {env_name} → {out_path}")
    with open(log_file, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                            text=True, cwd=str(_ROOT)).returncode
    if rc == 0:
        print(f"[DONE] Koopman {env_name}")
        return f"koopman/{env_name}", "ok"
    print(f"[FAIL] Koopman {env_name}  (rc={rc}, see {log_file})")
    return f"koopman/{env_name}", f"failed(rc={rc})"


def _run_kcbf_job(args: tuple) -> tuple[str, str]:
    """Run one (env, variant, seed) KCBF training job."""
    env_name, variant_name, seed, model_path, log_dir, python, total_steps = args
    label = f"{env_name}/{variant_name}/seed_{seed}"
    done = Path(log_dir) / "eval_final.csv"
    if done.exists():
        print(f"[SKIP] {label}")
        return label, "skipped"
    if not Path(model_path).exists():
        print(f"[SKIP] {label} — Koopman model missing: {model_path}")
        return label, "no_model"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    env = ENVS[env_name]
    var = VARIANTS[variant_name]
    alg_cfg = env[var["cfg_key"]]
    cmd = [python, "-m", var["module"],
           "--env_cfg", env["env_cfg"],
           var["cfg_flag"], alg_cfg,
           "--model", model_path,
           "--log_dir", str(log_dir),
           "--seed", str(seed)]
    if total_steps is not None:
        cmd += ["--total_steps", str(total_steps)]
    log_file = Path(log_dir) / "stdout.log"
    print(f"[RUN]  {label}")
    with open(log_file, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                            text=True, cwd=str(_ROOT)).returncode
    if rc == 0:
        print(f"[DONE] {label}")
        return label, "ok"
    print(f"[FAIL] {label}  (rc={rc}, see {log_file})")
    return label, f"failed(rc={rc})"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-seed KCBF-SAC + KCBF-PPO sweep across all environments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model_dir", required=True,
                    help="Directory for Koopman models (one sub-folder per env)")
    ap.add_argument("--log_root", required=True,
                    help="Root directory for KCBF training logs")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--envs", nargs="+", default=list(ENVS), choices=list(ENVS))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    choices=list(VARIANTS))
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel workers for KCBF training jobs (default: 1)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip_koopman", action="store_true",
                    help="Skip Koopman training (requires models to already exist)")
    ap.add_argument("--total_steps", type=int, default=None,
                    help="Override total training steps for all jobs (default: use config value)")
    a = ap.parse_args()

    model_dir = Path(a.model_dir)
    log_root = Path(a.log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Train Koopman models ───────────────────────────────────────
    if not a.skip_koopman:
        print("=== Step 1: Koopman model training ===")
        for env_name in a.envs:
            _train_koopman(env_name, model_dir, a.python)
    else:
        print("=== Step 1: Koopman training skipped (--skip_koopman) ===")

    # ── Step 2: KCBF training jobs ─────────────────────────────────────────
    print(f"\n=== Step 2: KCBF training "
          f"({len(a.envs)} envs × {len(a.variants)} variants × {len(a.seeds)} seeds) ===")

    jobs = []
    for env_name in a.envs:
        model_path = str(model_dir / env_name / "koopman.npz")
        for variant_name in a.variants:
            for seed in a.seeds:
                log_dir = log_root / env_name / variant_name / f"seed_{seed}"
                jobs.append((env_name, variant_name, seed, model_path,
                              log_dir, a.python, a.total_steps))

    results: dict[str, str] = {}
    if a.workers == 1:
        for job in jobs:
            label, status = _run_kcbf_job(job)
            results[label] = status
    else:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = {pool.submit(_run_kcbf_job, job): job for job in jobs}
            for fut in as_completed(futures):
                label, status = fut.result()
                results[label] = status

    ok = sum(1 for s in results.values() if s == "ok")
    skipped = sum(1 for s in results.values() if s == "skipped")
    failed = [lbl for lbl, s in results.items() if s.startswith("failed")]
    no_model = sum(1 for s in results.values() if s == "no_model")

    print(f"\n=== KCBF sweep: {ok} done, {skipped} skipped, "
          f"{len(failed)} failed, {no_model} missing model ===")
    if failed:
        print("Failed jobs:")
        for lbl in failed:
            print(f"  {lbl}: {results[lbl]}")

    print("\nKCBF sweep complete.")
    print(f"Model dir: {model_dir}")
    print(f"Log root:  {log_root}")


if __name__ == "__main__":
    main()
