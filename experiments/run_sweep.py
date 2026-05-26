"""Multi-environment, multi-seed sweep over all baselines.

Each (env × baseline × seed) triple runs as an isolated subprocess; completed
jobs (eval_final.csv present) are automatically skipped for idempotent restarts.

Output layout:
    {log_root}/{env_name}/{baseline}/seed_{seed}/eval_final.csv
    {log_root}/{env_name}/{baseline}/seed_{seed}/stdout.log
    {log_root}/{env_name}/lqr/eval_final.csv          (safe-control-gym only)
    {log_root}/{env_name}/pid/eval_final.csv          (safe-control-gym only)
    {log_root}/aggregate_summary.csv
    {log_root}/aggregate_comparison.png

Usage:
    # All environments, 3 seeds, sequential
    python experiments/run_sweep.py --log_root logs/sweep

    # CartPole only, 2 seeds, 2 parallel workers
    python experiments/run_sweep.py \\
      --log_root logs/sweep \\
      --envs cartpole_stab cartpole_track \\
      --seeds 0 1 2 \\
      --workers 2

    # Single baseline for quick debugging
    python experiments/run_sweep.py \\
      --log_root logs/debug \\
      --envs cartpole_stab \\
      --baselines sac \\
      --seeds 0
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_CFG = _ROOT / "robust_koopman_cbf_rl" / "configs"

ENVS: dict[str, dict] = {
    "cartpole_stab": {
        "env_cfg": str(_CFG / "env_cartpole_stab.yaml"),
        "sac_cfg": str(_CFG / "sac_cartpole.yaml"),
        "ppo_cfg": str(_CFG / "ppo_cartpole.yaml"),
        "kind": "safe_control_gym",
    },
    "cartpole_track": {
        "env_cfg": str(_CFG / "env_cartpole_track.yaml"),
        "sac_cfg": str(_CFG / "sac_cartpole.yaml"),
        "ppo_cfg": str(_CFG / "ppo_cartpole.yaml"),
        "kind": "safe_control_gym",
    },
    "safety_halfcheetah": {
        "env_cfg": str(_CFG / "env_safety_halfcheetah.yaml"),
        "sac_cfg": str(_CFG / "sac_safety_gym.yaml"),
        "ppo_cfg": str(_CFG / "ppo_safety_gym.yaml"),
        "kind": "safety_gymnasium",
    },
    "safety_walker": {
        "env_cfg": str(_CFG / "env_safety_walker.yaml"),
        "sac_cfg": str(_CFG / "sac_safety_gym.yaml"),
        "ppo_cfg": str(_CFG / "ppo_safety_gym.yaml"),
        "kind": "safety_gymnasium",
    },
    "quadrotor2d_stab": {
        "env_cfg": str(_CFG / "env_quadrotor2d_stab.yaml"),
        "sac_cfg": str(_CFG / "sac_quadrotor.yaml"),
        "ppo_cfg": str(_CFG / "ppo_quadrotor.yaml"),
        "kind": "safe_control_gym",
    },
    "quadrotor2d_track": {
        "env_cfg": str(_CFG / "env_quadrotor2d_track.yaml"),
        "sac_cfg": str(_CFG / "sac_quadrotor.yaml"),
        "ppo_cfg": str(_CFG / "ppo_quadrotor.yaml"),
        "kind": "safe_control_gym",
    },
}

BASELINES: dict[str, dict] = {
    "sac": {
        "module": "robust_koopman_cbf_rl.train.train_sac",
        "cfg_flag": "--sac_cfg",
        "use_ppo": False,
    },
    "ppo": {
        "module": "robust_koopman_cbf_rl.train.train_ppo",
        "cfg_flag": "--ppo_cfg",
        "use_ppo": True,
    },
    "sac_penalty": {
        "module": "robust_koopman_cbf_rl.train.train_sac_penalty",
        "cfg_flag": "--sac_cfg",
        "use_ppo": False,
    },
    "sac_lagrangian": {
        "module": "robust_koopman_cbf_rl.train.train_sac_lagrangian",
        "cfg_flag": "--sac_cfg",
        "use_ppo": False,
    },
}


def _build_cmd(env_name: str, baseline_name: str, log_dir: Path,
               seed: int, python: str, total_steps: int | None = None) -> list[str]:
    env = ENVS[env_name]
    bl = BASELINES[baseline_name]
    alg_cfg = env["ppo_cfg"] if bl["use_ppo"] else env["sac_cfg"]
    cmd = [
        python, "-m", bl["module"],
        "--env_cfg", env["env_cfg"],
        bl["cfg_flag"], alg_cfg,
        "--log_dir", str(log_dir),
        "--seed", str(seed),
    ]
    if total_steps is not None:
        cmd += ["--total_steps", str(total_steps)]
    return cmd


def _run_job(args: tuple) -> tuple[str, str]:
    env_name, baseline_name, seed, log_dir, python, total_steps = args
    label = f"{env_name}/{baseline_name}/seed_{seed}"
    done = Path(log_dir) / "eval_final.csv"
    if done.exists():
        print(f"[SKIP] {label}")
        return label, "skipped"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd(env_name, baseline_name, Path(log_dir), seed, python, total_steps)
    log_file = Path(log_dir) / "stdout.log"
    print(f"[RUN]  {label}")
    with open(log_file, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True,
                            cwd=str(_ROOT)).returncode
    if rc == 0:
        print(f"[DONE] {label}")
        return label, "ok"
    print(f"[FAIL] {label}  (rc={rc}, see {log_file})")
    return label, f"failed(rc={rc})"


def _run_classical_evals(envs: list[str], log_root: Path) -> None:
    """LQR and PID: eval-only, seed-independent, safe-control-gym envs only."""
    import pandas as pd
    from robust_koopman_cbf_rl.utils.config import load_yaml, merge, EnvCfg
    from robust_koopman_cbf_rl.train.train_koopman import build_env
    from robust_koopman_cbf_rl.train.evaluate_baseline import evaluate_baseline

    for env_name in envs:
        if ENVS[env_name]["kind"] != "safe_control_gym":
            continue
        env_cfg = merge(EnvCfg, load_yaml(ENVS[env_name]["env_cfg"]))
        # PID in safe_control_gym is quadrotor-only; skip for cartpole
        is_quadrotor = getattr(env_cfg, "env_id", "") == "quadrotor"
        controllers = [("LQRBaseline", "lqr")]
        if is_quadrotor:
            controllers.append(("PIDBaseline", "pid"))
        for cls_name, bl_name in controllers:
            out = log_root / env_name / bl_name / "eval_final.csv"
            if out.exists():
                print(f"[SKIP] {env_name}/{bl_name}")
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f"[EVAL] {env_name}/{bl_name}")
            try:
                env = build_env(env_cfg)
                if cls_name == "LQRBaseline":
                    from robust_koopman_cbf_rl.baselines.lqr import LQRBaseline
                    agent = LQRBaseline(env)
                else:
                    from robust_koopman_cbf_rl.baselines.pid import PIDBaseline
                    agent = PIDBaseline(env)
                results = evaluate_baseline(env, agent, n_episodes=20)
                pd.DataFrame(results).to_csv(out, index=False)
                print(f"  → {out}")
            except Exception as exc:
                warnings.warn(f"{env_name}/{bl_name} failed: {exc}")
                print(f"  FAILED: {exc}")
            finally:
                try:
                    env.close()
                except Exception:
                    pass


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-env, multi-seed sweep over all baselines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--log_root", required=True, help="Root directory for all run logs")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="Random seeds to run (default: 0 1 2)")
    ap.add_argument("--envs", nargs="+", default=list(ENVS), choices=list(ENVS),
                    help="Environments to include (default: all)")
    ap.add_argument("--baselines", nargs="+", default=list(BASELINES), choices=list(BASELINES),
                    help="Baselines to include (default: all)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel subprocess workers (default: 1; safe to increase for CPU envs)")
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter (default: current interpreter)")
    ap.add_argument("--skip_classical", action="store_true",
                    help="Skip LQR/PID eval (useful when safe-control-gym not installed)")
    ap.add_argument("--skip_aggregate", action="store_true",
                    help="Skip aggregation/plotting after training")
    ap.add_argument("--total_steps", type=int, default=None,
                    help="Override total training steps for all jobs (default: use config value)")
    a = ap.parse_args()

    log_root = Path(a.log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    jobs = [
        (env_name, baseline_name, seed,
         log_root / env_name / baseline_name / f"seed_{seed}", a.python, a.total_steps)
        for env_name in a.envs
        for baseline_name in a.baselines
        for seed in a.seeds
    ]

    n_total = len(jobs)
    print(f"Sweep: {n_total} jobs  ({len(a.envs)} envs × {len(a.baselines)} baselines"
          f" × {len(a.seeds)} seeds), {a.workers} worker(s)")
    print(f"  envs:      {a.envs}")
    print(f"  baselines: {a.baselines}")
    print(f"  seeds:     {a.seeds}")
    print(f"  python:    {a.python}\n")

    results: dict[str, str] = {}
    if a.workers == 1:
        for job in jobs:
            label, status = _run_job(job)
            results[label] = status
    else:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = {pool.submit(_run_job, job): job for job in jobs}
            for fut in as_completed(futures):
                label, status = fut.result()
                results[label] = status

    ok = sum(1 for s in results.values() if s == "ok")
    skipped = sum(1 for s in results.values() if s == "skipped")
    failed = [lbl for lbl, s in results.items() if s.startswith("failed")]
    print(f"\n=== Training sweep: {ok} done, {skipped} skipped, {len(failed)} failed ===")
    if failed:
        print("Failed jobs:")
        for lbl in failed:
            print(f"  {lbl}: {results[lbl]}")

    if not a.skip_classical:
        print("\n--- Classical baselines (LQR / PID) ---")
        _run_classical_evals(a.envs, log_root)

    if not a.skip_aggregate:
        print("\n--- Aggregating results ---")
        try:
            sys.path.insert(0, str(_ROOT))
            from experiments.aggregate_results import aggregate
            aggregate(str(log_root))
        except Exception as exc:
            warnings.warn(f"Aggregation failed: {exc}")
            print(f"  FAILED: {exc}")

    print("\nSweep complete.")


if __name__ == "__main__":
    main()
