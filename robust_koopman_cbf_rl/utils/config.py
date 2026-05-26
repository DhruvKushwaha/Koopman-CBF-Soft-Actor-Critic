"""YAML -> dataclass config loader."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml


@dataclass
class KoopmanCfg:
    n_rbf: int = 64
    bandwidth: float = 1.0
    ridge: float = 1e-6
    alpha: float = 0.95
    collection_steps: int = 50_000
    seed: int = 0
    margin_mode: str = "global"
    n_clusters: int = 8

@dataclass
class SACCfg:
    total_steps: int = 200_000
    warmup_steps: int = 1_000
    batch_size: int = 256
    buffer_capacity: int = 1_000_000
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4        # fallback used when actor_lr/critic_lr are None
    actor_lr: float = None  # None → falls back to lr
    critic_lr: float = None # None → falls back to lr
    alpha: float = 0.2
    lam_h: float = 1.0
    eta: float = 0.5
    lam_slack: float = 1e3

@dataclass
class PPOCfg:
    total_steps: int = 1_000_000
    rollout_len: int = 2048
    epochs: int = 10
    minibatch: int = 64
    lr: float = 3e-4        # fallback used when actor_lr/critic_lr are None
    actor_lr: float = None  # None → falls back to lr
    critic_lr: float = None # None → falls back to lr
    clip: float = 0.2
    c_v: float = 0.5
    c_e: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    lam_h: float = 1.0
    eta: float = 0.5
    lam_slack: float = 1e3

@dataclass
class EnvCfg:
    kind: str = "safe_control_gym"   # or "safety_gymnasium"
    env_id: str = "cartpole"
    task_config: dict = field(default_factory=dict)
    velocity_limit: float = 2.0
    seed: int = 0
    barrier: dict = field(default_factory=dict)


def load_yaml(path) -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge(cfg_cls, *dicts):
    import dataclasses
    import warnings
    merged = {}
    for d in dicts:
        merged.update(d or {})
    known = {f.name for f in dataclasses.fields(cfg_cls)}
    unknown = set(merged) - known
    if unknown:
        warnings.warn(f"merge: ignoring unknown keys for {cfg_cls.__name__}: {unknown}")
    return cfg_cls(**{k: v for k, v in merged.items() if k in known})
