"""Map a barrier config dict to a SafetyConstraint instance."""
from __future__ import annotations
from .barrier_base import SafetyConstraint
from .cartpole_barriers import CartPolePositionBarrier, CartPoleAngleBarrier
from .quadrotor_barriers import (
    Quadrotor2DAltitudeBarrier, Quadrotor2DCompositeAltitudeBarrier,
    Quadrotor2DPitchBarrier, Quadrotor3DPositionBarrier,
)
from .velocity_barriers import VelocityNormBarrier

_REGISTRY = {
    "CartPolePositionBarrier": CartPolePositionBarrier,
    "CartPoleAngleBarrier": CartPoleAngleBarrier,
    "Quadrotor2DAltitudeBarrier": Quadrotor2DAltitudeBarrier,
    "Quadrotor2DCompositeAltitudeBarrier": Quadrotor2DCompositeAltitudeBarrier,
    "Quadrotor2DPitchBarrier": Quadrotor2DPitchBarrier,
    "Quadrotor3DPositionBarrier": Quadrotor3DPositionBarrier,
    "VelocityNormBarrier": VelocityNormBarrier,
}


def make_barrier(cfg: dict) -> SafetyConstraint:
    if not cfg or "class" not in cfg:
        raise ValueError("barrier config must contain a 'class' key naming the barrier")
    cls_name = cfg["class"]
    if cls_name not in _REGISTRY:
        raise ValueError(
            f"Unknown barrier class '{cls_name}'. Registered: {sorted(_REGISTRY)}"
        )
    kwargs = {k: v for k, v in cfg.items() if k != "class"}
    return _REGISTRY[cls_name](**kwargs)
