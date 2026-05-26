import numpy as np
import torch
from unittest.mock import MagicMock
import pytest


def _make_mock_qp(u_min, u_max):
    qp = MagicMock()
    qp.u_min = np.array(u_min, dtype=np.float32)
    qp.u_max = np.array(u_max, dtype=np.float32)
    qp.project.return_value = (
        np.zeros(len(u_min), dtype=np.float32),
        {"h_value": 1.0, "cbf_gap": 0.1, "slack": 0.0, "intervention": False},
    )
    return qp


def _make_mock_koopman(z_dim=5, dim_y=3):
    km = MagicMock()
    km.z_dim = z_dim
    km.observables.dim_y = dim_y
    km.lift.return_value = np.zeros(z_dim)
    return km


def test_sac_checkpoint_roundtrip(tmp_path):
    from robust_koopman_cbf_rl.agents.sac import KCBFSACAgent

    qp = _make_mock_qp([-1.0], [1.0])
    km = _make_mock_koopman()
    agent = KCBFSACAgent(dim_obs=3, dim_action=1, dim_z=5,
                         koopman_model=km, qp_filter=qp)

    ckpt = tmp_path / "sac.pt"
    agent.save(ckpt)
    assert ckpt.exists()

    loaded = KCBFSACAgent.load(ckpt, koopman_model=km, qp_filter=qp)
    for p1, p2 in zip(agent.actor.parameters(), loaded.actor.parameters()):
        assert torch.allclose(p1, p2)
    for p1, p2 in zip(agent.q1.parameters(), loaded.q1.parameters()):
        assert torch.allclose(p1, p2)
    for p1, p2 in zip(agent.q2.parameters(), loaded.q2.parameters()):
        assert torch.allclose(p1, p2)
    for p1, p2 in zip(agent.q1_t.parameters(), loaded.q1_t.parameters()):
        assert torch.allclose(p1, p2)
    for p1, p2 in zip(agent.q2_t.parameters(), loaded.q2_t.parameters()):
        assert torch.allclose(p1, p2)
    assert loaded.dim_obs == 3
    assert loaded.dim_action == 1


def test_ppo_checkpoint_roundtrip(tmp_path):
    from robust_koopman_cbf_rl.agents.ppo import KCBFPPOAgent

    qp = _make_mock_qp([-1.0], [1.0])
    km = _make_mock_koopman()
    agent = KCBFPPOAgent(dim_obs=3, dim_action=1, dim_z=5,
                         koopman_model=km, qp_filter=qp)

    ckpt = tmp_path / "ppo.pt"
    agent.save(ckpt)
    assert ckpt.exists()

    loaded = KCBFPPOAgent.load(ckpt, koopman_model=km, qp_filter=qp)
    for p1, p2 in zip(agent.ac.parameters(), loaded.ac.parameters()):
        assert torch.allclose(p1, p2)
    assert loaded.dim_obs == 3
    assert loaded.dim_action == 1
    assert loaded.dim_z == 5


def test_eval_runner_imports():
    # smoke test: import must not raise
    from robust_koopman_cbf_rl.eval import run_eval  # noqa: F401
