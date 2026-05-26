import numpy as np
import torch

def test_ppo_loss_includes_cbf_penalty():
    from robust_koopman_cbf_rl.agents.ppo import KCBFPPOAgent

    class DummyFilter:
        eta = 0.5
        def project(self, z, u_nom):
            return np.clip(u_nom, -1, 1), {
                "h_value": 0.0, "cbf_gap": 0.0, "slack": 0.0,
                "correction_norm": 0.0, "intervention": False, "rho": 0.0,
            }
        def cbf_penalty_terms(self, z, u):
            return torch.zeros(z.shape[0], u.shape[1]), torch.zeros(z.shape[0])

    class DummyModel:
        z_dim = 4
        A = np.eye(4); B = np.zeros((4, 1))
        class observables: dim_y = 2
        observables = observables()
        def lift(self, y): return np.concatenate([y, np.zeros(2)])

    agent = KCBFPPOAgent(dim_obs=2, dim_action=1, dim_z=4,
                        koopman_model=DummyModel(), qp_filter=DummyFilter(),
                        lam_h=1.0, gamma=0.99, lam=0.95, lr=3e-4,
                        clip=0.2, c_v=0.5, c_e=0.01, device="cpu")
    u_safe, u_nom, logp, val, diag = agent.select_action(np.zeros(2))
    assert u_safe.shape == (1,)
    assert isinstance(logp, float)


def test_ppo_update_runs_and_returns_finite_stats():
    import numpy as np
    import torch
    from robust_koopman_cbf_rl.agents.ppo import KCBFPPOAgent
    from robust_koopman_cbf_rl.agents.rollout_buffer import KCBFRolloutBuffer

    class DummyFilter:
        eta = 0.5
        def project(self, z, u_nom):
            return np.clip(u_nom, -1, 1), {
                "h_value": 0.0, "cbf_gap": 0.0, "slack": 0.0,
                "correction_norm": 0.0, "intervention": False, "rho": 0.0,
            }
        def cbf_penalty_terms(self, z, u):
            return torch.zeros(z.shape[0], u.shape[1]), torch.ones(z.shape[0])

    class DummyModel:
        z_dim = 4
        A = np.eye(4); B = np.zeros((4, 1))
        class observables: dim_y = 2
        observables = observables()
        def lift(self, y): return np.concatenate([y, np.zeros(2)])

    agent = KCBFPPOAgent(dim_obs=2, dim_action=1, dim_z=4,
                         koopman_model=DummyModel(), qp_filter=DummyFilter(),
                         lam_h=1.0, gamma=0.99, lam=0.95, lr=3e-4,
                         clip=0.2, c_v=0.5, c_e=0.01, device="cpu")

    buf = KCBFRolloutBuffer(rollout_len=8, dim_obs=2, dim_action=1, dim_z=4)
    for _ in range(8):
        buf.add(obs=np.zeros(2), z=np.zeros(4),
                action_nom=np.array([0.1]), action_safe=np.array([0.05]),
                logprob_nom=-0.3, value=0.5,
                reward=1.0, cost=0.0, done=False,
                h_value=0.2, cbf_gap=0.1, slack=0.0, intervention=False)
    buf.compute_advantages(last_value=0.0, gamma=0.99, lam=0.95)
    batch = buf.batched()

    stats = agent.update(batch, epochs=2, minibatch=4)
    assert "loss" in stats
    assert "cbf_pen" in stats
    assert np.isfinite(stats["loss"])
    assert np.isfinite(stats["cbf_pen"])
    assert stats["cbf_pen"] > 0  # b_cbf=1 > 0, so penalty must be positive


def test_ppo_actor_grad_flows_from_cbf_penalty():
    """Actor parameters must receive non-zero gradient from the CBF penalty alone.

    Zero out PPO clip loss (advantages=0), value loss (c_v=0), and entropy bonus (c_e=0).
    Set a_cbf=1, b_cbf=1 so gap = 1 - sum(a_fresh) is mostly positive at init -> penalty fires.
    Snapshot actor params, run one update, assert params moved.
    """
    import numpy as np
    import torch
    from robust_koopman_cbf_rl.agents.ppo import KCBFPPOAgent

    class GradFilter:
        eta = 0.5
        def project(self, z, u_nom):
            return np.clip(u_nom, -1, 1), {
                "h_value": 0.0, "cbf_gap": 0.0, "slack": 0.0,
                "correction_norm": 0.0, "intervention": False, "rho": 0.0,
            }
        def cbf_penalty_terms(self, z, u):
            a = torch.ones(z.shape[0], u.shape[1], dtype=u.dtype, device=u.device)
            b = torch.ones(z.shape[0], dtype=u.dtype, device=u.device)
            return a, b

    class DummyModel:
        z_dim = 4
        A = np.eye(4); B = np.zeros((4, 1))
        class _Obs: dim_y = 2
        observables = _Obs()
        def lift(self, y): return np.concatenate([y, np.zeros(2)])

    agent = KCBFPPOAgent(
        dim_obs=2, dim_action=1, dim_z=4,
        koopman_model=DummyModel(), qp_filter=GradFilter(),
        lam_h=1.0, gamma=0.99, lam=0.95, lr=3e-4,
        clip=0.2, c_v=0.0, c_e=0.0, device="cpu",
    )

    n = 8
    batch = {
        "obs": np.zeros((n, 2), dtype=np.float32),
        "z": np.zeros((n, 4), dtype=np.float32),
        "action_nom": np.zeros((n, 1), dtype=np.float32),
        "logprob_nom": np.zeros(n, dtype=np.float32),
        "advantages": np.zeros(n, dtype=np.float32),
        "returns": np.zeros(n, dtype=np.float32),
    }

    mu_w_before = agent.ac.mu.weight.detach().clone()
    log_std_before = agent.ac.log_std.detach().clone()

    agent.update(batch, epochs=1, minibatch=n)

    # Both the mean head (direct grad via rsample mean) and log_std (grad via rsample std * eps)
    # should have moved purely from the CBF penalty term.
    assert not torch.allclose(agent.ac.mu.weight, mu_w_before), (
        "Actor mu weights did not move -- CBF penalty grad did not reach the actor mean head."
    )
    assert not torch.allclose(agent.ac.log_std, log_std_before), (
        "Actor log_std did not move -- CBF penalty grad did not flow through the std (rsample)."
    )
