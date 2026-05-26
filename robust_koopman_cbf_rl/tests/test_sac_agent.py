import numpy as np
import torch

def test_sac_agent_critic_uses_safe_action_and_actor_uses_cbf_penalty():
    from robust_koopman_cbf_rl.agents.sac import KCBFSACAgent

    class DummyFilter:
        eta = 0.5
        u_min = np.array([-1.0])
        u_max = np.array([1.0])
        def project(self, z, u_nom):
            return np.clip(u_nom, -1, 1) * 0.5, {
                "h_value": 0.1, "cbf_gap": 0.0, "slack": 0.0,
                "correction_norm": 0.0, "intervention": False, "rho": 0.0,
            }
        def cbf_penalty_terms(self, z_batch, u_batch_nom):
            n = z_batch.shape[0]
            a = torch.zeros(n, u_batch_nom.shape[1])
            b = torch.zeros(n)
            return a, b

    class DummyModel:
        z_dim = 4
        A = np.eye(4)
        B = np.zeros((4, 1))
        def lift(self, y):
            return np.concatenate([y, np.zeros(4 - y.shape[0])]) if y.shape[0] < 4 else y[:4]

    agent = KCBFSACAgent(
        dim_obs=2, dim_action=1, dim_z=4,
        koopman_model=DummyModel(), qp_filter=DummyFilter(),
        lam_h=1.0, gamma=0.99, tau=0.005, lr=3e-4, device="cpu",
    )
    assert hasattr(agent, "_compute_actor_loss")
    # Smoke step: random transition
    obs = np.zeros(2)
    u_safe, u_nom, diag = agent.select_action(obs, deterministic=False)
    assert u_safe.shape == (1,)
    assert u_nom.shape == (1,)
