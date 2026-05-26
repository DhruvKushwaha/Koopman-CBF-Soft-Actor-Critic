"""KCBF-augmented SAC: critic trained on safe action; actor loss adds CBF penalty."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, dim_obs, dim_action, hidden=(256, 256)):
        super().__init__()
        self.net = _mlp([dim_obs, *hidden])
        self.mu = nn.Linear(hidden[-1], dim_action)
        self.log_std = nn.Linear(hidden[-1], dim_action)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), -5.0, 2.0)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        a = torch.tanh(x)
        logp = dist.log_prob(x) - torch.log1p(-a.pow(2) + 1e-6)
        return a, logp.sum(-1, keepdim=True), mu, std


class QCritic(nn.Module):
    def __init__(self, dim_obs, dim_action, hidden=(256, 256)):
        super().__init__()
        self.q = _mlp([dim_obs + dim_action, *hidden, 1])

    def forward(self, obs, a):
        return self.q(torch.cat([obs, a], dim=-1))


class KCBFSACAgent:
    def __init__(self, dim_obs, dim_action, dim_z,
                 koopman_model, qp_filter, lam_h=1.0,
                 gamma=0.99, tau=0.005, lr=3e-4,
                 actor_lr=None, critic_lr=None,
                 alpha=0.2, target_entropy=None,
                 device="cpu"):
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.dim_z = dim_z
        self.koopman_model = koopman_model
        self.qp_filter = qp_filter
        self.lam_h = float(lam_h)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.device = device

        self.actor = GaussianActor(dim_obs, dim_action).to(device)
        self.q1 = QCritic(dim_obs, dim_action).to(device)
        self.q2 = QCritic(dim_obs, dim_action).to(device)
        self.q1_t = QCritic(dim_obs, dim_action).to(device)
        self.q2_t = QCritic(dim_obs, dim_action).to(device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        _actor_lr = actor_lr if actor_lr is not None else lr
        _critic_lr = critic_lr if critic_lr is not None else lr
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=_actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=_critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=_critic_lr)
        # Precompute action rescaling tensors: maps tanh output (-1,1) → [u_min, u_max].
        self._u_lo = torch.as_tensor(qp_filter.u_min, dtype=torch.float32)
        self._u_hi = torch.as_tensor(qp_filter.u_max, dtype=torch.float32)

    def _rescale(self, a_tanh: np.ndarray) -> np.ndarray:
        """Maps tanh output a ∈ (-1,1) to actual action space [u_min, u_max]."""
        lo, hi = self.qp_filter.u_min, self.qp_filter.u_max
        return lo + 0.5 * (a_tanh + 1.0) * (hi - lo)

    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _, mu, _ = self.actor(obs_t)
        a_raw = (torch.tanh(mu) if deterministic else a).cpu().numpy()[0]
        u_nom = self._rescale(a_raw)
        z = self.koopman_model.lift(
            np.asarray(obs, dtype=np.float64)[:self.koopman_model.observables.dim_y]
            if hasattr(self.koopman_model, "observables")
            else np.asarray(obs, dtype=np.float64)
        )
        u_safe, diag = self.qp_filter.project(z, u_nom)
        return u_safe.astype(np.float32), u_nom.astype(np.float32), diag

    def _compute_critic_loss(self, batch):
        obs = torch.as_tensor(batch["obs"], device=self.device)
        a_safe = torch.as_tensor(batch["action_safe"], device=self.device)
        r = torch.as_tensor(batch["reward"], device=self.device).unsqueeze(-1)
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device)
        next_z_np = batch["next_z"]                                  # (N, z_dim) numpy
        done = torch.as_tensor(batch["done"], device=self.device).unsqueeze(-1)
        with torch.no_grad():
            a_next, logp_next, _, _ = self.actor(next_obs)
            lo = self._u_lo.to(obs.device)
            hi = self._u_hi.to(obs.device)
            a_next_scaled = lo + 0.5 * (a_next + 1.0) * (hi - lo)
            # Filter next action through the CBF QP so the target Q sees filtered
            # actions (matching what was trained on).  project_batch is a vectorised
            # analytic projection — no OSQP overhead.  See plan §3.1 / §9.
            if hasattr(self.qp_filter, "project_batch"):
                a_next_safe_np = self.qp_filter.project_batch(
                    next_z_np, a_next_scaled.cpu().numpy()
                )
                a_next_safe = torch.as_tensor(a_next_safe_np, dtype=a_next_scaled.dtype,
                                              device=obs.device)
            else:
                a_next_safe = a_next_scaled          # NullFilter fallback
            q_next = torch.min(self.q1_t(next_obs, a_next_safe), self.q2_t(next_obs, a_next_safe))
            target = r + self.gamma * (1 - done) * (q_next - self.alpha * logp_next)
        q1_loss = F.mse_loss(self.q1(obs, a_safe), target)
        q2_loss = F.mse_loss(self.q2(obs, a_safe), target)
        return q1_loss, q2_loss

    def _compute_actor_loss(self, batch):
        obs = torch.as_tensor(batch["obs"], device=self.device)
        z = torch.as_tensor(batch["z"], device=self.device)
        a, logp, _, _ = self.actor(obs)
        # Rescale tanh output to actual action space so critic sees consistent scale.
        lo = self._u_lo.to(obs.device)
        hi = self._u_hi.to(obs.device)
        a_scaled = lo + 0.5 * (a + 1.0) * (hi - lo)
        # No no_grad here: gradient must flow through a_scaled→a→actor params.
        # Critic params also receive gradients but actor_opt does not own them,
        # so actor_opt.step() leaves the critics unchanged.
        q = torch.min(self.q1(obs, a_scaled), self.q2(obs, a_scaled))
        actor_loss = (self.alpha * logp - q).mean()
        if hasattr(self.qp_filter, "cbf_penalty_terms"):
            a_cbf, b_cbf = self.qp_filter.cbf_penalty_terms(z, a_scaled)
            gap = b_cbf - torch.sum(a_cbf * a_scaled, dim=-1)
            cbf_pen = (torch.clamp(gap, min=0.0) ** 2).mean()
            actor_loss = actor_loss + self.lam_h * cbf_pen
        return actor_loss

    def update(self, batch):
        q1_loss, q2_loss = self._compute_critic_loss(batch)
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()
        a_loss = self._compute_actor_loss(batch)
        self.actor_opt.zero_grad(); a_loss.backward(); self.actor_opt.step()
        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return {"q1_loss": float(q1_loss), "q2_loss": float(q2_loss), "actor_loss": float(a_loss)}

    def save(self, path) -> None:
        # Evaluation checkpoint — optimizer state is not preserved.
        torch.save({
            "dim_obs": self.dim_obs,
            "dim_action": self.dim_action,
            "dim_z": self.dim_z,
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_t": self.q1_t.state_dict(),
            "q2_t": self.q2_t.state_dict(),
        }, path)

    @classmethod
    def load(cls, path, koopman_model, qp_filter, **kwargs) -> "KCBFSACAgent":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(
            dim_obs=int(ckpt["dim_obs"]),
            dim_action=int(ckpt["dim_action"]),
            dim_z=int(ckpt["dim_z"]),
            koopman_model=koopman_model,
            qp_filter=qp_filter,
            **kwargs,
        )
        agent.actor.load_state_dict(ckpt["actor"])
        agent.q1.load_state_dict(ckpt["q1"])
        agent.q2.load_state_dict(ckpt["q2"])
        agent.q1_t.load_state_dict(ckpt["q1_t"])
        agent.q2_t.load_state_dict(ckpt["q2_t"])
        return agent
