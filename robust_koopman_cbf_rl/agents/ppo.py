"""KCBF-augmented PPO. Env-applied action is u_safe, but PPO updates ratio on u_nom."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, dim_obs, dim_action, hidden=(64, 64)):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(dim_obs, hidden[0]), nn.Tanh(),
            nn.Linear(hidden[0], hidden[1]), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden[1], dim_action)
        self.log_std = nn.Parameter(torch.zeros(dim_action))
        self.v = nn.Linear(hidden[1], 1)

    def forward(self, obs):
        h = self.shared(obs)
        mu_out = self.mu(h)
        return mu_out, self.log_std.exp().expand_as(mu_out), self.v(h)

    def distribution(self, obs):
        mu, std, v = self.forward(obs)
        return torch.distributions.Normal(mu, std), v


class KCBFPPOAgent:
    def __init__(self, dim_obs, dim_action, dim_z,
                 koopman_model, qp_filter,
                 lam_h=1.0, gamma=0.99, lam=0.95, lr=3e-4,
                 actor_lr=None, critic_lr=None,
                 clip=0.2, c_v=0.5, c_e=0.01, device="cpu"):
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.dim_z = dim_z
        self.koopman_model = koopman_model
        self.qp_filter = qp_filter
        self.lam_h = float(lam_h)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.clip = float(clip)
        self.c_v = float(c_v)
        self.c_e = float(c_e)
        self.device = device
        self.ac = ActorCritic(dim_obs, dim_action).to(device)
        _actor_lr = actor_lr if actor_lr is not None else lr
        _critic_lr = critic_lr if critic_lr is not None else lr
        # Backbone shared by both heads; actor optimizer owns it so the policy
        # gradient signal drives the representation. Critic optimizer governs
        # only the value head, using critic_lr independently.
        actor_params = (list(self.ac.shared.parameters()) +
                        list(self.ac.mu.parameters()) + [self.ac.log_std])
        critic_params = list(self.ac.v.parameters())
        self.actor_opt = torch.optim.Adam(actor_params, lr=_actor_lr)
        self.critic_opt = torch.optim.Adam(critic_params, lr=_critic_lr)
        self._u_lo = torch.as_tensor(qp_filter.u_min, dtype=torch.float32)
        self._u_hi = torch.as_tensor(qp_filter.u_max, dtype=torch.float32)

    @torch.no_grad()
    def select_action(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist, v = self.ac.distribution(obs_t)
        a = dist.sample()
        logp = dist.log_prob(a).sum(-1)
        u_nom = a.cpu().numpy()[0]
        y_dim = getattr(self.koopman_model.observables, "dim_y", obs.shape[0])
        z = self.koopman_model.lift(np.asarray(obs, dtype=np.float64)[:y_dim])
        u_safe, diag = self.qp_filter.project(z, u_nom)
        return (u_safe.astype(np.float32), u_nom.astype(np.float32),
                float(logp.item()), float(v.item()), diag)

    def update(self, batch, epochs: int = 10, minibatch: int = 64):
        obs = torch.as_tensor(batch["obs"], device=self.device)
        z = torch.as_tensor(batch["z"], device=self.device)
        a_nom = torch.as_tensor(batch["action_nom"], device=self.device)
        old_logp = torch.as_tensor(batch["logprob_nom"], device=self.device)
        adv = torch.as_tensor(batch["advantages"], device=self.device)
        ret = torch.as_tensor(batch["returns"], device=self.device)
        n = obs.shape[0]
        idx = np.arange(n)
        stats = {"loss": 0.0, "cbf_pen": 0.0}
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, n, minibatch):
                mb = idx[s:s + minibatch]
                dist, v = self.ac.distribution(obs[mb])
                logp = dist.log_prob(a_nom[mb]).sum(-1)
                ratio = torch.exp(logp - old_logp[mb])
                s1 = ratio * adv[mb]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[mb]
                l_clip = -torch.min(s1, s2).mean()
                l_v = F.mse_loss(v.squeeze(-1), ret[mb])
                entropy = dist.entropy().sum(-1).mean()
                cbf_pen = torch.zeros((), device=obs.device)
                if hasattr(self.qp_filter, "cbf_penalty_terms"):
                    # Use rsample (reparameterised) so gradient flows through actor params.
                    # Squash via tanh then rescale to [u_min, u_max] — matches the domain
                    # the CBF constraint is defined over and prevents ±∞ samples from
                    # zeroing the penalty when the policy is most unsafe.
                    lo = self._u_lo.to(obs.device)
                    hi = self._u_hi.to(obs.device)
                    a_fresh = lo + 0.5 * (torch.tanh(dist.rsample()) + 1.0) * (hi - lo)
                    a_cbf, b_cbf = self.qp_filter.cbf_penalty_terms(z[mb], a_fresh)
                    gap = b_cbf - torch.sum(a_cbf * a_fresh, dim=-1)
                    cbf_pen = (torch.clamp(gap, min=0.0) ** 2).mean()
                loss = l_clip + self.c_v * l_v - self.c_e * entropy + self.lam_h * cbf_pen
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                self.actor_opt.step()
                self.critic_opt.step()
                stats["loss"] += float(loss); stats["cbf_pen"] += float(cbf_pen)
        return stats

    def save(self, path) -> None:
        # Evaluation checkpoint — optimizer state is not preserved.
        torch.save({
            "dim_obs": self.dim_obs,
            "dim_action": self.dim_action,
            "dim_z": self.dim_z,
            "ac": self.ac.state_dict(),
        }, path)

    @classmethod
    def load(cls, path, koopman_model, qp_filter, **kwargs) -> "KCBFPPOAgent":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(
            dim_obs=int(ckpt["dim_obs"]),
            dim_action=int(ckpt["dim_action"]),
            dim_z=int(ckpt["dim_z"]),
            koopman_model=koopman_model,
            qp_filter=qp_filter,
            **kwargs,
        )
        agent.ac.load_state_dict(ckpt["ac"])
        return agent
