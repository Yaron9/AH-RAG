from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None     # type: ignore
    optim = None  # type: ignore


class ActorCritic(nn.Module):  # type: ignore[misc]
    def __init__(self, in_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value


@dataclass
class PPOConfig:
    epochs: int = 3
    gamma: float = 0.99
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 3e-4
    batch_size: int = 256


def _to_torch(x: np.ndarray) -> "torch.Tensor":
    return torch.tensor(x, dtype=torch.float32)


def ppo_update(model: ActorCritic, opt: Any, cfg: PPOConfig,
               obs: np.ndarray, actions: np.ndarray, old_logp: np.ndarray,
               returns: np.ndarray, adv: np.ndarray) -> Dict[str, float]:
    if torch is None:
        raise RuntimeError("PyTorch is required for PPO training.")
    model.train()
    n = obs.shape[0]
    idx = np.arange(n)
    losses: Dict[str, float] = {"policy": 0.0, "value": 0.0, "entropy": 0.0}
    for _ in range(cfg.epochs):
        np.random.shuffle(idx)
        for i in range(0, n, cfg.batch_size):
            b = idx[i:i+cfg.batch_size]
            xb = _to_torch(obs[b])
            ab = torch.tensor(actions[b], dtype=torch.long)
            lb = _to_torch(old_logp[b])
            rb = _to_torch(returns[b])
            advb = _to_torch(adv[b])

            logits, v = model(xb)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(ab)
            ratio = torch.exp(logp - lb)
            # policy loss with clipping
            unclipped = ratio * advb
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advb
            policy_loss = -torch.mean(torch.min(unclipped, clipped))
            # value loss
            value_loss = torch.mean((v - rb) ** 2)
            # entropy bonus
            entropy = torch.mean(dist.entropy())

            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses["policy"] += float(policy_loss.item()) * len(b)
            losses["value"] += float(value_loss.item()) * len(b)
            losses["entropy"] += float(entropy.item()) * len(b)
    # average
    for k in list(losses.keys()):
        losses[k] /= max(1, n * cfg.epochs)
    return losses


def compute_gae(rews: List[float], vals: List[float], dones: List[bool], gamma: float = 0.99, lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    n = len(rews)
    adv = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_non_terminal = 0.0 if (t == n - 1 or dones[t]) else 1.0
        next_value = 0.0 if (t == n - 1 or dones[t]) else vals[t + 1]
        delta = rews[t] + gamma * next_value * next_non_terminal - vals[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        adv[t] = last_gae
    returns = adv + np.asarray(vals, dtype=np.float32)
    # normalize adv
    if np.std(adv) > 1e-8:
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
    return adv.astype(np.float32), returns.astype(np.float32)


def act_and_logp(model: ActorCritic, obs_vec: np.ndarray, mask: np.ndarray | None = None) -> Tuple[int, float, float]:
    if torch is None:
        raise RuntimeError("PyTorch is required for PPO training.")
    with torch.no_grad():
        x = _to_torch(obs_vec.reshape(1, -1))
        logits, v = model(x)
        if mask is not None:
            # set invalid actions to large negative before building distribution
            m = torch.tensor(mask.reshape(1, -1), dtype=torch.float32)
            neg_inf = torch.full_like(logits, -1e9)
            logits = torch.where(m > 0.5, logits, neg_inf)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
    return int(a.item()), float(logp.item()), float(v.item())


def ppo_train(env_ctor,
              total_episodes: int = 50,
              max_steps: int = 6,
              ppo_cfg: PPOConfig | None = None,
              save_path: str = "artifacts/rl/ppo_policy.pt",
              n_envs: int = 1,
              early_stop_patience: int = 5,
              early_stop_min_improve: float = 0.05) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for PPO training.")
    ppo_cfg = ppo_cfg or PPOConfig()
    # vector of envs for batched collection (sequential roll)
    envs = [env_ctor() for _ in range(max(1, int(n_envs)))]
    # infer input dim
    obs0, _ = envs[0].reset("warmup question")
    in_dim = int(obs0.shape[0])
    n_actions = int(envs[0].action_size)
    model = ActorCritic(in_dim, n_actions)
    opt = optim.Adam(model.parameters(), lr=ppo_cfg.lr)

    # Simple loop over questions pulled from dataset
    import os, sys
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    sys.path.append(ROOT)
    from scripts.run_benchmark import load_dataset  # type: ignore
    data = load_dataset("hotpotqa", limit=total_episodes * max(1, int(n_envs)))

    ep_idx = 0
    best_mavg = -1e9
    stale = 0
    for i in range(0, len(data), max(1, int(n_envs))):
        items = data[i:i+max(1, int(n_envs))]
        # reset envs
        obs0s: List[np.ndarray] = []
        for e, env in enumerate(envs):
            q = items[e]["question"] if e < len(items) else ""
            ov, _ = env.reset(q)
            obs0s.append(ov)

        # collect episodes from each env
        batch_obs_list: List[np.ndarray] = []
        batch_act_list: List[np.ndarray] = []
        batch_logp_list: List[np.ndarray] = []
        batch_ret_list: List[np.ndarray] = []
        batch_adv_list: List[np.ndarray] = []
        ep_rewards: List[float] = []

        for e, env in enumerate(envs):
            obs = obs0s[e]
            obs_list: List[np.ndarray] = []
            act_list: List[int] = []
            logp_list: List[float] = []
            rew_list: List[float] = []
            val_list: List[float] = []
            done_list: List[bool] = []

            steps = 0
            done = False
            while not done and steps < max_steps:
                try:
                    m = np.asarray(env.get_action_mask(), dtype=np.float32)
                except Exception:
                    m = None
                a, lp, v = act_and_logp(model, obs, mask=m)
                nobs, r, done, _info = env.step(a)
                obs_list.append(obs)
                act_list.append(a)
                logp_list.append(lp)
                rew_list.append(r)
                val_list.append(v)
                done_list.append(done)
                obs = nobs
                steps += 1

            adv, ret = compute_gae(rew_list, val_list, done_list, gamma=ppo_cfg.gamma, lam=0.95)
            batch_obs_list.append(np.stack(obs_list, axis=0))
            batch_act_list.append(np.asarray(act_list, dtype=np.int64))
            batch_logp_list.append(np.asarray(logp_list, dtype=np.float32))
            batch_ret_list.append(ret)
            batch_adv_list.append(adv)
            ep_idx += 1
            ep_rewards.append(float(np.sum(rew_list)))
            print(f"[PPO] episode={ep_idx} env={e} steps={steps} ep_reward={ep_rewards[-1]:.3f}")

        # concatenate and update once
        batch_obs = np.concatenate(batch_obs_list, axis=0)
        batch_act = np.concatenate(batch_act_list, axis=0)
        batch_logp = np.concatenate(batch_logp_list, axis=0)
        batch_ret = np.concatenate(batch_ret_list, axis=0)
        batch_adv = np.concatenate(batch_adv_list, axis=0)
        losses = ppo_update(model, opt, ppo_cfg, batch_obs, batch_act, batch_logp, batch_ret, batch_adv)
        mavg = float(np.mean(ep_rewards))
        print(f"[PPO] update mavg_ep_reward={mavg:.3f} loss={losses}")

        # early stopping on moving-average ep reward
        if mavg > best_mavg + float(early_stop_min_improve):
            best_mavg = mavg
            stale = 0
        else:
            stale += 1
            if stale >= max(1, int(early_stop_patience)):
                print(f"[PPO] early stopping at update {i//max(1,int(n_envs))}: best mavg={best_mavg:.3f}")
                break

    # save model
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": in_dim, "n_actions": n_actions}, save_path)
    print(f"Saved PPO policy to {save_path}")


def load_ppo(path: str) -> ActorCritic:
    if torch is None:
        raise RuntimeError("PyTorch is required to load PPO policy.")
    payload = torch.load(path, map_location="cpu")
    model = ActorCritic(int(payload["in_dim"]), int(payload["n_actions"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def act_ppo(model: ActorCritic, obs_vec: np.ndarray) -> int:
    with torch.no_grad():
        x = _to_torch(obs_vec.reshape(1, -1))
        logits, _ = model(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
    return int(a.item())
