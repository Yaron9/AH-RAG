from __future__ import annotations

from typing import List, Tuple
import json
import math
import os

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None     # type: ignore
    optim = None  # type: ignore


class MLPPolicy(nn.Module):  # type: ignore[misc]
    def __init__(self, in_dim: int, n_actions: int = 6, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def _infer_dim_from_traj(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                steps = obj.get("steps") or []
                if steps:
                    vec = steps[0].get("obs_vec") or []
                    return int(len(vec))
            except Exception:
                continue
    raise RuntimeError("Cannot infer obs_vec dimension from trajectory file")


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    X: List[List[float]] = []
    y: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            for s in obj.get("steps", []):
                vec = s.get("obs_vec") or []
                act = s.get("action")
                if isinstance(act, int) and vec and all(isinstance(v, (int, float)) for v in vec):
                    X.append([float(v) for v in vec])
                    y.append(int(act))
    if not X:
        raise RuntimeError("No (obs_vec, action) pairs found in trajectories")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def train_bc(traj_path: str, out_path: str, epochs: int = 5, lr: float = 1e-3, n_actions: int = 6) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for BC training. Please install torch.")
    in_dim = _infer_dim_from_traj(traj_path)
    X, y = load_dataset(traj_path)
    model = MLPPolicy(in_dim, n_actions=n_actions)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    bs = 256
    n = X.shape[0]
    for ep in range(epochs):
        perm = np.random.permutation(n)
        Xp = X[perm]
        yp = y[perm]
        total = 0.0
        for i in range(0, n, bs):
            xb = torch.tensor(Xp[i:i+bs])
            yb = torch.tensor(yp[i:i+bs])
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * (yb.shape[0])
        avg = total / max(1, n)
        # simple print for monitoring
        print(f"[BC] epoch={ep+1} loss={avg:.4f}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": in_dim, "n_actions": n_actions}, out_path)
    print(f"Saved BC policy to {out_path}")


def load_bc(out_path: str) -> MLPPolicy:
    if torch is None:
        raise RuntimeError("PyTorch is required to load BC policy.")
    payload = torch.load(out_path, map_location="cpu")
    model = MLPPolicy(int(payload["in_dim"]), int(payload["n_actions"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def act_bc(model: MLPPolicy, obs_vec: np.ndarray) -> int:
    with torch.no_grad():
        x = torch.tensor(obs_vec.reshape(1, -1))
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    # sample action
    r = np.random.rand()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += float(p)
        if r <= acc:
            return i
    return int(np.argmax(probs))

