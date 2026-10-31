"""Content-based filtering with a two-tower neural network, in plain NumPy.

    user features --> [Dense(hidden, ReLU) -> Dense(out)] --> L2-normalize --> v_u
    movie features --> [Dense(hidden, ReLU) -> Dense(out)] --> L2-normalize --> v_m
    predicted (scaled) rating = v_u . v_m        (a value in [-1, 1])

Ratings 0.5..5 are scaled to [-1, 1] so they match the range of a dot product of unit vectors.
Backpropagation is written by hand and verified with a numerical gradient check (tests/).
"""
import numpy as np

from .optim import Adam

RATING_MID, RATING_HALF = 2.75, 2.25  # maps 0.5..5.0 to -1..1


def scale_rating(r):
    return (r - RATING_MID) / RATING_HALF


def unscale_rating(y):
    return np.clip(y * RATING_HALF + RATING_MID, 0.5, 5.0)


def init_tower(n_in, hidden, n_out, rng):
    return {
        "W1": rng.normal(0, np.sqrt(2.0 / n_in), (n_in, hidden)),
        "b1": np.zeros(hidden),
        "W2": rng.normal(0, np.sqrt(1.0 / hidden), (hidden, n_out)),
        "b2": np.zeros(n_out),
    }


def tower_forward(x, p):
    h = np.maximum(x @ p["W1"] + p["b1"], 0.0)
    z = h @ p["W2"] + p["b2"]
    norm = np.sqrt((z ** 2).sum(axis=1, keepdims=True) + 1e-12)
    return z / norm, (x, h, z, norm)


def tower_backward(dv, cache, p):
    x, h, z, norm = cache
    v = z / norm
    dz = (dv - v * (dv * v).sum(axis=1, keepdims=True)) / norm  # gradient through L2-normalization
    dh = (dz @ p["W2"].T) * (h > 0)
    return {"W1": x.T @ dh, "b1": dh.sum(axis=0), "W2": h.T @ dz, "b2": dz.sum(axis=0)}


class TwoTower:
    def __init__(self, n_user_feats, n_item_feats, hidden=128, n_out=32, lr=0.003, seed=0):
        rng = np.random.default_rng(seed)
        self.user = init_tower(n_user_feats, hidden, n_out, rng)
        self.item = init_tower(n_item_feats, hidden, n_out, rng)
        self.lr = lr
        self.seed = seed
        self.epochs_ = 0

    def loss_and_grads(self, xu, xi, y):
        vu, cu = tower_forward(xu, self.user)
        vm, cm = tower_forward(xi, self.item)
        pred = (vu * vm).sum(axis=1)
        loss = np.mean((pred - y) ** 2)
        dpred = (2.0 / len(y)) * (pred - y)
        gu = tower_backward(dpred[:, None] * vm, cu, self.user)
        gi = tower_backward(dpred[:, None] * vu, cm, self.item)
        return loss, gu, gi

    def fit(self, xu, xi, y, val=None, epochs=60, batch_size=256, patience=6):
        """xu, xi: feature rows for every training rating; y: scaled ratings."""
        params = {**{f"u_{k}": v for k, v in self.user.items()},
                  **{f"i_{k}": v for k, v in self.item.items()}}
        opt = Adam(params, lr=self.lr)  # `params` shares memory with self.user / self.item
        rng = np.random.default_rng(self.seed)
        best, best_snapshot, best_epoch = np.inf, None, 0
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(y))
            for start in range(0, len(y), batch_size):
                b = order[start:start + batch_size]
                _, gu, gi = self.loss_and_grads(xu[b], xi[b], y[b])
                opt.step({**{f"u_{k}": v for k, v in gu.items()},
                          **{f"i_{k}": v for k, v in gi.items()}})
            if val is not None:
                v_loss = float(np.mean((self.predict_scaled(*val[:2]) - val[2]) ** 2))
                if v_loss < best:
                    best, best_epoch = v_loss, epoch
                    best_snapshot = {k: v.copy() for k, v in params.items()}
                elif epoch - best_epoch >= patience:
                    break
        if best_snapshot is not None:
            for k, v in best_snapshot.items():
                params[k][...] = v
        self.epochs_ = best_epoch or epochs
        return self

    def predict_scaled(self, xu, xi):
        return (tower_forward(xu, self.user)[0] * tower_forward(xi, self.item)[0]).sum(axis=1)

    def embed_users(self, xu):
        return tower_forward(xu, self.user)[0]

    def embed_items(self, xi):
        return tower_forward(xi, self.item)[0]

    def to_arrays(self):
        out = {f"tt_u_{k}": v for k, v in self.user.items()}
        out.update({f"tt_i_{k}": v for k, v in self.item.items()})
        return out
