"""Collaborative filtering as biased matrix factorization, written in plain NumPy.

    predicted rating(user u, movie i) = mu + b_u + b_i + p_u . q_i

Trained with full-batch Adam on the squared error plus L2 regularization:

    J = 1/2 * sum (pred - rating)^2  +  reg/2 * (|P|^2 + |Q|^2 + |b_u|^2 + |b_i|^2)
"""
import numpy as np

from .optim import Adam


class MatrixFactorization:
    def __init__(self, n_factors=20, reg=6.0, lr=0.03, seed=0):
        self.k, self.reg, self.lr, self.seed = n_factors, reg, lr, seed
        self.mu = 0.0
        self.p = {}
        self.n_iter_ = 0

    # ----- core math -----
    def _raw_predict(self, u, i, p=None):
        p = p or self.p
        return self.mu + p["bu"][u] + p["bi"][i] + (p["P"][u] * p["Q"][i]).sum(axis=1)

    def loss(self, u, i, r, p=None):
        p = p or self.p
        err = self._raw_predict(u, i, p) - r
        penalty = sum((v ** 2).sum() for v in p.values())
        return 0.5 * (err ** 2).sum() + 0.5 * self.reg * penalty

    def grads(self, u, i, r):
        p = self.p
        err = self._raw_predict(u, i) - r
        n_users, n_items = len(p["bu"]), len(p["bi"])
        Pu, Qi = p["P"][u], p["Q"][i]
        g = {
            "bu": np.bincount(u, weights=err, minlength=n_users) + self.reg * p["bu"],
            "bi": np.bincount(i, weights=err, minlength=n_items) + self.reg * p["bi"],
            "P": np.empty_like(p["P"]),
            "Q": np.empty_like(p["Q"]),
        }
        for f in range(self.k):  # scatter-add the gradient of each latent factor
            g["P"][:, f] = np.bincount(u, weights=err * Qi[:, f], minlength=n_users)
            g["Q"][:, f] = np.bincount(i, weights=err * Pu[:, f], minlength=n_items)
        g["P"] += self.reg * p["P"]
        g["Q"] += self.reg * p["Q"]
        return g

    # ----- training -----
    def fit(self, u, i, r, n_users, n_items, n_iter=400, val=None, eval_every=10, patience=5):
        """Train. If `val=(u, i, r)` is given, keep the parameters with the best validation RMSE."""
        rng = np.random.default_rng(self.seed)
        self.mu = float(r.mean())
        self.p = {
            "bu": np.zeros(n_users),
            "bi": np.zeros(n_items),
            "P": rng.normal(0, 0.1, (n_users, self.k)),
            "Q": rng.normal(0, 0.1, (n_items, self.k)),
        }
        opt = Adam(self.p, lr=self.lr)
        best_rmse, best_params, best_iter = np.inf, None, n_iter
        for it in range(1, n_iter + 1):
            opt.step(self.grads(u, i, r))
            if val is not None and it % eval_every == 0:
                rmse = self.rmse(*val)
                if rmse < best_rmse:
                    best_rmse, best_iter = rmse, it
                    best_params = {k: v.copy() for k, v in self.p.items()}
                elif it - best_iter >= patience * eval_every:
                    break
        if best_params is not None:
            self.p = best_params
        self.n_iter_ = best_iter
        return self

    # ----- inference -----
    def predict(self, u, i):
        return np.clip(self._raw_predict(u, i), 0.5, 5.0)

    def rmse(self, u, i, r):
        return float(np.sqrt(np.mean((self.predict(u, i) - r) ** 2)))

    def fold_in(self, item_idx, ratings):
        """Predict scores for ALL movies for a brand-new user, without retraining.

        With movie parameters frozen, the new user's (bias, factors) have a closed-form
        ridge-regression solution. This is what makes the app interactive.
        """
        item_idx, ratings = np.asarray(item_idx), np.asarray(ratings, dtype=float)
        Q, bi = self.p["Q"], self.p["bi"]
        A = np.hstack([Q[item_idx], np.ones((len(item_idx), 1))])
        target = ratings - self.mu - bi[item_idx]
        theta = np.linalg.solve(A.T @ A + self.reg * np.eye(self.k + 1), A.T @ target)
        user_vec, user_bias = theta[:-1], theta[-1]
        return np.clip(self.mu + bi + user_bias + Q @ user_vec, 0.5, 5.0)

    # ----- persistence -----
    def to_arrays(self):
        return {"cf_mu": np.array(self.mu), "cf_reg": np.array(self.reg),
                "cf_bi": self.p["bi"], "cf_Q": self.p["Q"]}
