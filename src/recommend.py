"""Inference for the app: loads the saved models and scores movies for a brand-new user."""
import json

import numpy as np
import pandas as pd

from . import data
from .pipeline import MODELS_DIR
from .two_tower import tower_forward, unscale_rating


class Recommender:
    def __init__(self, models_dir=MODELS_DIR):
        self.catalog = pd.read_csv(models_dir / "catalog.csv")
        self.G = data.genre_matrix(self.catalog["genres"])
        self.results = json.loads((models_dir / "results.json").read_text())

        cf = np.load(models_dir / "cf.npz")
        self.mu, self.reg = float(cf["cf_mu"]), float(cf["cf_reg"])
        self.bi, self.Q = cf["cf_bi"], cf["cf_Q"]

        tt = np.load(models_dir / "tower.npz")
        self.user_tower = {k: tt[f"tt_u_{k}"] for k in ("W1", "b1", "W2", "b2")}
        self.u_mean, self.u_std = tt["u_mean"], tt["u_std"]
        self.item_vecs = tt["item_vecs"]

    # ----- scoring a new user -----
    def cf_scores(self, rated: dict) -> np.ndarray:
        """rated: {item_index: rating}. Returns a predicted rating for every movie."""
        idx = np.array(list(rated))
        r = np.array(list(rated.values()), dtype=float)
        k = self.Q.shape[1]
        A = np.hstack([self.Q[idx], np.ones((len(idx), 1))])
        theta = np.linalg.solve(A.T @ A + self.reg * np.eye(k + 1),
                                A.T @ (r - self.mu - self.bi[idx]))
        return np.clip(self.mu + self.bi + theta[-1] + self.Q @ theta[:-1], 0.5, 5.0)

    def tower_scores(self, rated: dict) -> np.ndarray:
        idx = np.array(list(rated))
        r = np.array(list(rated.values()), dtype=float)
        feats = data.user_features(np.zeros(len(idx), dtype=int), idx, r, 1, self.G)
        v_user, _ = tower_forward((feats - self.u_mean) / self.u_std, self.user_tower)
        return unscale_rating(self.item_vecs @ v_user[0])

    # ----- helpers -----
    def top_n(self, scores, exclude, n=10, genres=None) -> pd.DataFrame:
        df = self.catalog.assign(predicted=scores)
        df = df.drop(index=list(exclude))
        if genres:
            wanted = [data.GENRES.index(g) for g in genres]
            df = df[self.G[df.index][:, wanted].max(axis=1) > 0]
        df = df.sort_values("predicted", ascending=False).head(n)
        return df[["title", "genres", "predicted", "mean_rating", "n_ratings"]]

    def similar(self, item, method="collaborative", n=10) -> pd.DataFrame:
        vecs = self.Q if method == "collaborative" else self.item_vecs
        unit = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
        sims = unit @ unit[item]
        sims[item] = -np.inf
        top = np.argsort(-sims)[:n]
        return self.catalog.loc[top, ["title", "genres"]].assign(similarity=sims[top])

    def find(self, title_prefix):
        """Index of the most-rated movie whose title starts with `title_prefix`, or None."""
        hits = self.catalog[self.catalog["title"].str.lower().str.startswith(title_prefix.lower())]
        return None if hits.empty else int(hits["n_ratings"].idxmax())
