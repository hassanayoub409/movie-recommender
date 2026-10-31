"""Train both recommenders and save everything the app needs to models/."""
import json
import time

import numpy as np

from . import data
from .mf import MatrixFactorization
from .two_tower import TwoTower, scale_rating, unscale_rating

MODELS_DIR = data.ROOT / "models"
ARTIFACTS = ["catalog.csv", "cf.npz", "tower.npz", "results.json"]
REG_GRID = [2.0, 6.0, 15.0]  # matrix-factorization regularization strengths to try
N_FACTORS = 20


def artifacts_exist() -> bool:
    return all((MODELS_DIR / name).exists() for name in ARTIFACTS)


def _rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _standardizer(X):
    mean, std = X.mean(axis=0), X.std(axis=0)
    return mean, np.where(std < 1e-8, 1.0, std)


def train_all(log=print) -> dict:
    start = time.time()
    log("Getting MovieLens data...")
    ratings, movies = data.load_raw(data.download())
    ds = data.prepare(ratings, movies)
    u, i, r = ds.train
    tu, ti, tr = ds.test
    log(f"{ds.n_users} users, {ds.n_items} movies, {len(r)} train / {len(tr)} test ratings")

    # A slice of the training data is held out to pick hyperparameters / stop early.
    is_val = np.random.default_rng(0).random(len(r)) < 0.1
    fit_rows = (u[~is_val], i[~is_val], r[~is_val])
    val_rows = (u[is_val], i[is_val], r[is_val])

    # ---------- 1) Collaborative filtering ----------
    log("Training collaborative filtering (matrix factorization)...")
    best = None
    for reg in REG_GRID:
        mf = MatrixFactorization(n_factors=N_FACTORS, reg=reg).fit(
            *fit_rows, ds.n_users, ds.n_items, val=val_rows)
        score = mf.rmse(*val_rows)
        log(f"  reg={reg:<5} validation RMSE {score:.4f} ({mf.n_iter_} iterations)")
        if best is None or score < best[0]:
            best = (score, reg, mf.n_iter_)
    _, best_reg, best_iter = best
    cf = MatrixFactorization(n_factors=N_FACTORS, reg=best_reg).fit(
        u, i, r, ds.n_users, ds.n_items, n_iter=best_iter)
    bias_only = MatrixFactorization(n_factors=0, reg=best_reg).fit(
        u, i, r, ds.n_users, ds.n_items, n_iter=200)

    # ---------- 2) Content-based two-tower network ----------
    log("Training content-based two-tower network...")
    user_X = data.user_features(u, i, r, ds.n_users, ds.G)
    item_X = data.item_features(ds.movies, ds.G, i, r)
    u_mean, u_std = _standardizer(user_X)
    i_mean, i_std = _standardizer(item_X)
    user_S, item_S = (user_X - u_mean) / u_std, (item_X - i_mean) / i_std
    y = scale_rating(r)
    tower = TwoTower(user_S.shape[1], item_S.shape[1])
    tower.fit(
        user_S[u[~is_val]], item_S[i[~is_val]], y[~is_val],
        val=(user_S[u[is_val]], item_S[i[is_val]], y[is_val]),
    )
    log(f"  stopped at epoch {tower.epochs_}")
    item_vecs = tower.embed_items(item_S)
    tower_pred = unscale_rating((tower.embed_users(user_S)[tu] * item_vecs[ti]).sum(axis=1))

    # ---------- 3) Evaluate on the untouched test split ----------
    metrics = {
        "Global average": _rmse(np.full(len(tr), r.mean()), tr),
        "User + movie biases only": bias_only.rmse(tu, ti, tr),
        "Collaborative filtering": cf.rmse(tu, ti, tr),
        "Content-based two-tower": _rmse(tower_pred, tr),
    }
    for name, value in metrics.items():
        log(f"  test RMSE  {name:<26} {value:.4f}")

    # ---------- 4) Save ----------
    MODELS_DIR.mkdir(exist_ok=True)
    ds.movies[["movieId", "title", "year", "genres", "n_ratings", "mean_rating"]].to_csv(
        MODELS_DIR / "catalog.csv", index=False)
    np.savez(MODELS_DIR / "cf.npz", **cf.to_arrays())
    np.savez(MODELS_DIR / "tower.npz", **tower.to_arrays(), u_mean=u_mean, u_std=u_std,
             i_mean=i_mean, i_std=i_std, item_vecs=item_vecs)
    results = {
        "test_rmse": {k: round(v, 4) for k, v in metrics.items()},
        "dataset": {"users": ds.n_users, "movies": ds.n_items,
                    "train_ratings": int(len(r)), "test_ratings": int(len(tr))},
        "cf": {"factors": N_FACTORS, "reg": best_reg, "iterations": best_iter},
        "two_tower": {"epochs": tower.epochs_, "user_features": user_S.shape[1],
                      "item_features": item_S.shape[1]},
        "seconds": round(time.time() - start, 1),
    }
    (MODELS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    log(f"Done in {results['seconds']}s. Saved models to {MODELS_DIR}")
    return results
