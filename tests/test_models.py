"""Run from the repo root:  python -m pytest -q"""
import numpy as np

from src.data import user_features
from src.mf import MatrixFactorization
from src.two_tower import TwoTower, scale_rating, unscale_rating


def numeric_grad(f, array, idx, eps=1e-6):
    old = array[idx]
    array[idx] = old + eps
    up = f()
    array[idx] = old - eps
    down = f()
    array[idx] = old
    return (up - down) / (2 * eps)


def test_two_tower_gradients_match_numerical():
    rng = np.random.default_rng(0)
    xu, xi = rng.normal(size=(12, 5)), rng.normal(size=(12, 7))
    y = rng.uniform(-1, 1, 12)
    model = TwoTower(5, 7, hidden=8, n_out=4, seed=1)
    _, gu, gi = model.loss_and_grads(xu, xi, y)
    loss = lambda: model.loss_and_grads(xu, xi, y)[0]
    for tower, grads in ((model.user, gu), (model.item, gi)):
        for name, arr in tower.items():
            for _ in range(4):
                idx = tuple(rng.integers(0, s) for s in arr.shape)
                assert np.isclose(grads[name][idx], numeric_grad(loss, arr, idx), rtol=1e-4, atol=1e-7)


def test_matrix_factorization_gradients_match_numerical():
    rng = np.random.default_rng(0)
    u, i = rng.integers(0, 6, 40), rng.integers(0, 9, 40)
    r = rng.uniform(0.5, 5, 40)
    mf = MatrixFactorization(n_factors=3, reg=1.5)
    mf.fit(u, i, r, 6, 9, n_iter=3)  # just to initialise the parameters
    g = mf.grads(u, i, r)
    loss = lambda: mf.loss(u, i, r)
    for name, arr in mf.p.items():
        for _ in range(4):
            idx = tuple(rng.integers(0, s) for s in arr.shape)
            assert np.isclose(g[name][idx], numeric_grad(loss, arr, idx), rtol=1e-4, atol=1e-6)


def test_matrix_factorization_learns_low_rank_structure():
    rng = np.random.default_rng(1)
    n_u, n_i, k = 60, 80, 2
    truth = rng.normal(size=(n_u, k)) @ rng.normal(size=(n_i, k)).T * 0.5 + 3.0
    u, i = np.meshgrid(np.arange(n_u), np.arange(n_i), indexing="ij")
    u, i = u.ravel(), i.ravel()
    r = np.clip(truth.ravel() + rng.normal(0, 0.1, u.size), 0.5, 5)
    keep = rng.random(u.size) < 0.5
    mf = MatrixFactorization(n_factors=4, reg=2.0).fit(u[keep], i[keep], r[keep], n_u, n_i, n_iter=300)
    baseline = np.sqrt(np.mean((r[~keep] - r[keep].mean()) ** 2))
    assert mf.rmse(u[~keep], i[~keep], r[~keep]) < 0.6 * baseline


def test_fold_in_reproduces_a_known_users_tastes():
    rng = np.random.default_rng(2)
    n_u, n_i = 50, 60
    P, Q = rng.normal(size=(n_u, 2)), rng.normal(size=(n_i, 2))
    truth = np.clip(3 + 0.6 * P @ Q.T, 0.5, 5)
    u, i = np.meshgrid(np.arange(n_u), np.arange(n_i), indexing="ij")
    mf = MatrixFactorization(n_factors=4, reg=1.0).fit(u.ravel(), i.ravel(), truth.ravel(), n_u, n_i, n_iter=300)
    rated = np.arange(0, n_i, 3)
    scores = mf.fold_in(rated, truth[0, rated])
    assert np.corrcoef(scores, truth[0])[0, 1] > 0.9


def test_user_features_are_genre_averages():
    G = np.array([[1, 0], [1, 1], [0, 1]], dtype=float)  # 3 movies, 2 genres
    u, i, r = np.array([0, 0, 0]), np.array([0, 1, 2]), np.array([4.0, 2.0, 5.0])
    feats = user_features(u, i, r, 1, G)
    assert np.allclose(feats[0], [(4 + 2) / 2, (2 + 5) / 2, (4 + 2 + 5) / 3])


def test_rating_scaling_roundtrip():
    r = np.array([0.5, 2.75, 5.0])
    assert np.allclose(unscale_rating(scale_rating(r)), r)
    assert np.allclose(scale_rating(r), [-1, 0, 1])
