"""Download, clean and split the MovieLens "ml-latest-small" dataset.

The data is NOT stored in this repository: MovieLens is distributed by GroupLens
under terms that restrict redistribution, so each user downloads it directly from
GroupLens (see download()).
"""
import os
import re
import ssl
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_URL = os.environ.get(
    "MOVIELENS_URL", "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
)

# "IMAX" (a screen format) and "(no genres listed)" are not used as genres.
GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


def download(dest: Path = DATA_DIR) -> Path:
    """Download and unzip ml-latest-small (once). Returns the folder with the CSV files."""
    folder = dest / "ml-latest-small"
    if (folder / "ratings.csv").exists():
        return folder
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "ml-latest-small.zip"
    try:
        import certifi  # python.org builds of Python on macOS often lack CA certificates

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = None
    try:
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "movie-recommender"})
        with urllib.request.urlopen(request, context=context, timeout=60) as resp:
            zip_path.write_bytes(resp.read())
    except Exception as err:
        raise RuntimeError(
            f"Could not download MovieLens ({err}).\n"
            "Download 'ml-latest-small.zip' manually from https://grouplens.org/datasets/movielens/latest/ "
            f"and unzip it so that {folder / 'ratings.csv'} exists."
        ) from err
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    zip_path.unlink()
    return folder


def load_raw(folder: Path):
    ratings = pd.read_csv(folder / "ratings.csv", usecols=["userId", "movieId", "rating"])
    movies = pd.read_csv(folder / "movies.csv")
    return ratings, movies


def parse_year(title: str):
    match = re.search(r"\((\d{4})\)\s*$", title)
    return float(match.group(1)) if match else np.nan


def genre_matrix(genre_strings) -> np.ndarray:
    """Multi-hot matrix (n_movies, n_genres) from 'Action|Adventure' style strings."""
    G = np.zeros((len(genre_strings), len(GENRES)))
    for row, text in enumerate(genre_strings):
        for g in str(text).split("|"):
            if g in GENRES:
                G[row, GENRES.index(g)] = 1.0
    return G


@dataclass
class Dataset:
    movies: pd.DataFrame      # one row per kept movie; row number == item index
    G: np.ndarray             # (n_items, n_genres) multi-hot
    train: tuple              # (user_idx, item_idx, rating)
    test: tuple
    n_users: int
    n_items: int


def prepare(ratings, movies, min_movie_ratings=10, min_user_ratings=5,
            test_size=0.2, seed=42) -> Dataset:
    """Filter to movies with enough ratings, index users/movies, and make a random split."""
    counts = ratings["movieId"].value_counts()
    ratings = ratings[ratings["movieId"].isin(counts[counts >= min_movie_ratings].index)]
    ucounts = ratings["userId"].value_counts()
    ratings = ratings[ratings["userId"].isin(ucounts[ucounts >= min_user_ratings].index)]

    movies = movies[movies["movieId"].isin(ratings["movieId"].unique())]
    movies = movies.sort_values("movieId").reset_index(drop=True)
    item_of = {m: i for i, m in enumerate(movies["movieId"])}
    user_of = {u: j for j, u in enumerate(sorted(ratings["userId"].unique()))}

    u = ratings["userId"].map(user_of).to_numpy()
    i = ratings["movieId"].map(item_of).to_numpy()
    r = ratings["rating"].to_numpy(dtype=float)

    is_test = np.random.default_rng(seed).random(len(r)) < test_size
    seen_users, seen_items = set(u[~is_test]), set(i[~is_test])
    # Test rows must involve a user and a movie that appear in the training data.
    is_test &= np.array([a in seen_users and b in seen_items for a, b in zip(u, i)])
    tr, te = ~is_test, is_test

    movies = movies.copy()
    movies["year"] = movies["title"].map(parse_year)
    movies["year"] = movies["year"].fillna(movies["year"].median())
    movies["n_ratings"] = np.bincount(i, minlength=len(movies))
    movies["mean_rating"] = np.bincount(i, weights=r, minlength=len(movies)) / movies["n_ratings"]
    return Dataset(
        movies=movies,
        G=genre_matrix(movies["genres"]),
        train=(u[tr], i[tr], r[tr]),
        test=(u[te], i[te], r[te]),
        n_users=len(user_of),
        n_items=len(movies),
    )


# ---------- Content features for the two-tower model ----------

def user_features(u, i, r, n_users, G) -> np.ndarray:
    """Per-user taste profile: average rating given to each genre (0 if never rated) + overall mean.

    Works for any number of users, including a single brand-new user (n_users=1, u all zeros).
    """
    n_genres = G.shape[1]
    out = np.zeros((n_users, n_genres + 1))
    n_rated = np.bincount(u, minlength=n_users)
    out[:, -1] = np.bincount(u, weights=r, minlength=n_users) / np.maximum(n_rated, 1)
    for g in range(n_genres):
        has_g = G[i, g] > 0
        total = np.bincount(u[has_g], weights=r[has_g], minlength=n_users)
        count = np.bincount(u[has_g], minlength=n_users)
        out[:, g] = total / np.maximum(count, 1)
    return out


def item_features(movies: pd.DataFrame, G, i_train, r_train) -> np.ndarray:
    """Per-movie features: year, mean rating and popularity (from TRAIN ratings only) + genres."""
    n_items = len(movies)
    count = np.bincount(i_train, minlength=n_items)
    total = np.bincount(i_train, weights=r_train, minlength=n_items)
    mean = np.where(count > 0, total / np.maximum(count, 1), r_train.mean())
    return np.column_stack([movies["year"].to_numpy(), mean, np.log1p(count), G])
