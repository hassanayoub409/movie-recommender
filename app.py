"""Streamlit app: Movie Recommender (collaborative filtering + content-based two-tower).

Run with:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from src import data
from src.pipeline import artifacts_exist, train_all
from src.recommend import Recommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner=False)
def get_recommender() -> Recommender:
    if not artifacts_exist():
        with st.spinner("First run: downloading MovieLens and training both models (about a minute)..."):
            train_all(log=lambda *_: None)
    return Recommender()


try:
    rec = get_recommender()
except RuntimeError as err:  # e.g. download problems
    st.error(str(err))
    st.stop()

catalog = rec.catalog
by_popularity = catalog.sort_values("n_ratings", ascending=False).index.tolist()
RATING_STEPS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# Taste presets: (exact MovieLens title, rating). Titles missing from the dataset are skipped.
PRESETS = {
    "Sci-fi fan": [
        ("Matrix, The (1999)", 5), ("Blade Runner (1982)", 5), ("Interstellar (2014)", 4.5),
        ("Inception (2010)", 5), ("Alien (1979)", 4.5), ("Terminator 2: Judgment Day (1991)", 4.5),
        ("Star Wars: Episode IV - A New Hope (1977)", 4.5), ("Aladdin (1992)", 2),
    ],
    "Family & animation": [
        ("Toy Story (1995)", 5), ("Finding Nemo (2003)", 5), ("Incredibles, The (2004)", 5),
        ("Shrek (2001)", 4.5), ("Lion King, The (1994)", 4.5), ("Monsters, Inc. (2001)", 4.5),
        ("Pulp Fiction (1994)", 2), ("Fargo (1996)", 2),
    ],
    "Crime & drama": [
        ("Godfather, The (1972)", 5), ("Pulp Fiction (1994)", 5), ("Goodfellas (1990)", 4.5),
        ("Shawshank Redemption, The (1994)", 5), ("Fargo (1996)", 4.5),
        ("Usual Suspects, The (1995)", 4.5), ("Silence of the Lambs, The (1991)", 4.5),
        ("Toy Story (1995)", 2), ("Shrek (2001)", 2),
    ],
}


def apply_preset(name):
    picks = {}
    for title, rating in PRESETS[name]:
        idx = rec.find(title)
        if idx is not None and catalog.title[idx] == title:
            picks[idx] = float(rating)
    st.session_state["picked"] = list(picks)
    for idx, rating in picks.items():
        st.session_state[f"r_{idx}"] = rating


# ---------------- Sidebar: the new user's ratings ----------------
with st.sidebar:
    st.header("1 · Rate some movies")
    st.caption("Pick a preset, or search for movies you know and rate them.")
    for name in PRESETS:
        st.button(name, on_click=apply_preset, args=(name,))
    st.multiselect("Search and add movies", by_popularity, key="picked",
                   format_func=lambda i: catalog.title[i])
    rated = {}
    for idx in st.session_state["picked"]:
        key = f"r_{idx}"
        if key not in st.session_state:
            st.session_state[key] = 3.5
        rated[idx] = st.select_slider(catalog.title[idx], options=RATING_STEPS, key=key)

# ---------------- Main area ----------------
st.title("Movie Recommender")
st.caption("Two recommender systems, written from scratch in NumPy, trained on MovieLens ratings.")

tab_recs, tab_similar, tab_perf, tab_how = st.tabs(
    ["Recommendations", "Similar movies", "Model performance", "How it works"])

TABLE_CONFIG = {
    "title": st.column_config.TextColumn("Movie"),
    "predicted": st.column_config.NumberColumn("Predicted", format="%.2f"),
    "genres": st.column_config.TextColumn("Genres"),
    "mean_rating": st.column_config.NumberColumn("Avg rating", format="%.2f"),
    "n_ratings": st.column_config.NumberColumn("# ratings"),
}

with tab_recs:
    if len(rated) < 3:
        st.info("Rate at least **3 movies** in the sidebar (or click a preset) to get recommendations.")
    else:
        genre_filter = st.multiselect("Only show these genres (optional)", data.GENRES)
        left, right = st.columns(2)
        cf_top = rec.top_n(rec.cf_scores(rated), rated, genres=genre_filter)
        tt_top = rec.top_n(rec.tower_scores(rated), rated, genres=genre_filter)
        left.subheader("Collaborative filtering")
        left.caption("\"People with similar rating patterns liked these.\"")
        left.dataframe(cf_top, hide_index=True, column_config=TABLE_CONFIG)
        right.subheader("Content-based (two-tower network)")
        right.caption("\"These match your taste by genre, year and popularity.\"")
        right.dataframe(tt_top, hide_index=True, column_config=TABLE_CONFIG)
        st.caption("Movies you already rated are excluded. Only movies with at least 10 ratings are considered.")

with tab_similar:
    movie = st.selectbox("Pick a movie", by_popularity, format_func=lambda i: catalog.title[i])
    method = st.radio("Measure similarity using",
                      ["Collaborative (learned from ratings)", "Content-based (learned from genres, year, popularity)"],
                      horizontal=True)
    kind = "collaborative" if method.startswith("Collaborative") else "content"
    st.dataframe(rec.similar(movie, kind), hide_index=True, column_config={
        "title": "Movie", "genres": "Genres",
        "similarity": st.column_config.NumberColumn("Cosine similarity", format="%.3f")})

with tab_perf:
    res = rec.results
    st.subheader("Test-set error (RMSE, lower is better)")
    rmse = pd.Series(res["test_rmse"], name="RMSE")
    c1, c2 = st.columns([1, 1])
    c1.dataframe(rmse.to_frame().style.format("{:.4f}"))
    c2.bar_chart(rmse)
    d = res["dataset"]
    st.caption(
        f"{d['users']} users, {d['movies']} movies. Models trained on {d['train_ratings']:,} ratings and "
        f"evaluated on {d['test_ratings']:,} held-out ratings. RMSE is in rating stars (scale 0.5 to 5)."
    )

with tab_how:
    res = rec.results
    st.markdown(
        f"""
**Collaborative filtering** learns a vector of {res['cf']['factors']} numbers for every user and every movie,
plus a bias for each, so that `global mean + user bias + movie bias + user vector · movie vector` matches the known
ratings. It only uses *who rated what*, never the movie's genre.

**Content-based two-tower network** turns a user's taste profile (average rating per genre) into a vector with one
small neural network, and a movie's features (year, average rating, popularity, genres) into a vector with another.
Their dot product is the predicted rating.

**New users without retraining.** The models were trained once on existing users. For *you*, the collaborative model
solves a small ridge regression for your vector while the movie vectors stay fixed, and the content model simply
computes your genre averages and runs them through the user tower. Both answer instantly.

**Limitations.** The dataset is small (hundreds of users), so the recommendations lean towards popular movies.
The content model sees only genres, year and popularity, so it cannot tell two thrillers apart by plot. Predictions
are for exploring ideas, not for deciding what you will enjoy.
"""
    )

st.divider()
st.caption(
    "Data: MovieLens ml-latest-small (GroupLens Research). F. M. Harper and J. A. Konstan (2015), "
    "The MovieLens Datasets: History and Context, ACM TiiS 5(4). The concepts were learned in the Machine Learning "
    "Specialization (DeepLearning.AI / Stanford Online on Coursera); this code is an independent re-implementation."
)
