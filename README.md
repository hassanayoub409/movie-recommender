# Movie Recommender

An interactive web app with **two recommender systems written from scratch in NumPy**, trained on the [MovieLens](https://grouplens.org/datasets/movielens/latest/) ratings:

| Engine | Idea | Uses |
|---|---|---|
| **Collaborative filtering** | Learns a hidden vector per user and per movie so that their dot product (plus biases) predicts ratings | Only *who rated what* |
| **Content-based two-tower network** | Two small neural networks turn a user's taste profile and a movie's features into vectors; their dot product is the predicted rating | Genres, year, popularity |

Rate a few movies you know (or click a preset), and both models recommend what to watch next. You can compare their lists side by side, filter by genre, and look up similar movies.

## Run it locally

Requires Python 3.9+.

```bash
git clone https://github.com/hassanayoub409/movie-recommender.git
cd movie-recommender
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On the first launch the app downloads MovieLens from GroupLens and trains both models (well under a minute). After that it starts instantly. To do this step manually: `python train.py`.

Run the tests with `pip install pytest && python -m pytest -q`.

## What's interesting about the implementation

- **No deep learning framework.** Matrix factorization, the two-tower network, backpropagation and the Adam optimizer are all plain NumPy (`src/`). The hand-written gradients are verified against numerical gradients in `tests/`. A side benefit: the whole project installs in seconds and runs anywhere.
- **New users without retraining.** The models are trained once. For a new user, the collaborative model solves a small ridge regression for that user's vector with the movie vectors frozen (closed form), and the content model computes the user's genre averages and runs them through the user tower. Recommendations update instantly as you move a slider.
- **Honest evaluation.** Random 80/20 split of ratings. The test set is never used for tuning: regularization and early stopping use a validation slice of the training data. All statistics used as features (genre averages, movie means) come from the training split only. The app's *Model performance* tab compares against two baselines (global average, and user and movie biases without latent factors) so the gain from each model is visible.
- **Biased matrix factorization.** Prediction is `global mean + user bias + movie bias + user vector · movie vector`, with L2 regularization.

## Project structure

```
app.py               # Streamlit app
train.py             # Download data + train both models
src/data.py          # Download, filtering, train/test split, feature building
src/mf.py            # Matrix factorization (collaborative filtering)
src/two_tower.py     # Two-tower neural network with manual backprop
src/optim.py         # Adam optimizer
src/pipeline.py      # Training and evaluation pipeline
src/recommend.py     # Scoring new users, similar movies
tests/               # Gradient checks and unit tests
```

## Data and license

This project uses the **MovieLens ml-latest-small** dataset (GroupLens Research, University of Minnesota). The data is **not included in this repository**: MovieLens has its own usage terms (including restrictions on redistribution), so `train.py` downloads it directly from GroupLens. Please read the README inside the dataset zip for its terms.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems* 5, 4, Article 19. https://doi.org/10.1145/2827872

## Limitations

- Small dataset (about 600 users), and only movies with at least 10 ratings are used, so results lean towards well-known films.
- The content model only sees genres, year and popularity, so it cannot separate two movies of the same genre by plot or style.
- Predictions are for exploration, not guarantees of what you will enjoy.

## Acknowledgments

The concepts behind this project (collaborative filtering and content-based filtering with neural networks) were learned in the *Machine Learning Specialization* by DeepLearning.AI and Stanford Online on Coursera. The code in this repository was written independently and contains no course materials. This project is not affiliated with or endorsed by Coursera, DeepLearning.AI, Stanford Online or GroupLens.

## License

Code: MIT, see [LICENSE](LICENSE). MovieLens data is covered by its own terms.
