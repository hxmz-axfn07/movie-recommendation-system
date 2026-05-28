import os
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

sys.path.append(str(ROOT_DIR))

from src.data_loader import load_data
from src.recommender import hybrid_recommend

load_dotenv(ROOT_DIR / ".env")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

app = Flask(__name__)
CORS(app)

movies, _, links = load_data(
    DATA_DIR / "movie.csv",
    links_path=DATA_DIR / "link.csv"
)
movies = movies.reset_index(drop=True)
movies["search_title"] = movies["title"].str.lower()

model = joblib.load(MODEL_DIR / "knn_model.pkl")
matrix = joblib.load(MODEL_DIR / "matrix.pkl")
movie_to_index = joblib.load(MODEL_DIR / "movie_index.pkl")

genre_matrix = CountVectorizer().fit_transform(movies["genres"].fillna(""))
index_to_movie = {index: movie_id for movie_id, index in movie_to_index.items()}
title_by_movie_id = movies.set_index("movieId")["title"].to_dict()

links = links.dropna(subset=["tmdbId"])
tmdb_by_movie_id = links.set_index("movieId")["tmdbId"].astype(int).to_dict()


@lru_cache(maxsize=1024)
def fetch_movie_details(tmdb_id):
    if not TMDB_API_KEY:
        return {"error": "TMDB_API_KEY is not configured"}

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    response = requests.get(
        url,
        params={"api_key": TMDB_API_KEY},
        timeout=5
    )
    response.raise_for_status()
    return response.json()


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(silent=True) or {}
    movie_title = data.get("movie", "").strip()

    if not movie_title:
        return jsonify({"error": "Enter a movie name"}), 400

    movie_row = movies[
        movies["title"].str.contains(movie_title, case=False, na=False, regex=False)
    ]

    if movie_row.empty:
        return jsonify({"error": "Movie not found in dataset"}), 404

    movie_id = movie_row.iloc[0]["movieId"]

    if movie_id not in movie_to_index:
        return jsonify({"error": "Movie not available for recommendation"}), 404

    results = hybrid_recommend(
        movie_title,
        movies,
        movie_to_index,
        model,
        matrix,
        genre_matrix,
        index_to_movie=index_to_movie,
        title_by_movie_id=title_by_movie_id
    )

    if isinstance(results, dict) and "error" in results:
        return jsonify(results), 404

    output = []

    for title in results:
        row = movies[movies["title"] == title]
        if row.empty:
            continue

        movie_id = row.iloc[0]["movieId"]
        tmdb_id = tmdb_by_movie_id.get(movie_id)

        output.append({
            "title": title,
            "tmdbId": tmdb_id
        })

    return jsonify(output)


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip().lower()

    if not query:
        return jsonify([])

    matches = movies[movies["search_title"].str.contains(query, na=False, regex=False)]
    suggestions = matches["title"].head(8).tolist()

    return jsonify(suggestions)


@app.route("/movie/<int:tmdb_id>")
def movie_details(tmdb_id):
    try:
        return jsonify(fetch_movie_details(tmdb_id))
    except requests.RequestException:
        return jsonify({"error": "Unable to fetch movie details"}), 502


if __name__ == "__main__":
    app.run(debug=True)
