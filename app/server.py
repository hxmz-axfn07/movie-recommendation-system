import sys
import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("API_KEY")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_data
from src.recommender import hybrid_recommend

app = Flask(__name__)
CORS(app)

from src.data_loader import load_data

movies, ratings, links = load_data(
    "data/movies.csv",
    "data/ratings.csv",
    "data/links.csv"
)


model = joblib.load("models/knn_model.pkl")
matrix = joblib.load("models/matrix.pkl")
movie_to_index = joblib.load("models/movie_index.pkl")
genre_sim = joblib.load("models/genre_sim.pkl")


@app.route("/recommend", methods=["POST"])
def recommend():

    data = request.get_json()
    movie_title = data.get("movie", "").strip()

    if movie_title == "":
        return jsonify({"error": "Enter a movie name"})


    movie_row = movies[movies["title"].str.contains(movie_title, case=False, na=False, regex=False)]

    # if no movie found
    if movie_row.empty:
        return jsonify({"error": "Movie not found in dataset"})


    movie_id = movie_row.iloc[0]["movieId"]

    if movie_id not in movie_to_index:
        return jsonify({"error": "Movie not available for recommendation"})


    results = hybrid_recommend(
        movie_title,
        movies,
        movie_to_index,
        model,
        matrix,
        genre_sim
    )
    
    output = []
    
    for title in results:
    
        row = movies[movies["title"] == title]
    
        if row.empty:
            continue
    
        movie_id = row.iloc[0]["movieId"]
    
        link_row = links[links["movieId"] == movie_id]
    
        tmdb_id = None
        if not link_row.empty:
            tmdb_id = int(link_row.iloc[0]["tmdbId"])
    
        output.append({
            "title": title,
            "tmdbId": tmdb_id
        })
    
    return jsonify(output)
    

@app.route("/search", methods=["GET"])
def search():

    query = request.args.get("q","").lower()

    if not query:
        return jsonify([])

    matches = movies[movies["title"].str.lower().str.contains(query)]

    suggestions = matches["title"].head(8).tolist()

    return jsonify(suggestions)

if __name__ == "__main__":
    app.run(debug=True)
