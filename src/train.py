import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_data
from src.model import train_model, build_genre_similarity

movies, ratings, links = load_data(
    "data/movies.csv",
    "data/ratings.csv",
    "data/links.csv"
)
model, matrix, movie_to_index = train_model(ratings)

genre_sim = build_genre_similarity(movies)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/knn_model.pkl")
joblib.dump(matrix, "models/matrix.pkl")
joblib.dump(movie_to_index, "models/movie_index.pkl")
joblib.dump(genre_sim, "models/genre_sim.pkl")

print("Training complete.")