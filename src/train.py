import os
import sys
from pathlib import Path
import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

sys.path.append(str(ROOT_DIR))

from src.data_loader import load_data
from src.model import train_model, build_genre_similarity

movies, ratings, links = load_data(
    DATA_DIR / "movie.csv",
    DATA_DIR / "rating.csv",
    DATA_DIR / "link.csv"
)
model, matrix, movie_to_index = train_model(ratings)

genre_sim = build_genre_similarity(movies)

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, MODEL_DIR / "knn_model.pkl")
joblib.dump(matrix, MODEL_DIR / "matrix.pkl")
joblib.dump(movie_to_index, MODEL_DIR / "movie_index.pkl")
joblib.dump(genre_sim, MODEL_DIR / "genre_sim.pkl")

print("Training complete.")
