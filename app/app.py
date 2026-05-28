import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import CountVectorizer

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

sys.path.append(str(ROOT_DIR))

from src.data_loader import load_data
from src.recommender import hybrid_recommend

movies, _, _ = load_data(DATA_DIR / "movie.csv")
movies = movies.reset_index(drop=True)

model = joblib.load(MODEL_DIR / "knn_model.pkl")
matrix = joblib.load(MODEL_DIR / "matrix.pkl")
movie_to_index = joblib.load(MODEL_DIR / "movie_index.pkl")

genre_matrix = CountVectorizer().fit_transform(movies["genres"].fillna(""))
index_to_movie = {index: movie_id for movie_id, index in movie_to_index.items()}
title_by_movie_id = movies.set_index("movieId")["title"].to_dict()

print("Movie Recommender System")
print("Type 'exit' to quit\n")

while True:
    movie = input("Enter a movie name: ")

    if movie.lower() == "exit":
        print("Goodbye")
        break

    results = hybrid_recommend(
        movie,
        movies,
        movie_to_index,
        model,
        matrix,
        genre_matrix,
        index_to_movie=index_to_movie,
        title_by_movie_id=title_by_movie_id
    )

    if not results:
        continue

    print("\nRecommended Movies:\n")

    for result in results:
        print(result)

    print("\n" + "-" * 40 + "\n")
