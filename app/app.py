import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_data
from src.recommender import recommend

# Load dataset

movies, ratings = load_data("data/movies.csv", "data/ratings.csv")

# Load trained model

model = joblib.load("models/knn_model.pkl")
matrix = joblib.load("models/matrix.pkl")
movie_to_index = joblib.load("models/movie_index.pkl")

print("Movie Recommender System")
print("Type 'exit' to quit\n")

while True:
    
    movie = input("Enter a movie name: ")
    
    if movie.lower() == "exit":
        print("Goodbye 👋")
        break
    
    results = recommend(movie, movies, movie_to_index, model, matrix)
    
    if not results:
        continue
    
    print("\nRecommended Movies:\n")
    
    for r in results:
        print(r)
    
    print("\n" + "-"*40 + "\n")
    