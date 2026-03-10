import pandas as pd

def load_data(movie_path, rating_path, links_path):

    movies = pd.read_csv(movie_path)
    ratings = pd.read_csv(rating_path)
    links = pd.read_csv(links_path)

    movies["genres"] = movies["genres"].str.replace("|"," ", regex=False)

    return movies, ratings, links