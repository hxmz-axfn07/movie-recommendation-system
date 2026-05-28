import pandas as pd

def load_data(movie_path, rating_path=None, links_path=None):

    movies = pd.read_csv(movie_path)
    ratings = pd.read_csv(rating_path) if rating_path else None
    links = pd.read_csv(links_path) if links_path else None

    movies["genres"] = movies["genres"].str.replace("|"," ", regex=False)

    return movies, ratings, links
