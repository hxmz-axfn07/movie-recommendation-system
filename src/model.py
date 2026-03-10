import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def train_model(ratings):

    # count ratings per movie
    rating_counts = ratings.groupby("movieId").size()

    # keep popular movies
    popular_movies = rating_counts[rating_counts >= 50].index

    ratings_filtered = ratings[ratings["movieId"].isin(popular_movies)]

    movie_user_matrix = ratings_filtered.pivot_table(
        index="movieId",
        columns="userId",
        values="rating"
    ).fillna(0)

    matrix = sp.csr_matrix(movie_user_matrix.values)

    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(matrix)

    movie_to_index = {
        movie_id: i for i, movie_id in enumerate(movie_user_matrix.index)
    }

    return model, matrix, movie_to_index


def build_genre_similarity(movies):

    vectorizer = CountVectorizer()

    genre_matrix = vectorizer.fit_transform(movies["genres"])

    genre_similarity = cosine_similarity(genre_matrix)

    return genre_similarity