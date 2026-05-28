import difflib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_movie(title, movies):
    """
    Find the closest movie title using fuzzy matching.
    """

    title = title.strip()
    titles = movies["title"].tolist()
    normalized_title = title.lower()

    exact_match = movies[movies["title"].str.lower() == normalized_title]
    if not exact_match.empty:
        return exact_match

    contains_match = movies[
        movies["title"].str.contains(title, case=False, na=False, regex=False)
    ]
    if not contains_match.empty:
        return contains_match

    match = difflib.get_close_matches(title, titles, n=1, cutoff=0.7)

    if match:
        return movies[movies["title"] == match[0]]

    return None


def hybrid_recommend(movie_title,
                     movies,
                     movie_to_index,
                     model,
                     matrix,
                     genre_features,
                     index_to_movie=None,
                     title_by_movie_id=None,
                     n=10):

    # find closest title
    movie_match = find_movie(movie_title, movies)

    if movie_match is None or movie_match.empty:
        return {"error": "Movie not found in dataset"}
    
    movie_id = movie_match.iloc[0]["movieId"]

    # movie might not exist in collaborative matrix
    if movie_id not in movie_to_index:
        return []

    movie_idx = movie_to_index[movie_id]

    # ------------------------
    # Collaborative filtering
    # ------------------------

    distances, indices = model.kneighbors(matrix[movie_idx], n_neighbors=n+1)

    collab_indices = indices.flatten()[1:]

    if index_to_movie is None:
        index_to_movie = {i: mid for mid, i in movie_to_index.items()}
    if title_by_movie_id is None:
        title_by_movie_id = movies.set_index("movieId")["title"].to_dict()

    collab_titles = []

    for idx in collab_indices:
        mid = index_to_movie[idx]
        title = title_by_movie_id.get(mid)
        if title:
            collab_titles.append(title)

    # ------------------------
    # Genre similarity
    # ------------------------

    genre_idx = movie_match.index[0]

    if hasattr(genre_features, "tocsr"):
        genre_scores = cosine_similarity(
            genre_features[genre_idx],
            genre_features
        ).ravel()
    else:
        genre_scores = np.asarray(genre_features[genre_idx]).ravel()

    candidate_count = min(n + 1, len(genre_scores))
    top_indices = np.argpartition(-genre_scores, candidate_count - 1)[:candidate_count]
    top_indices = top_indices[np.argsort(-genre_scores[top_indices])]

    genre_titles = []

    for idx in top_indices:
        if idx != genre_idx:
            genre_titles.append(movies.iloc[idx]["title"])

    # ------------------------
    # Hybrid merge
    # ------------------------

    combined = collab_titles + genre_titles
    
    seen = set()
    results = []
    
    for movie in combined:
        if movie not in seen:
            seen.add(movie)
            results.append(movie)
    
    return results[:n]
