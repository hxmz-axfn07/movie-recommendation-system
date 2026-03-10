import difflib


def find_movie(title, movies):
    """
    Find the closest movie title using fuzzy matching.
    """

    titles = movies["title"].tolist()

    match = difflib.get_close_matches(title, titles, n=1, cutoff=0.7)

    if match:
        return movies[movies["title"] == match[0]]

    return None


def hybrid_recommend(movie_title,
                     movies,
                     movie_to_index,
                     model,
                     matrix,
                     genre_sim,
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

    index_to_movie = {i: mid for mid, i in movie_to_index.items()}

    collab_titles = []

    for idx in collab_indices:
        mid = index_to_movie[idx]
        title = movies[movies["movieId"] == mid]["title"].values[0]
        collab_titles.append(title)

    # ------------------------
    # Genre similarity
    # ------------------------

    genre_idx = movie_match.index[0]

    genre_scores = list(enumerate(genre_sim[genre_idx]))

    genre_scores = sorted(
        genre_scores,
        key=lambda x: x[1],
        reverse=True
    )[1:n+1]

    genre_titles = []

    for idx, score in genre_scores:
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
