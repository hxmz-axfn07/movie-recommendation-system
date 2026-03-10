from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_content_model(movies):

    movies["genres"] = movies["genres"].str.replace("|", " ")

    tfidf = TfidfVectorizer(stop_words="english")

    genre_matrix = tfidf.fit_transform(movies["genres"])

    similarity = cosine_similarity(genre_matrix)

    return similarity