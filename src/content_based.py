import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_movies():
    return pd.read_csv("data/movies.csv")

def build_content_model(movies_df):
    
    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(movies_df["genres"])

    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix

def get_recommendations(title, movies_df, similarity_matrix, n=10):
    
    movie_index = movies_df[movies_df["title"] == title].index[0]

    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:n+1]

    movie_indices = [i[0] for i in similarity_scores]

    return movies_df.iloc[movie_indices][
        ["title", "genres"]
    ]

if __name__ == "__main__":
    
    movies = load_movies()

    similarity_matrix = build_content_model(movies)

    recommendations = get_recommendations(
        "Toy Story (1995)",
        movies,
        similarity_matrix
    )

    print(recommendations)

    