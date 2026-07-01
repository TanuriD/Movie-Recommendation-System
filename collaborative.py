import pandas as pd
from surprise import Dataset, Reader, SVD


# -----------------------------
# Load ratings data
# -----------------------------
def load_data():

    ratings = pd.read_csv("data/ratings.csv")

    reader = Reader(rating_scale=(0.5, 5))

    data = Dataset.load_from_df(
        ratings[['userId', 'movieId', 'rating']],
        reader
    )

    return data, ratings


# -----------------------------
# Train SVD Model
# -----------------------------
def train_model():

    data, ratings = load_data()

    trainset = data.build_full_trainset()

    model = SVD()

    model.fit(trainset)

    return model, ratings


# -----------------------------
# Recommend Top N Movies
# -----------------------------
def recommend_movies(model, user_id, movies_df, ratings_df, n=10):

    # Movies already watched by the user
    watched_movies = ratings_df[
        ratings_df["userId"] == user_id
    ]["movieId"].tolist()

    # All available movies
    all_movies = movies_df["movieId"].tolist()

    # Movies not watched yet
    unseen_movies = [
        movie
        for movie in all_movies
        if movie not in watched_movies
    ]

    predictions = []

    # Predict ratings for unseen movies
    for movie in unseen_movies:

        prediction = model.predict(user_id, movie)

        predictions.append(
            (movie, prediction.est)
        )

    # Sort by predicted rating
    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Get top N movie IDs
    top_movie_ids = [
        movie[0]
        for movie in predictions[:n]
    ]

    # Return movie details
    recommended_movies = movies_df[
        movies_df["movieId"].isin(top_movie_ids)
    ]

    return recommended_movies


# -----------------------------
# Test the program
# -----------------------------
if __name__ == "__main__":

    # Train model
    model, ratings = train_model()

    # Load movie data
    movies = pd.read_csv("data/movies.csv")

    # Get recommendations for User 1
    recommendations = recommend_movies(
        model,
        user_id=1,
        movies_df=movies,
        ratings_df=ratings,
        n=10
    )

    print("\nTop 10 Recommended Movies:\n")

    print(recommendations[["movieId", "title", "genres"]])