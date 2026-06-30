import pandas as pd


def load_data():
    """
    Load MovieLens datasets
    """

    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")

    return movies, ratings


def clean_data(movies, ratings):
    """
    Basic cleaning
    """

    movies = movies.drop_duplicates()
    ratings = ratings.drop_duplicates()

    movies = movies.fillna("")
    ratings = ratings.dropna()

    return movies, ratings


def merge_data(movies, ratings):
    """
    Merge movies and ratings datasets
    """

    merged_df = pd.merge(
        ratings,
        movies,
        on="movieId",
        how="left"
    )

    return merged_df
def get_movie_statistics(merged_df):
    
    stats = merged_df.groupby(
        "title"
    )["rating"].agg(
        ["count", "mean"]
    )

    stats = stats.rename(
        columns={
            "count": "rating_count",
            "mean": "average_rating"
        }
    )

    return stats

def preprocess_data():
    """
    Complete preprocessing pipeline
    """

    movies, ratings = load_data()

    movies, ratings = clean_data(
        movies,
        ratings
    )

    merged_df = merge_data(
        movies,
        ratings
    )

    return merged_df


if __name__ == "__main__":
    
    df = preprocess_data()

    print(df.head())

    movie_stats = get_movie_statistics(df)

    print("\nTop Movies:")
    print(
        movie_stats.sort_values(
            by="rating_count",
            ascending=False
        ).head()
    )

    print("Dataset Shape:", df.shape)
    print(df.head())