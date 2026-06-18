import pandas as pd
from surprise import Dataset, Reader, SVD


def load_data():

    ratings = pd.read_csv("data/ratings.csv")

    reader = Reader(rating_scale=(0.5, 5))

    data = Dataset.load_from_df(
        ratings[['userId', 'movieId', 'rating']],
        reader
    )

    return data, ratings


def train_model():

    data, ratings = load_data()

    trainset = data.build_full_trainset()

    model = SVD()

    model.fit(trainset)

    return model, ratings


if __name__ == "__main__":

    model, ratings = train_model()

    print("Model Trained Successfully")