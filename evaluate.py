from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import cross_validate

import pandas as pd


ratings = pd.read_csv("data/ratings.csv")

reader = Reader(rating_scale=(0.5, 5))

data = Dataset.load_from_df(
    ratings[['userId', 'movieId', 'rating']],
    reader
)

model = SVD()

results = cross_validate(
    model,
    data,
    measures=['RMSE', 'MAE'],
    cv=5,
    verbose=True
)

print(results)