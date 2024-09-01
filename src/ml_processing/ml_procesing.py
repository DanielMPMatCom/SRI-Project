import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from the specified path.

    Parameters:
        path (str): The path to the MovieLens dataset.

    Returns:
        pd.DataFrame: The dataset.
    """
    return pd.read_csv(path)


class MovieLensProcessing:
    ROOT = "./src/ml_processing/"

    def __init__(self, rating_path) -> None:

        self.rating_csv = rating_path
        self._ratings = load_data(rating_path)

        columns = ["userId", "movieId", "rating", "time"]
        self._ratings = pd.read_table(
            self.rating_csv, sep="::", header=None, names=columns, engine="python"
        )
        print(self._ratings)

    def get_ratings(self) -> pd.DataFrame:
        return self._ratings

    def numpy_user_movie_matrix(self) -> np.ndarray:
        print("Loading user_movie_matrix")
        user_movie_matrix = (
            self._ratings.pivot_table(
                index="userId", columns="movieId", values="rating"
            )
            .fillna(-1)
            .to_numpy()
        )
        print("user_movie_matrix loaded")
        print(user_movie_matrix)
        return user_movie_matrix


myClass = MovieLensProcessing(
    rating_path="./datasets/ml-1m/ratings.dat",
)

A = myClass.numpy_user_movie_matrix()
