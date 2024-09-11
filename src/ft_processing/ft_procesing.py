import pandas as pd
import numpy as np


class FilmTrustProcessing:
    rating_csv = "./datasets/film-trust/ratings.txt"

    def __init__(self) -> None:
        self._ratings = pd.read_table(
            self.rating_csv, sep=" ", header=None, engine="python"
        )
        print(len(self._ratings))
        self._ratings.columns = ["userId", "movieId", "rating"]

        self._ratings["userId"], user_index = pd.factorize(self._ratings["userId"])
        self._ratings["movieId"], movie_index = pd.factorize(self._ratings["movieId"])

        self._ratings.to_csv("ratings.csv", index=False)

    def get_ratings(self) -> pd.DataFrame:
        return self._ratings

    def separate_data_for_test(self, percent: float = 0.2):
        self._ratings = self._ratings.sample(frac=1, random_state=42)
        n = int(len(self._ratings) * percent)
        
        test = self._ratings[:n].to_numpy()
        test = np.array([[int(u), int(m), r * 2] for u, m, r in test])

        train = self._ratings[n:].to_numpy()
        train = np.array([[int(u), int(m), r * 2] for u, m, r in train])

        return train, test

    def numpy_user_movie_matrix(self, remove_data=None) -> np.ndarray:
        print("Loading user_movie_matrix ...")
        user_movie_matrix = (
            self._ratings.pivot_table(
                index="userId", columns="movieId", values="rating"
            )
            .fillna(-0.5)
            .to_numpy()
        )
        user_movie_matrix *= 2
        if remove_data is not None:
            user_movie_matrix = self.remove_test_data_from_matrix(
                user_movie_matrix, remove_data
            )

        print("user_movie_matrix loaded")
        return (user_movie_matrix, range(8))

    def remove_test_data_from_matrix(self, user_movie_matrix, test_data):
        for u, m, r in test_data:
            u, m = int(u), int(m)
            user_movie_matrix[u][m] = -1
        return user_movie_matrix
