import pandas as pd
import cupy as cp


class MovieLensProcessing:
    ROOT = "./src/ml_processing/"

    def __init__(self, rating_path, delete_rows_every=2) -> None:

        self.rating_csv = rating_path

        columns = ["userId", "movieId", "rating", "time"]
        self._ratings = pd.read_table(
            self.rating_csv, sep="::", header=None, engine="python"
        )
        self._ratings.columns = ["userId", "movieId", "rating", "time"]
        self._ratings = self._ratings.drop(
            self._ratings.index[delete_rows_every - 1 :: delete_rows_every]
        )
        self._ratings.to_csv("ratings.csv", index=False)

    def get_ratings(self) -> pd.DataFrame:
        return self._ratings

    def numpy_user_movie_matrix(self) -> cp.ndarray:
        print("Loading user_movie_matrix ...")
        user_movie_matrix = (
            self._ratings.pivot_table(
                index="userId", columns="movieId", values="rating"
            )
            .fillna(-1)
            .to_numpy()
        )
        print("user_movie_matrix loaded")

        return user_movie_matrix
