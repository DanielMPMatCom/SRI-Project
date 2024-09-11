import pandas as pd
import numpy as np


class FilmTrustProcessing:
    """
    Class for processing FilmTrust data.
    """

    rating_csv = "./datasets/film-trust/ratings.txt"

    def __init__(self) -> None:
        """
        Initialize the FilmTrustProcessing class.
        """

        self._ratings = pd.read_table(
            self.rating_csv, sep=" ", header=None, engine="python"
        )
        print(len(self._ratings))
        self._ratings.columns = ["userId", "movieId", "rating"]

        self._ratings["userId"], user_index = pd.factorize(self._ratings["userId"])
        self._ratings["movieId"], movie_index = pd.factorize(self._ratings["movieId"])

        self._ratings.to_csv("ratings.csv", index=False)

    def get_ratings(self) -> pd.DataFrame:
        """
        Get the ratings DataFrame.

        Returns:
            pd.DataFrame: The ratings DataFrame.
        """
        return self._ratings

    def separate_data_for_test(self, percent: float = 0.2):
        """
        Separate the data into training and testing sets.

        Args:
            percent (float, optional): The percentage of data to use for testing. Defaults to 0.2.

        Returns:
            tuple: A tuple containing the training and testing sets.
        """
        self._ratings = self._ratings.sample(frac=1, random_state=42)
        n = int(len(self._ratings) * percent)

        test = self._ratings[:n].to_numpy()
        test = np.array([[int(u), int(m), r * 2] for u, m, r in test])

        train = self._ratings[n:].to_numpy()
        train = np.array([[int(u), int(m), r * 2] for u, m, r in train])

        return train, test

    def numpy_user_movie_matrix(self, remove_data=None) -> np.ndarray:
        """
        Generate a user-movie matrix using numpy.

        Args:
            remove_data (optional): Data to be removed from the matrix. Defaults to None.

        Returns:
            tuple: A tuple containing the user-movie matrix and a range object.

        Raises:
            None

        """
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
        """
        Removes the test data from the user-movie matrix.

        Args:
            user_movie_matrix (list): The user-movie matrix.
            test_data (list): The test data containing user, movie, and rating.

        Returns:
            list: The updated user-movie matrix with test data removed.
        """
        for u, m, r in test_data:
            u, m = int(u), int(m)
            user_movie_matrix[u][m] = -1
        return user_movie_matrix
