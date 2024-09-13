import numpy as np

# The purpose of this class is to implement the Group Recommendation algorithm.
# User to Group Similarity (UGS) is calculated using the Jaccard index.

class GroupRecomendation:

    def __init__(self, rating, group):
        self.rating = rating
        self.rating_normalized = self.normalized_rating()

        self.movie_users = [
            self.rating[:, movie] != -1 for movie in range(self.rating.shape[1])
        ]
        self.user_movies = [
            self.rating[user, :] != -1 for user in range(self.rating.shape[0])
        ]

        self.pi = [self.unpopular(movie) for movie in range(self.rating.shape[1])]

        self.set_group(group)

    def set_group(self, group):
        self.group = group
        self.movie_users_in_group = [
            [user for user in group if self.rating[user][movie] != -1]
            for movie in range(self.rating.shape[1])
        ]
        self.group_movies = self._all_group_movies()
        self.group_movie = [
            [u for u in self.group if self.rating[u, movie] != -1]
            for movie in range(self.rating.shape[1])
        ]

    def unpopular(self, movie: int):  # pi
        """
        Calculate the unpopularity score of a movie based on the ratings.

        Parameters:
        rating (np.ndarray): The array of ratings for each movie by each user.
        movie (int): The index of the movie for which the unpopularity score is calculated.

        Returns:
        float: The unpopularity score of the movie, ranging from 0 to 1.
        """
        return 1 - np.sum(self.rating[:, movie] != -1) / self.rating.shape[0]

    def _all_group_movies(self):
        """
        Returns a boolean array indicating whether each movie in the rating matrix has been rated by any user in the given group.

        Parameters:
        rating (np.ndarray): The rating matrix where each row represents a user and each column represents a movie.
        group (np.ndarray): The array of user IDs in the group.

        Returns:
        np.ndarray: A boolean array of shape (rating.shape[1],) indicating whether each movie has been rated by any user in the group.
        """
        group_movies = np.full(self.rating.shape[1], False)
        for u in self.group:
            group_movies = np.logical_or(group_movies, self.rating[u, :] != -1)
        return group_movies

    def interception_movies_group_user(self, user: int):
        """
        Returns a boolean array indicating whether each movie in the group is rated by the given user.

        Parameters:
        rating (np.ndarray): The rating matrix.
        group (np.ndarray): The group of movies.
        user (int): The user index.

        Returns:
        np.ndarray: A boolean array indicating whether each movie in the group is rated by the given user.
        """

        group_movies = self.group_movies
        user_movies = self.user_movies[user]

        return np.logical_and(user_movies, group_movies)

    def user_group_similarity(self, user: int):  # Xgu Jacard index
        """
        Calculates the similarity between a user and a group based on their movie ratings.

        Parameters:
            rating (np.ndarray): 2D array representing the movie ratings of all users.
            group (np.ndarray): 1D array representing the group of users.
            user (int): Index of the user for which the similarity is calculated.

        Returns:
            float: The similarity between the user and the group, measured using the Jaccard index.
        """
        group_movies = self.group_movies
        user_movies = self.user_movies[user]

        interception = self.interception_movies_group_user(user=user)
        union = np.logical_or(user_movies, group_movies)

        return np.sum(
            [self.pi[i] for i in range(self.rating.shape[1]) if interception[i]]
        ) / np.sum([self.pi[i] for i in range(self.rating.shape[1]) if union[i]])

    def user_singularity(self, user: int, movie: int):
        """
        Calculate the singularity of a user for a specific movie.

        Parameters:
        rating (np.ndarray): The rating matrix.
        user (int): The index of the user.
        movie (int): The index of the movie.

        Returns:
        float: The singularity of the user for the movie.
        """
        movie_users = self.movie_users[movie]

        return np.sum(
            np.logical_and(
                movie_users, self.rating[:, movie] != self.rating[user, movie]
            )
        ) / np.sum(movie_users)

    def item_group_singularity(self, movie: int):
        """
        Calculate the singularity of a movie within a group of users.

        Parameters:
        rating (np.ndarray): The rating matrix of shape (num_users, num_movies).
        group (np.ndarray): The array of user indices representing the group.
        movie (int): The index of the movie.

        Returns:
        float: The singularity of the movie within the group. If there are no users in the group who have rated the movie, returns 0.
        """

        movie_users_in_group = self.movie_users_in_group[movie]

        return (
            np.power(
                np.prod(
                    [
                        self.user_singularity(user=u, movie=movie)
                        for u in movie_users_in_group
                    ]
                ),
                (1 / len(movie_users_in_group)),
            )
            if len(movie_users_in_group) > 0
            else 0
        )

    def normalized_rating(self):
        """
        Normalize the given rating array.

        Parameters:
        rating (np.ndarray): The input rating array.

        Returns:
        np.ndarray: The normalized rating array.
        """
        rating_min = np.min(self.rating[self.rating != -1])
        rating_max = np.max(self.rating[self.rating != -1])
        normalized = (self.rating - rating_min) / (rating_max - rating_min)
        normalized[self.rating == -1] = -1  # Restaurar los valores -1
        return normalized

    def mean_square_difference_group_rating(self, user: int, movie: int):
        """
        Calculate the mean square difference of the group ratings for a specific movie.

        Parameters:
            rating_normalized (np.ndarray): The normalized ratings matrix.
            group (np.ndarray): The group of users.
            user (int): The user for whom the mean square difference is calculated.
            movie (int): The movie for which the mean square difference is calculated.

        Returns:
            float: The mean square difference of the group ratings for the specified movie.
            None: If the user's rating or the group's ratings for the movie are missing.
        """
        group_movie = self.group_movie[movie]

        if self.rating_normalized[user, movie] == -1 or len(group_movie) == 0:
            return None

        return np.sum(
            [
                np.power(
                    self.rating_normalized[u, movie]
                    - self.rating_normalized[user, movie],
                    2,
                )
                for u in group_movie
            ]
        ) / len(group_movie)

    def singularity_dot(self, user: int, movie: int):
        """
        Calculates the singularity dot product for a given user and movie.

        Parameters:
        - rating (np.ndarray): The rating matrix.
        - group (np.ndarray): The group matrix.
        - user (int): The user index.
        - movie (int): The movie index.

        Returns:
        - float: The singularity dot product.
        """
        group_movie = self.group_movie[movie]

        return (
            self.user_singularity(user=user, movie=movie)
            * self.item_group_singularity(movie=movie)
            * len(group_movie)
        )

    def smgu(
        self,
        user: int,
        alpha: float,
    ):
        """
        Calculate the SMGU (Similarity Measure Group User) value for a given user in a group.

        Parameters:
            rating (np.ndarray): The rating matrix of shape (num_users, num_movies).
            normalized_rating (np.ndarray): The normalized rating matrix of shape (num_users, num_movies).
            group (np.ndarray): The group matrix of shape (num_users, num_groups).
            user (int): The index of the user for whom to calculate the SMGU value.
            alpha (float): The weight parameter for balancing the user-group similarity and the mean square difference.

        Returns:
            float: The SMGU value for the given user in the group.
        """
        interception = self.interception_movies_group_user(user=user)

        n = 0
        d = 0

        for i in range(self.rating.shape[1]):
            if interception[i]:
                weight = self.singularity_dot(user=user, movie=i)
                n += weight * self.mean_square_difference_group_rating(
                    user=user, movie=i
                )

                d += weight
        y = 1 - n / d
        x = self.user_group_similarity(user=user)

        return np.power(x, alpha) * np.power(y, 1 - alpha)
