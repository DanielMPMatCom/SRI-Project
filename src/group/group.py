import numpy as np


def unpopular(rating: np.ndarray, movie: int):  # pi
    """
    Calculate the unpopularity score of a movie based on the ratings.

    Parameters:
    rating (np.ndarray): The array of ratings for each movie by each user.
    movie (int): The index of the movie for which the unpopularity score is calculated.

    Returns:
    float: The unpopularity score of the movie, ranging from 0 to 1.
    """
    return 1 - np.sum(rating[:, movie] != -1) / rating.shape[0]


def all_group_movies(rating: np.ndarray, group: np.ndarray):
    """
    Returns a boolean array indicating whether each movie in the rating matrix has been rated by any user in the given group.

    Parameters:
    rating (np.ndarray): The rating matrix where each row represents a user and each column represents a movie.
    group (np.ndarray): The array of user IDs in the group.

    Returns:
    np.ndarray: A boolean array of shape (rating.shape[1],) indicating whether each movie has been rated by any user in the group.
    """
    group_movies = np.full(rating.shape[1], False)
    for u in group:
        group_movies = np.logical_or(group_movies, rating[u, :] != -1)
    return group_movies


def interception_movies_group_user(rating: np.ndarray, group: np.ndarray, user: int):
    """
    Returns a boolean array indicating whether each movie in the group is rated by the given user.

    Parameters:
    rating (np.ndarray): The rating matrix.
    group (np.ndarray): The group of movies.
    user (int): The user index.

    Returns:
    np.ndarray: A boolean array indicating whether each movie in the group is rated by the given user.
    """
    group_movies = all_group_movies(rating=rating, group=group)
    user_movies = rating[user, :] != -1
    return np.logical_and(user_movies, group_movies)


def user_group_similarity(
    rating: np.ndarray, group: np.ndarray, user: int
):  # Xgu Jacard index
    """
    Calculates the similarity between a user and a group based on their movie ratings.

    Parameters:
        rating (np.ndarray): 2D array representing the movie ratings of all users.
        group (np.ndarray): 1D array representing the group of users.
        user (int): Index of the user for which the similarity is calculated.

    Returns:
        float: The similarity between the user and the group, measured using the Jaccard index.
    """
    group_movies = all_group_movies(rating=rating, group=group)
    user_movies = rating[user, :] != -1

    interception = interception_movies_group_user(rating=rating, group=group, user=user)
    union = np.logical_or(user_movies, group_movies)

    return np.sum(
        [
            unpopular(rating=rating, movie=i)
            for i in range(rating.shape[1])
            if interception[i]
        ]
    ) / np.sum(
        [unpopular(rating=rating, movie=i) for i in range(rating.shape[1]) if union[i]]
    )


def user_singularity(rating: np.ndarray, user: int, movie: int):
    """
    Calculate the singularity of a user for a specific movie.

    Parameters:
    rating (np.ndarray): The rating matrix.
    user (int): The index of the user.
    movie (int): The index of the movie.

    Returns:
    float: The singularity of the user for the movie.
    """
    movie_users = rating[:, movie] != -1
    return np.sum(
        np.logical_and(movie_users, rating[:, movie] != rating[user, movie])
    ) / np.sum(movie_users)


def item_group_singularity(rating: np.ndarray, group: np.ndarray, movie: int):
    """
    Calculate the singularity of a movie within a group of users.

    Parameters:
    rating (np.ndarray): The rating matrix of shape (num_users, num_movies).
    group (np.ndarray): The array of user indices representing the group.
    movie (int): The index of the movie.

    Returns:
    float: The singularity of the movie within the group. If there are no users in the group who have rated the movie, returns 0.
    """

    movie_users_in_group = [user for user in group if rating[user][movie] != -1]

    return (
        np.power(
            np.prod(
                [
                    user_singularity(rating=rating, user=u, movie=movie)
                    for u in movie_users_in_group
                ]
            ),
            (1 / len(movie_users_in_group)),
        )
        if len(movie_users_in_group) > 0
        else 0
    )


def normalized_rating(rating: np.ndarray):
    """
    Normalize the given rating array.

    Parameters:
    rating (np.ndarray): The input rating array.

    Returns:
    np.ndarray: The normalized rating array.
    """
    rating_min = np.min(rating[rating != -1])
    rating_max = np.max(rating[rating != -1])
    normalized = (rating - rating_min) / (rating_max - rating_min)
    normalized[rating == -1] = -1  # Restaurar los valores -1
    return normalized


def mean_square_difference_group_rating(
    rating_normalized: np.ndarray, group: np.ndarray, user: int, movie: int
):
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
    group_movie = [u for u in group if rating_normalized[u, movie] != -1]

    if rating_normalized[user, movie] == -1 or len(group_movie) == 0:
        return None

    return np.sum(
        [
            np.power(rating_normalized[u, movie] - rating_normalized[user, movie], 2)
            for u in group_movie
        ]
    ) / len(group_movie)


def singularity_dot(rating: np.ndarray, group: np.ndarray, user: int, movie: int):
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
    movie_group = [u for u in group if rating[u, movie] != -1]
    return (
        user_singularity(rating=rating, user=user, movie=movie)
        * item_group_singularity(rating=rating, group=group, movie=movie)
        * len(movie_group)
    )


def smgu(
    rating: np.ndarray,
    normalized_rating: np.ndarray,
    group: np.ndarray,
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
    interception = interception_movies_group_user(rating=rating, group=group, user=user)

    n = 0
    d = 0

    for i in range(rating.shape[1]):
        if interception[i]:
            weight = singularity_dot(rating=rating, group=group, user=user, movie=i)
            n += weight * mean_square_difference_group_rating(
                rating_normalized=normalized_rating, group=group, user=user, movie=i
            )

            d += weight
    y = 1 - n / d
    x = user_group_similarity(rating=rating, group=group, user=user)

    return np.power(x, alpha) * np.power(y, 1 - alpha)
