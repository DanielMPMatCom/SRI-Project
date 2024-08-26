class WatchHistory:
    """
    A class to track movies that a user has watched and prevent recommending them again.
    """

    def __init__(self):
        self.watched_movies = set()

    def add_watched_movie(self, movie_id):
        """
        Adds a movie to the user's watch history.

        Args:
            movie_id (str): The ID of the movie that the user has watched.
        """
        self.watched_movies.add(movie_id)

    def is_watched(self, movie_id):
        """
        Checks if a movie has already been watched by the user.

        Args:
            movie_id (str): The ID of the movie to check.

        Returns:
            bool: True if the movie has been watched, False otherwise.
        """
        return movie_id in self.watched_movies

    def filter_watched_movies(self, recommendations):
        """
        Filters out movies that the user has already watched from the recommendations.

        Args:
            recommendations (list): The list of recommended movies.

        Returns:
            list: The filtered list of recommendations.
        """
        return [rec for rec in recommendations if rec['movie_id'] not in self.watched_movies]
