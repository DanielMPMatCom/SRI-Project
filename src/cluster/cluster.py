import numpy as np
from ft_processing.ft_procesing import FilmTrustProcessing


class ClousterNBCF:
    def __init__(self):
        preprocessing = FilmTrustProcessing()
        rating, qualified = preprocessing.numpy_user_movie_matrix()

        # Create recommenders
        alpha = 0.01
        r = 8

        