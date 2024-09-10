import numpy as np
from extended_naive_bayes.nbp import group_prediction
from ft_processing.ft_processing import FilmTrustProcessing
from nbcf.nbcf_opt import NBCF

import time
from cluster.cluster import ClusterNBCF


def main():

    preprocessing = FilmTrustProcessing()
    rating, qualified_array = preprocessing.numpy_user_movie_matrix()
    NBCF_instance = NBCF(
        rating=rating, alpha=0.01, r=8, qualified_array=qualified_array
    )

    cluster_instance = ClusterNBCF(preprocessing, NBCF_instance)


if __name__ == "__main__":
    main()
