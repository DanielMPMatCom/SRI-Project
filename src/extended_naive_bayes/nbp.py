import numpy as np

def group_prediction(
    rating: np.ndarray,
    group: np.ndarray,
    hp: np.ndarray,
    r: int,
):
    
    _, movies = rating.shape
    group_prediction_hf = np.zeros(movies, r)

    for item in range(movies):
        for qualified in range(r):

            hybrid_based = 1
            inverted_hybrid_based = 1

            for g_user in group:
                hybrid_based *= hp[g_user, item, qualified]
                inverted_hybrid_based *= 1 - hp[g_user, item, qualified]

            group_prediction_hf[item, qualified] = (hybrid_based) / (hybrid_based + inverted_hybrid_based)

    return group_prediction_hf
                     

