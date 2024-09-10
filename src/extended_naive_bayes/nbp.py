import numpy as np


def group_prediction(
    rating: np.ndarray, group: np.ndarray, hp: np.ndarray, q: list[int]
):

    _, movies = rating.shape
    group_prediction_hf = np.zeros((movies, len(q)))

    for item in range(movies):
        for qualified in q:

            hybrid_based = 0
            inverted_hybrid_based = 0

            for g_user in group:
                print("HB", hybrid_based)
                hybrid_based += np.log(hp[g_user, item, qualified])

                if hybrid_based == 0:
                    print("++++++++++++++++++++++++++++++++")
                    print(
                        f"{g_user}, {item}, {qualified}: {hp[g_user, item, qualified]}"
                    )
                    raise

                print("IHB", inverted_hybrid_based)
                inverted_hybrid_based += np.log(1 - hp[g_user, item, qualified])

                if inverted_hybrid_based == 0:
                    print("--------------------------------")
                    print(
                        f"{g_user}, {item}, {qualified}: {hp[g_user, item, qualified]}"
                    )
                    raise

            group_prediction_hf[item, qualified] = (hybrid_based) / (
                hybrid_based + inverted_hybrid_based
            )

    return group_prediction_hf
