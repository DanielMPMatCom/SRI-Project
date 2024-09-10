import numpy as np


def group_prediction(
    rating: np.ndarray, group: np.ndarray, hp: np.ndarray, qualified_array: "list[int]"
):
    EPSILON = np.finfo(float).eps
    _, movies = rating.shape
    group_prediction_hf = np.zeros((movies, len(qualified_array)))

    for item in range(1):

        for qualified in qualified_array:

            hybrid_based = 0
            inverted_hybrid_based = 0

            for g_user in group:
                if rating[g_user][item] != -1:
                    continue
                print("hp", hp[g_user, item, qualified], g_user, item, qualified)
                hybrid_based += np.log(hp[g_user, item, qualified])

                if hybrid_based == 0:
                    print("++++++++++++++++++++++++++++++++")
                    print(
                        f"User {g_user}, Movie {item}, Qualified {qualified}: Prediction {hp[g_user, item, qualified]}"
                    )
                    raise
                print(
                    "sum",
                    np.sum(
                        [hp[g_user, item, q] for q in qualified_array if q != qualified]
                    ),
                )
                inverted_hybrid_based += np.log(
                    np.sum(
                        [hp[g_user, item, q] for q in qualified_array if q != qualified]
                    )
                )

                if inverted_hybrid_based == 0:
                    print("--------------------------------")
                    print(
                        f"User {g_user}, Movie {item}, Qualified {qualified}: Prediction {1 - hp[g_user, item, qualified]}",
                        hp[g_user, item, qualified],
                    )
                    raise

            group_prediction_hf[item, qualified] = (hybrid_based) / (
                hybrid_based + inverted_hybrid_based
            )
            print(
                "recommnedation",
                group_prediction_hf[item, qualified],
                (hybrid_based),
                inverted_hybrid_based,
                hybrid_based + inverted_hybrid_based,
            )

    return group_prediction_hf
