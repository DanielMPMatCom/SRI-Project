import sys

sys.path.append("src")
from group.group import *


def attempt(value, expected, test_name=""):
    try:
        assert value == expected
        return "\033[92m" + f"✅ Test {test_name} passed!" + "\033[0m"
    except AssertionError as e:
        raise AssertionError(
            f"❌ Test failed! Expected {expected}, but got {value}"
        ) from e


def test_group_of_users():
    rating = np.array(
        [
            [-1, 1, -1, 1, 1, -1, 0, -1, 0, -1, -1, 0],
            [0, 1, -1, 0, 1, -1, 1, -1, 0, -1, -1, -1],
            [-1, 1, -1, 0, 0, -1, 0, 0, -1, -1, -1, 0],
            [-1, 1, 1, 1, 1, 1, 1, -1, 1, 0, 1, 1],
            [1, 0, -1, 1, -1, 1, -1, -1, 1, 0, -1, 1],
            [0, 1, -1, 0, 1, -1, 0, 0, 0, -1, -1, 0],
            [1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1],
            [1, 1, -1, 0, 0, 0, 0, -1, 0, -1, 1, 0],
            [-1, 1, -1, 1, 0, -1, 1, -1, -1, 1, 1, -1],
        ]
    )

    groups = np.array([[0, 1, 2]])
    group = groups[0]

    gr = GroupRecomendation(rating, group)

    attempt(
        [gr.unpopular(i) for i in range(rating.shape[1])],
        [
            0.4444444444444444,
            0.11111111111111116,
            0.8888888888888888,
            0.11111111111111116,
            0.2222222222222222,
            0.5555555555555556,
            0.2222222222222222,
            0.6666666666666667,
            0.33333333333333337,
            0.5555555555555556,
            0.6666666666666667,
            0.33333333333333337,
        ],
    )

    # Test 2 User group similarity
    attempt(
        [
            gr.user_group_similarity(user=u)
            for u in range(rating.shape[0])
            if u not in group
        ],
        [0.2608695652173913, 0.375, 1.0, 0.3125, 0.484848484848485, 0.1818181818181818],
        "user_group_similarity",
    )

    expected = [
        [
            (2, 0.125),
            (3, 0.0),
            (4, 0.5),
            (5, 0.42857142857142855),
            (6, 0.25),
            (7, 0.5714285714285714),
            (9, 0.6666666666666666),
            (10, 0.5),
            (11, 0.0),
            (12, 0.6666666666666666),
        ],
        [
            (1, 0.4),
            (2, 0.875),
            (4, 0.5),
            (6, 0.25),
            (9, 0.6666666666666666),
            (10, 0.5),
            (12, 0.6666666666666666),
        ],
        [
            (1, 0.6),
            (2, 0.125),
            (4, 0.5),
            (5, 0.42857142857142855),
            (7, 0.42857142857142855),
            (8, 0.3333333333333333),
            (9, 0.3333333333333333),
            (12, 0.3333333333333333),
        ],
        [(1, 0.4), (6, 0.25), (8, 0.6666666666666666), (10, 0.5)],
        [
            (1, 0.4),
            (2, 0.125),
            (4, 0.5),
            (5, 0.5714285714285714),
            (6, 0.75),
            (7, 0.42857142857142855),
            (9, 0.3333333333333333),
            (11, 0.0),
            (12, 0.3333333333333333),
        ],
        [
            (2, 0.125),
            (4, 0.5),
            (5, 0.5714285714285714),
            (7, 0.5714285714285714),
            (10, 0.5),
            (11, 0.0),
        ],
    ]

    # Test 3 User singularity
    cont = 0
    for u in range(rating.shape[0]):
        if u not in group:
            attempt(
                [
                    (i + 1, gr.user_singularity(user=u, movie=i))
                    for i in range(rating.shape[1])
                    if rating[u, i] != -1
                ],
                expected[cont],
                "user_singularity",
            )
            cont += 1

    # Test 4 Item group singularity
    attempt(
        [
            gr.item_group_singularity(movie=i)
            for i in range(rating.shape[1])
        ],
        [
            0.6,
            0.12500000000000003,
            0,
            0.5,
            0.4717038926992324,
            0,
            0.47170389269923235,
            0.3333333333333333,
            0.3333333333333333,
            0,
            0,
            0.3333333333333333,
        ],
        "item_group_singularity",
    )

    # # # Test 5 Normalized rating
    # # attempt(
    # #     normalized_rating(rating),
    # #     normalized_rating(rating),
    # #     "normalized_rating",
    # # )


    # Test 6 Mean square difference group rating

    expected = [
        [
            None,
            0.0,
            None,
            0.6666666666666666,
            0.3333333333333333,
            None,
            0.6666666666666666,
            None,
            1.0,
            None,
            None,
            1.0,
        ],
        [
            1.0,
            1.0,
            None,
            0.6666666666666666,
            None,
            None,
            None,
            None,
            1.0,
            None,
            None,
            1.0,
        ],
        [
            0.0,
            0.0,
            None,
            0.3333333333333333,
            0.3333333333333333,
            None,
            0.3333333333333333,
            0.0,
            0.0,
            None,
            None,
            0.0,
        ],
        [1.0, None, None, None, None, None, None, 1.0, None, None, None, None],
        [
            1.0,
            0.0,
            None,
            0.3333333333333333,
            0.6666666666666666,
            None,
            0.3333333333333333,
            None,
            0.0,
            None,
            None,
            0.0,
        ],
        [
            None,
            0.0,
            None,
            0.6666666666666666,
            0.6666666666666666,
            None,
            0.6666666666666666,
            None,
            None,
            None,
            None,
            None,
        ],
        [0.2608695652173913, 0.375, 1.0, 0.3125, 0.484848484848485, 0.1818181818181818],
    ]
    cont = 0
    for u in range(rating.shape[0]):
        if u not in group:
            attempt(
                [
                    gr.mean_square_difference_group_rating(
                        user=u,
                        movie=i,
                    )
                    for i in range(rating.shape[1])
                ],
                expected[cont],
                "mean_square_difference_group_rating",
            )
            cont += 1

    # Test 7 Final similarity (SMGU)
    expected = [
        [0.2608695652173913, 0.375, 1.0, 0.3125, 0.484848484848485, 0.1818181818181818],
        [
            0.2730372810649739,
            0.27800793149820285,
            0.9386673898125864,
            0.0,
            0.5059460006895177,
            0.21359159945550824,
        ],
        [
            0.28577253459685525,
            0.2061024266024252,
            0.881096468697574,
            0.0,
            0.5279615459532919,
            0.2509175424687925,
        ],
        [
            0.2991017974225909,
            0.15279495812400107,
            0.8270565224454389,
            0.0,
            0.5509350674291535,
            0.29476633575045147,
        ],
        [
            0.3130527758646582,
            0.11327522733708817,
            0.7763309871513349,
            0.0,
            0.5749082501361167,
            0.34627787215218064,
        ],
    ]
    cont = 0
    for alpha in [0, 0.25, 0.5, 0.75, 1]:
        attempt(
            [
                gr.smgu(
                    user=u,
                    alpha=1 - alpha,
                )
                for u in range(rating.shape[0])
                if u not in group
            ],
            expected[cont],
            "smgu",
        )
        cont += 1


if __name__ == "__main__":
    try:
        test_group_of_users()
        print("\033[92m" + "✅ Test group_of_users passed!" + "\033[0m")
    except Exception as e:
        print(
            "\033[91m"
            + f"❌ Test group_of_user_passed! failed with error: {e}"
            + "\033[0m"
        )
