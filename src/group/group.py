import numpy as np


def unpopular(rating: np.ndarray, movie: int):  # pi
    return 1 - np.sum(rating[:, movie] != -1) / rating.shape[0]

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

# # Test 1 Unpopularity
# print([unpopular(rating, i) for i in range(rating.shape[1])])