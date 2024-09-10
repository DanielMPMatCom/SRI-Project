from sklearn.cluster import KMeans


class ClusterNBCF:
    def __init__(self, preprocessing, nbcf) -> None:

        rating, qualified = preprocessing.numpy_user_movie_matrix()

        for user in range(nbcf.users):
            for movie in range(nbcf.movies):
                if rating[user][movie] == -1:
                    rating[user][movie] = nbcf.prediction[user][movie].argmax() + 1

        vectorized_users = [rating[user] for user in range(nbcf.users)]

        k = 100

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(vectorized_users)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        print("Labels", labels)
        print("Centroids", centroids)
