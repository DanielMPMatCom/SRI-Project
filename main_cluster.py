from sklearn.cluster import KMeans
import numpy as np
import os
from src.ft_processing.ft_processing import FilmTrustProcessing
from src.nbcf.nbcf_opt import NBCF
from src.extended_naive_bayes.nbp import group_prediction


class ClusterNBCF:
    ROOT_PATH = "./db/cluster"
    alpha = 0.01
    r = 8
    k = 20
    vectorized_users = []
    qualified = []
    original_rating = []

    def __init__(self, calculate_nbcf_base, save = True):
        if not os.path.exists(self.ROOT_PATH):
            os.makedirs(self.ROOT_PATH)

        
        if(not os.path.exists(f"{self.ROOT_PATH}/cluster/")):
            os.makedirs(f"{self.ROOT_PATH}/cluster/")
        
        if calculate_nbcf_base:
            preprocessing = FilmTrustProcessing()
            self.original_rating, self.qualified = preprocessing.numpy_user_movie_matrix()
            if save:
                np.save(f"{self.ROOT_PATH}/original_rating", self.original_rating)
            

            self.nbcf = NBCF(self.original_rating, self.alpha, self.r, self.qualified)

            rating = np.copy(self.original_rating)
            for user in range(self.nbcf.users):
                for movie in range(self.nbcf.movies):
                    if rating[user][movie] == -1:
                        rating[user][movie] = self.nbcf.prediction[user][movie].argmax() + 1

            self.vectorized_users = np.array([rating[user] for user in range(self.nbcf.users)])

            if save:
                np.save(f"{self.ROOT_PATH}/vectorized_users", self.vectorized_users)
                np.save(f"{self.ROOT_PATH}/qualified", self.qualified)

           
    def load_user_rating_qualified_data(self):
        self.vectorized_users = np.load(f"{self.ROOT_PATH}/vectorized_users.npy")
        self.qualified = np.load(f"{self.ROOT_PATH}/qualified.npy")
        self.original_rating = np.load(f"{self.ROOT_PATH}/original_rating.npy")

    def load_labels_centroid_cluster_qualified(self):
        self.labels = np.load(f"{self.ROOT_PATH}/labels.npy")
        self.centroids = np.load(f"{self.ROOT_PATH}/centroids.npy")
        self.cluster_qualification = np.load(f"{self.ROOT_PATH}/cluster_qualification.npy")


    def make_groups(self,important_qualified=[6,7,8] , save=True ):
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(self.vectorized_users)
        self.labels = kmeans.labels_
        self.centroids = kmeans.cluster_centers_

        print("Labels", self.labels)
        print("Centroids", self.centroids)

        self.important_qualifications = important_qualified
        self.cluster_qualification = np.zeros((self.k, self.original_rating.shape[1]))

        for cluster in range(self.k):
            group = np.where(self.labels == cluster)[0]
            ci = np.zeros(self.original_rating.shape[1])
            for movie in range(self.original_rating.shape[1]):
                ci[movie] = np.sum(
                    [self.original_rating[user][movie] in self.important_qualifications for user in group]
                )
            self.cluster_qualification[cluster] = ci

        for cluster in range(self.k):
            print(
                f"Cluster {cluster} -> {(sorted(enumerate(self.cluster_qualification[cluster]), 
                key=lambda x: x[1], reverse=True))[0]} / {len(np.where(self.labels == cluster)[0])}"
            )

        if save:
            np.save(f"{self.ROOT_PATH}/labels", self.labels)
            np.save(f"{self.ROOT_PATH}/centroids", self.centroids)
            np.save(f"{self.ROOT_PATH}/cluster_qualification", self.cluster_qualification)
        
    def group_prediction(self):
        # Remove the most qualified movie from the cluster
        for cluster in range(self.k):
            group = np.where(self.labels == cluster)[0]
            most_qualified_movie = (
                sorted(
                    enumerate(self.cluster_qualification[cluster]),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )[0][0]

            prev_rating = []
            for user in group:
                prev_rating.append(self.original_rating[user][most_qualified_movie])
                self.original_rating[user][most_qualified_movie] = -1
            
            # calculate the new nbcf for the cluster
          
            if os.path.exists(
                f"{self.ROOT_PATH}/cluster/group_nbcf_{cluster}"):
                prediction_cl = np.load(f"{self.ROOT_PATH}/cluster/group_nbcf_{group}.npy")
            else:
                group_nbcf = NBCF(self.original_rating, self.alpha, self.r, self.qualified)
                np.save(f"{self.ROOT_PATH}/cluster/group_nbcf_{cluster}", group_nbcf.prediction)
                prediction_cl = group_nbcf.prediction

            prediction = group_prediction(self.original_rating, group, prediction_cl, self.qualified, most_qualified_movie)

            print(
            "Prediction", [(i + 1,v) for i, v in enumerate(prediction[most_qualified_movie])]
        )
            

            for user, rating in zip(group, prev_rating):
                self.original_rating[user][most_qualified_movie] = rating

if __name__ == "__main__":
    pipeline = ClusterNBCF(False, False)
    pipeline.load_user_rating_qualified_data()
    # pipeline.make_groups()
    pipeline.load_labels_centroid_cluster_qualified()
    pipeline.group_prediction()
