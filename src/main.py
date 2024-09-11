import numpy as np
from utils.create_grups import generate_groups
from extended_naive_bayes.nbp import group_prediction
from nbcf.nbcf_opt import NBCF
from ml_processing.ml_procesing import MovieLensProcessing
from ft_processing.ft_procesing import FilmTrustProcessing
import time


def main():
    # Load data
    preprocessing = FilmTrustProcessing()
    _, test = preprocessing.separate_data_for_test()
    rating, qualified = preprocessing.numpy_user_movie_matrix(remove_data=test)

    # Create recommenders
    alpha = 0.01
    r = 8

    # Iniciar la medición del tiempo
    duration = time.time()

    # Crear grupos
    groups = generate_groups(rating, 25, [1, 2, 3])
    # [print(f"{movie}: {len(x)}") for (movie, _), x in groups.items()]
    # print(f"Created {len(groups)} groups")
    print(len(groups))

    # return
   
    # Crear grupos finales 
    final_groups = {}

    np.random.seed(42)
    for (movie, q), users in groups.items():
        final_groups[movie, q] = np.random.choice(users, 20, replace=False)

        for user in final_groups[movie, q]:
            rating[user, movie] = -1
  

    print("Iniciando el entrenamiento del modelo ...", rating.shape)
    nbcf_instance = NBCF(
        rating=rating, alpha=alpha, r=r, qualified_array=qualified, load=True
    )
    np.save(
        "./db/prediction",
        nbcf_instance.prediction,
    )
    np.save(
        "./db/test",
        test,
    )
    
    np.save(
        "./db/rating",
        rating,
    )
    

    # duration = time.time() - duration

    # print(f"⏰ Tiempo de ejecución : {duration} ")

    # for user in range(nbcf_instance.users):
    #     for movie in range(nbcf_instance.movies):
    #         if rating[user][movie] == -1:
    #             for q in qualified:
    #                 if(nbcf_instance.prediction[user,movie,q] == 0):
    #                     print(user, movie, q)

    hybrid_prediction = np.load("./db/prediction.npy")
    test = np.load("./db/test.npy")
    rating = np.load("./db/rating.npy")


    # print([nbcf_instance.prediction[911, 2, r ] for r in qualified],' - - - - - - - - - -- - - - - - - - -')
    # print([hybrid_prediction[911, 2, r ] for r in qualified],' - - - - - - - - - -- - - - - - - - -')

    # for user in range(nbcf_instance.users):
    #     for movie in range(nbcf_instance.movies):
    #         if rating[user][movie] == -1:
    #             for q in qualified:
    #                 if(nbcf_instance.prediction[user,movie,q] == 0):
    #                     print("Failed preload ", user, movie, q)
    #                 if(hybrid_prediction[user, movie, q] == 0):
    #                     print("Failed preload ", user, movie, q)

    print("Iniciando test...")
    # for u, m, r in test:
    #     u, m = int(u), int(m)
    #     print(
    #         f"\033[93mPredicción: u, p = ({u},{m}): {hybrid_prediction[u, m].argmax() + 1}, Real: {r}\033[0m"
    #     )

    from extended_naive_bayes.nbp import group_prediction

    prediction = {}

    for (movie, q), group in final_groups.items():
        prediction[movie, q] = group_prediction(rating, group, hybrid_prediction, qualified, movie)[movie]
        # print("++++++++++++++++++")
        # print(prediction[movie, q][movie])
        # print("++++++++++++++++++")

    for movie, q in final_groups.keys():
        print(f"Movie: {movie}")
        print(f"Expected {q}, recived {prediction[movie, q].argmax() + 1}, distribution {prediction[movie, q]}")
    
    # print(sorted([ (i + 1, v) for i, v in enumerate(prediction[0])], key=lambda x : x[1] ,reverse=True))
    # print(prediction[0].argmax() + 1)
    # print("NBCF le daria por usuario")
    
    # for i in g[1]:
    #     print(
    #         f"\033[93mPredicción: u, p = ({i},{0}): {hybrid_prediction[i, 0].argmax() + 1}\033[0m"
    #     )

    # # load the model
    # t = time.time()
    # hybrid_prediction = torch.load("hybrid_prediction_ft.pt")
    # print("Modelo cargado en ", time.time() - t)


if __name__ == "__main__":
    main()
