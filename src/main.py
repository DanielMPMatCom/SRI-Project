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
    groups = generate_groups(rating, 50)
    # [print(f"{movie}: {len(x)}") for (movie, _), x in groups.items()]
    # print(f"Created {len(groups)} groups")

    # Crear grupo de prueba
    g = (0, groups[(0, 5)])

    print(" == = = = == = =  ", len(g[1]))

    # Eliminar datos del rating principal
    for user in g[1][:50]:
        rating[user][g[0]] = -1

  

    # print("Iniciando el entrenamiento del modelo ...", rating.shape)
    # nbcf_instance = NBCF(
    #     rating=rating, alpha=alpha, r=r, qualified_array=qualified, load=True
    # )
    # np.save(
    #     "./db/prediction",
    #     nbcf_instance.prediction,
    # )
    # np.save(
    #     "./db/test",
    #     test,
    # )

    # duration = time.time() - duration

    # print(f"⏰ Tiempo de ejecución : {duration} ")

    hybrid_prediction = np.load("./db/prediction.npy")
    test = np.load("./db/test.npy")

    print("Iniciando test...")
    # for u, m, r in test:
    #     u, m = int(u), int(m)
    #     print(
    #         f"\033[93mPredicción: u, p = ({u},{m}): {hybrid_prediction[u, m].argmax() + 1}, Real: {r}\033[0m"
    #     )

    from extended_naive_bayes.nbp import group_prediction

    prediction = group_prediction(rating, g[1], hybrid_prediction, qualified)
    print(sorted([ (i + 1, v) for i, v in enumerate(prediction[0])], key=lambda x : x[1] ,reverse=True))
    print(prediction[0].argmax() + 1)
    print("NBCF le daria por usuario")
    for i in g[1]:
        print(
            f"\033[93mPredicción: u, p = ({i},{0}): {hybrid_prediction[i, 0].argmax() + 1}\033[0m"
        )

    # # load the model
    # t = time.time()
    # hybrid_prediction = torch.load("hybrid_prediction_ft.pt")
    # print("Modelo cargado en ", time.time() - t)


if __name__ == "__main__":
    main()
