import numpy as np
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

    print("Iniciando el entrenamiento del modelo ...", rating.shape)
    nbcf_instance = NBCF(
        rating=rating, alpha=alpha, r=r, qualified_array=qualified, load=True
    )
    # return
    np.save(
        "./db/prediction.npy",
        nbcf_instance.prediction,
    )
    np.save(
        "./db/test.npy",
        test,
    )

    print("Modelo guardado")

    # Finalizar la medición del tiempo
    duration = time.time() - duration

    print(f"⏰ Tiempo de ejecución : {duration} ")

    hybrid_prediction = np.load("./db/prediction.npy")
    test = np.load("./db/test.npy")

    print("Iniciando test...")
    for u, m, r in test:
        u, m = int(u), int(m)
        print(
            f"\033[93mPredicción: u, p = ({u},{m}): {hybrid_prediction[u, m].argmax() + 1}, Real: {r}\033[0m"
        )

    # # load the model
    # t = time.time()
    # hybrid_prediction = torch.load("hybrid_prediction_ft.pt")
    # print("Modelo cargado en ", time.time() - t)


if __name__ == "__main__":
    main()
