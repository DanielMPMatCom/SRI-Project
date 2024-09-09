import numpy as np
from extended_naive_bayes.nbp import group_prediction
from nbcf.nbcf_opt import nbcf, predict_hybrid
from ml_processing.ml_procesing import MovieLensProcessing
from ft_processing.ft_procesing import FilmTrustProcessing
import time


def main():
    # Load data
    preprocessing = FilmTrustProcessing()
    _, test = preprocessing.separate_data_for_test()
    rating, qualified = preprocessing.numpy_user_movie_matrix(remove_data=test)

    # Create recommenders
    alpha = 0.001
    r = qualified[-1] + 1

    # Iniciar la medición del tiempo
    duration = time.time()

    print("Iniciando el entrenamiento del modelo ...", rating.shape)
    pi, pu, user_map, movie_map = nbcf(
        rating=rating, alpha=alpha, r=r, qualified_array=qualified
    )

    hybrid_prediction = predict_hybrid(
        rating=rating,
        r=r,
        predict_item=pi,
        predict_user=pu,
        user_map=user_map,
        movie_map=movie_map,
        qualified_array=qualified,
    )

    np.save(
        "hybrid_prediction_ft_data_train.npy",
        hybrid_prediction,
    )
    np.save(
        "hybrid_prediction_ft_data_test.npy",
        test,
    )

    print("Modelo guardado")

    # Finalizar la medición del tiempo
    duration = time.time() - duration

    print(f"⏰ Tiempo de ejecución : {duration} ")
    hybrid_prediction = np.load(
        "hybrid_prediction_ft_data_train.npy",
    )
    test = np.load("hybrid_prediction_ft_data_test.npy")

    print("Iniciando test...")
    for u, m, r in test:
        u, m = int(u), int(m)
        print(
            f"\033[93mPredicción para el usuario {u}, pelicula {m}: {hybrid_prediction[u, m].argmax()}, Real: {r}\033[0m"
        )

    # # load the model
    # t = time.time()
    # hybrid_prediction = torch.load("hybrid_prediction_ft.pt")
    # print("Modelo cargado en ", time.time() - t)


if __name__ == "__main__":
    main()
