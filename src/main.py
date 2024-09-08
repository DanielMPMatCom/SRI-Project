import torch
from extended_naive_bayes.nbp import group_prediction
from nbcf.nbcf_opt import nbcf, predict_hybrid
from ml_processing.ml_procesing import MovieLensProcessing
from ft_processing.ft_procesing import FilmTrustProcessing
import time
import os


def main():
    # # Load data
    # ml_processing = FilmTrustProcessing()
    # rating, qualified = ml_processing.numpy_user_movie_matrix()

    # # Create recommenders
    # alpha = 0.01
    # r = qualified[-1] + 1
    # print("R", r)
    # print("QUALIFIEDS", qualified)
    # # Iniciar la medición del tiempo
    # duration = time.time()
    # # start_time.record()
    # print("Iniciando el entrenamiento del modelo ...", rating.shape)
    # pi, pu, user_map, movie_map = nbcf(
    #     rating=rating, alpha=alpha, r=r, qualified_array=qualified
    # )

    # hybrid_prediction = predict_hybrid(
    #     rating=rating,
    #     r=r,
    #     predict_item=pi,
    #     predict_user=pu,
    #     user_map=user_map,
    #     movie_map=movie_map,
    #     qualified_array=qualified,
    # )

    # # save the model
    # name = "hybrid_prediction_fm.pt"
    # while os.path.exists(name):
    #     print(
    #         "\033[93mEl archivo ya existe. Escriba -name seguido de un nombre para guardar el archivo, o -r para remplazarlo. \033[0m"
    #     )
    #     name = input()
    #     if name == "-r":
    #         break
    #     else:
    #         name = name + ".pt"

    # torch.save(hybrid_prediction, "hybrid_prediction_ft.pt")

    # # Finalizar la medición del tiempo
    # duration = time.time() - duration

    # print(f"Tiempo de ejecución : {duration} ")

    # load the model
    t = time.time()
    hybrid_prediction = torch.load("hybrid_prediction_ft.pt")
    print("Modelo cargado en ", time.time() - t)


if __name__ == "__main__":
    main()
