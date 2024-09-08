# import os
# import numpy as np
# import torch
# from extended_naive_bayes.nbp import group_prediction
# from nbcf.nbcf_opt import nbcf, predict_hybrid
# from ml_processing.ml_procesing import MovieLensProcessing


# def main():
#     # Load data
#     if torch.cuda.is_available():
#         print(f"Usando GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
#     else:
#         print("No se está utilizando GPU. Usando CPU.")

#     time = os.times()
#     ml_processing = MovieLensProcessing(rating_path="./datasets/ml-1m/ratings.dat")
#     rating = ml_processing.numpy_user_movie_matrix()

#     # Create recommenders
#     alpha = 0.01
#     r = 5

#     pi, pu = nbcf(rating=rating, alpha=alpha, r=r)

#     hybrid_prediction = predict_hybrid(
#         rating=rating, r=r, predict_item=pi, predict_user=pu
#     )

#     print("group recommender")
#     # Group recommender
#     groups = []
#     group_predict = np.ndarray((len(groups), rating.shape[1], r))

#     for i in len(groups):
#         prediction = group_prediction(
#             rating=rating, group=groups[i], hp=hybrid_prediction, r=r
#         )
#         for j in range(rating.shape[1]):
#             for y in range(r):
#                 group_predict[j][y] = prediction[j][y]

#     print("final prediction")
#     prediction = np.full(rating.shape, -1)

#     users, movies = rating.shape
#     for user in users:
#         for movie in movies:
#             if rating[user][movie] == -1:
#                 current_y = 0
#                 current_trust = 0
#                 for y in range(r):
#                     if hybrid_prediction[user][movie][y] > current_trust:
#                         current_trust = hybrid_prediction[user][movie][y]
#                         current_y = y
#                 prediction[user][movie] = current_y

#     print("Time:", os.times() - time)


# if __name__ == "__main__":
#     main()


import os
import numpy as np
import torch
from extended_naive_bayes.nbp import group_prediction
from nbcf.nbcf_opt import nbcf, predict_hybrid
from ml_processing.ml_procesing import MovieLensProcessing


def main():
    # Verificar si PyTorch está utilizando la GPU
    if torch.cuda.is_available():
        print(f"Usando GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("No se está utilizando GPU. Usando CPU.")

    # Medir el tiempo de las operaciones
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    # Load data
    time = os.times()
    ml_processing = MovieLensProcessing(rating_path="./datasets/ml-1m/ratings.dat")
    rating = ml_processing.numpy_user_movie_matrix()
    rating = rating[:100, :100]  # Reducir el tamaño de la matriz para pruebas

    # Convertir rating a tensor de PyTorch y mover a GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rating_tensor = torch.tensor(rating, device=device)

    # Create recommenders
    alpha = 0.01
    r = 5

    # Iniciar la medición del tiempo
    start_time.record()

    pi, pu, user_map, movie_map = nbcf(rating=rating_tensor, alpha=alpha, r=r)

    hybrid_prediction = predict_hybrid(
        rating=rating_tensor,
        r=r,
        predict_item=pi,
        predict_user=pu,
        user_map=user_map,
        movie_map=movie_map,
    )

    # Finalizar la medición del tiempo
    end_time.record()

    # Sincronizar y calcular el tiempo transcurrido
    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time)  # Tiempo en milisegundos
    print(f"Tiempo de ejecución en GPU: {elapsed_time/1000} s")

    # Mostrar el uso de memoria de la GPU
    print(
        f"Memoria utilizada en GPU: {torch.cuda.memory_allocated(device) / 1024**2} MB"
    )
    print(
        f"Memoria reservada en GPU: {torch.cuda.memory_reserved(device) / 1024**2} MB"
    )


if __name__ == "__main__":
    main()
