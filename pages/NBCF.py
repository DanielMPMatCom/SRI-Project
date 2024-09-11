import streamlit as st


import numpy as np
from src.utils.create_grups import generate_groups
from src.extended_naive_bayes.nbp import group_prediction
from src.nbcf.nbcf_opt import NBCF
from src.ft_processing.ft_procesing import FilmTrustProcessing
import time

from src.utils.statistics import *
from src.extended_naive_bayes.nbp import group_prediction
from src.utils.plots import create_table, create_excel, create_differences_plot


def main():
    st.markdown("# Implementación de NBCF y NBP")
    st.warning(
        "Este script es una implementación de NBCF y NBP para el dataset de FilmTrust. Advertencia: Recomendamos ejecutar el script `main.py` para obtener los resultados con mayor rapidez"
    )

    preprocessing = FilmTrustProcessing()
    _, test = preprocessing.separate_data_for_test()
    rating, qualified = preprocessing.numpy_user_movie_matrix(remove_data=test)

    # Create recommenders
    st.write("## NBCF Parameters")
    alpha = st.slider("Alpha", 0.01, 1.0, 0.01, disabled=True)
    r = st.slider("R", 1, 10, 8, disabled=True)

    # Iniciar la medición del tiempo
    duration = time.time()

    # Crear grupos
    groups = generate_groups(rating, 100, [6, 7, 8])
    # [print(f"{movie}: {len(x)}") for (movie, _), x in groups.items()]
    # print(f"Created {len(groups)} groups")
    print(len(groups))

    st.info(f"Se han creado {len(groups)} grupos")

    final_groups = {}

    print(len(groups))

    np.random.seed(42)
    for (movie, q), users in groups.items():
        final_groups[movie, q] = np.random.choice(users, 20, replace=False)

        for user in final_groups[movie, q]:
            rating[user, movie] = -1

    st.info(
        f"Iniciando el modelo con {rating.shape[0]} usuarios y {rating.shape[1]} películas"
    )

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

    duration = time.time() - duration

    print(f"⏰ Tiempo de ejecución : {duration} ")
    st.info(f"Tiempo de ejecución: {duration}")

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

    # print("Iniciando test...")
    # for u, m, r in test:
    #     u, m = int(u), int(m)
    #     print(
    #         f"\033[93mPredicción: u, p = ({u},{m}): {hybrid_prediction[u, m].argmax() + 1}, Real: {r}\033[0m"
    #     )

    prediction = {}
    to_export = {
        "Users": [],
        "Movie": [],
        "Expected": [],
        "Recieved": [],
    }
    to_plot = []
    differences = []
    abs_differences = []

    # for (movie, q), group in final_groups.items():
    #     print("Movie ", movie)
    #     for user in group:
    #         print(f"User: {user}, nbcf prediction: {hybrid_prediction[user, movie].argmax() + 1}, probs: {hybrid_prediction[user, movie]}")
    #         if rating[user, movie] != -1:
    #             print("+++++++++++++++++++++++++++++")

    for (movie, q), group in final_groups.items():
        prediction[movie, q] = group_prediction(
            rating, group, hybrid_prediction, qualified, movie
        )[movie]

        to_export["Users"].append(", ".join(map(str, group)))
        to_export["Movie"].append(movie)
        to_export["Expected"].append(q)
        to_export["Recieved"].append(prediction[movie, q].argmax() + 1)

        # to_plot.append([movie, q, prediction[movie, q].argmax() + 1])
        # differences.append(q - prediction[movie, q].argmax() - 1)
        # abs_differences.append(np.abs(q - prediction[movie, q].argmax() - 1))

    # for movie, q in final_groups.keys():
    #     print(f"Movie: {movie}")
    #     print(f"Expected {q}, recived {prediction[movie, q].argmax() + 1}, distribution {prediction[movie, q]}")

    # create_excel(to_export)
    # create_table(("Movie", "Expected", "Recieved"), to_plot)
    # create_differences_plot(differences, 'Diferencia por grupo entre el valor esperado y el valor de la predicción')
    # create_differences_plot(abs_differences, 'Diferencia absoluta por grupo entre el valor esperado y el valor de la predicción')

    mse = calcular_mse(to_export["Expected"], to_export["Recieved"])
    mae = calcular_mae(to_export["Expected"], to_export["Recieved"])

    print(f"MSE: {mse}")
    print(f"MAE: {mae}")

    st.info(f"El error cuadrático medio es: {mse}")
    st.info(
        f"El error absoluto medio es: {mae}",
    )
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
    st.set_page_config(page_title="Informe", page_icon=":bar_chart:", layout="wide")
    main()
