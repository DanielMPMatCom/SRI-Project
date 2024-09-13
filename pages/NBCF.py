import streamlit as st


import numpy as np
from src.utils.create_grups import generate_groups
from src.extended_naive_bayes.nbp import group_prediction
from src.nbcf.nbcf_opt import NBCF
from src.ft_processing.ft_procesing import FilmTrustProcessing
import time

from src.utils.statistics import *
from src.extended_naive_bayes.nbp import group_prediction
from src.utils.plots import create_table_st, create_excel_st, create_differences_plot_st


def main():
    st.markdown("# NBCF and NBP Implementation.")

    st.warning(
        "Note: For faster results, it is recommended to run the main.py script."
    )

    if st.button("Load NBCF and NBP models."):

        alpha = 0.01
        r = 8
        preprocessing = FilmTrustProcessing()
        # _, test = preprocessing.separate_data_for_test()
        rating, qualified = preprocessing.numpy_user_movie_matrix()

        with st.spinner("Generating NBCF..."):
            duration = time.time()

            st.info("Initializing test...\Generating Groups with Ratings [6,7,8]")

            groups = generate_groups(rating, 100, [6, 7, 8])

            print(len(groups))

            st.info(f"{len(groups)} groupes have been created.")

            st.session_state.final_groups = {}

            st.info(f"Creating final groupes...")

            np.random.seed(42)

            for (movie, q), users in groups.items():
                st.session_state.final_groups[movie, q] = np.random.choice(
                    users, 20, replace=False
                )
                for user in st.session_state.final_groups[movie, q]:
                    rating[user, movie] = -1

            st.info(
                f"Initializing model with {rating.shape[0]} users and {rating.shape[1]} movies..."
            )

            nbcf_instance = NBCF(
                rating=rating,
                alpha=alpha,
                r=r,
                qualified_array=qualified,
                load=True,
                loading_streamlit_bootstrap=st.progress(0, "Loading..."),
            )

            st.session_state.hybrid_prediction = nbcf_instance.prediction

            duration = time.time() - duration

            st.info(f"⏰ Execution Time : {duration} ")

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
        st.info("Calculating predictions for groupes...")
        for (movie, q), group in st.session_state.final_groups.items():
            prediction[movie, q] = group_prediction(
                rating, group, st.session_state.hybrid_prediction, qualified, movie
            )[movie]

            to_export["Users"].append(", ".join(map(str, group)))
            to_export["Movie"].append(movie)
            to_export["Expected"].append(q)
            to_export["Recieved"].append(prediction[movie, q].argmax() + 1)

            to_plot.append([movie, q, prediction[movie, q].argmax() + 1])
            differences.append(q - prediction[movie, q].argmax() - 1)
            abs_differences.append(np.abs(q - prediction[movie, q].argmax() - 1))

        create_table_st(("Movie", "Expected", "Recieved"), to_plot)
        create_differences_plot_st(
            differences,
            "Difference in predicted versus expected values by group.",
        )
        create_differences_plot_st(
            abs_differences,
            "Absolute difference in predicted versus expected values by group.",
        )

        mse = calcular_mse(to_export["Expected"], to_export["Recieved"])
        mae = calcular_mae(to_export["Expected"], to_export["Recieved"])

        st.info(f"The obteined MSE was: {mse}")
        st.info(f"The obteined MAE was: {mae}")


if __name__ == "__main__":
    st.set_page_config(page_title="Informe", page_icon=":bar_chart:", layout="wide")
    main()
