import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def create_table_st(columnas, data):
    """
    Create a table plot using streamlit.
    Parameters:
    - columnas (list): A list of column labels for the table.
    - data (list of lists): A 2D list representing the data for the table.
    Returns:
    None
    """

    tabla = pd.DataFrame(data, columns=columnas)

    st.table(tabla)


def create_table(columnas, data):
    """
    Create a table plot using matplotlib.
    Parameters:
    - columnas (list): A list of column labels for the table.
    - data (list of lists): A 2D list representing the data for the table.
    Returns:
    None
    """

    fig, ax = plt.subplots(figsize=(10, 30))

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    tabla = ax.table(cellText=data, colLabels=columnas, loc="center", cellLoc="center")

    tabla.auto_set_font_size(False)
    tabla.set_fontsize(6)
    tabla.scale(0.4, 2)

    plt.show()


def create_differences_plot_st(y, title):
    """
    Create a differences plot using streamlit.
    Parameters:
    - y (list): A list of values for the y-axis.
    - title (str): The title of the plot.
    Returns:
    None
    """

    x = [i for i in range(len(y))]

    fig, ax = plt.subplots()

    ax.plot(x, y, marker="o", linestyle="-", color="b")

    ax.set_xlabel("Grupo")
    ax.set_ylabel("Diferencia")
    ax.set_title(title)

    plt.subplots_adjust(left=0.2, bottom=0.2)

    st.pyplot(fig)


def create_differences_plot(y, title):
    """
    Create a differences plot using matplotlib.
    Parameters:
    - y (list): A list of values for the y-axis.
    - title (str): The title of the plot.
    Returns:
    None
    """

    x = [i for i in range(len(y))]

    fig, ax = plt.subplots()

    ax.plot(x, y, marker="o", linestyle="-", color="b")

    ax.set_xlabel("Grupo")
    ax.set_ylabel("Diferencia")
    ax.set_title(title)

    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.show()


def create_excel(data: dict):
    """
    Creates an Excel file from the given data dictionary.

    Parameters:
    data (dict): A dictionary containing the data to be exported to Excel.

    Returns:
    None
    """

    df = pd.DataFrame(data)
    df.to_excel("Tabla exportada.xlsx", index=False)


def create_excel_st(data: dict):
    """
    Creates an Excel file from the given data dictionary and downloads it using Streamlit.

    Parameters:
    data (dict): A dictionary containing the data to be exported to Excel.

    Returns:
    None
    """

    df = pd.DataFrame(data)
    excel_file = df.to_excel(index=False)
    st.download_button(
        label="Download Excel",
        data=excel_file,
        file_name="Tabla_exportada.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
