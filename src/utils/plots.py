import pandas as pd
import matplotlib.pyplot as plt


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

    # Ocultar el gráfico de los ejes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Crear la tabla
    tabla = ax.table(cellText=data, colLabels=columnas, loc="center", cellLoc="center")

    # Ajustar el tamaño de la tabla
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(6)
    tabla.scale(0.4, 2)

    # Mostrar la tabla
    plt.show()


def create_differences_plot(y, title):
    """
    Create a differences plot using matplotlib.
    Parameters:
    - y (list): A list of values for the y-axis.
    - title (str): The title of the plot.
    Returns:
    None
    """

    # Datos para los ejes X e Y
    x = [i for i in range(len(y))]

    # Crear una figura y un conjunto de ejes
    fig, ax = plt.subplots()

    # Dibujar una gráfica de X vs Y
    ax.plot(x, y, marker="o", linestyle="-", color="b")

    # Etiquetas de los ejes
    ax.set_xlabel("Grupo")
    ax.set_ylabel("Diferencia")
    ax.set_title(title)

    # Ajustar el gráfico para hacer espacio a la tabla
    plt.subplots_adjust(left=0.2, bottom=0.2)

    # Mostrar el gráfico y la tabla
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
