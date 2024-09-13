import numpy as np

def calcular_mse(valores_esperados, valores_resultantes):
    """
    Calcula el error cuadrático medio (MSE) entre los valores esperados y los valores resultantes.

    Args:
        valores_esperados (array-like): Valores esperados.
        valores_resultantes (array-like): Valores resultantes.

    Returns:
        float: El error cuadrático medio.
    """
    valores_esperados = np.array(valores_esperados)
    valores_resultantes = np.array(valores_resultantes)
    
    mse = np.mean((valores_esperados - valores_resultantes) ** 2)
    
    return mse

def calcular_mae(valores_esperados, valores_resultantes):
    """
    Calculates the Mean Absolute Error (MAE) between the expected values and the resulting values.

    Parameters:
    valores_esperados (array-like): The expected values.
    valores_resultantes (array-like): The resulting values.

    Returns:
    float: The calculated MAE.

    """
    valores_esperados = np.array(valores_esperados)
    valores_resultantes = np.array(valores_resultantes)
    mae = np.mean(np.abs(valores_esperados - valores_resultantes))
    return mae