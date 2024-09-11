import numpy as np

def calcular_mse(valores_esperados, valores_resultantes):
    # Convertir a arrays de numpy
    valores_esperados = np.array(valores_esperados)
    valores_resultantes = np.array(valores_resultantes)
    
    # Calcular el error cuadrático medio
    mse = np.mean((valores_esperados - valores_resultantes) ** 2)
    
    return mse

# Calcular MAE (Error Medio Absoluto)
def calcular_mae(valores_esperados, valores_resultantes):
    valores_esperados = np.array(valores_esperados)
    valores_resultantes = np.array(valores_resultantes)
    mae = np.mean(np.abs(valores_esperados - valores_resultantes))
    return mae