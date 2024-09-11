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

# Calcular nDCG (Ganancia Acumulada Descontada Normalizada)
def calcular_ndcg(valores_esperados, valores_resultantes):
    # nDCG se basa en la posición de los resultados, se necesita ordenar por relevancia
    valores_esperados = np.array(valores_esperados)
    valores_resultantes = np.array(valores_resultantes)
    
    def dcg(relevancia):
        """Calcula la DCG"""
        relevancia = np.asarray(relevancia)
        return np.sum((relevancia / np.log2(np.arange(2, relevancia.size + 2))))
    
    # DCG de los resultados obtenidos
    dcg_valores = dcg(valores_resultantes)
    
    # DCG ideal (orden perfecto)
    dcg_ideal = dcg(sorted(valores_esperados, reverse=True))
    
    # nDCG
    ndcg = dcg_valores / dcg_ideal if dcg_ideal > 0 else 0.0
    return ndcg