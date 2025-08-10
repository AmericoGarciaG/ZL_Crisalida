# src/utils.py

import numpy as np
import os

"""
Archivo de Utilidades para el Proyecto Crisálida.
Contiene funciones de ayuda compartidas para mantener el código limpio y DRY.
"""

def ensure_dirs_exist():
    """Asegura que todos los directorios de salida existan."""
    from config import MODELS_DIR, RESULTS_DIR, REPORTS_DIR
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def convert_numpy_types(obj):
    """
    Convierte recursivamente tipos de datos de NumPy a tipos nativos de Python 
    para una serialización JSON segura.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj