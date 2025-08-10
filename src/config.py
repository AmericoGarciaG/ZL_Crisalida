# src/config.py

from pathlib import Path

"""
Archivo de Configuración Central para el Proyecto Crisálida.
Contiene todas las rutas, hiperparámetros y constantes para asegurar consistencia.
"""

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE RUTAS (Path Configuration)
# -----------------------------------------------------------------------------
# Raíz del proyecto, calculada dinámicamente.
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
except NameError:
    # Manejo para entornos interactivos como Jupyter
    PROJECT_ROOT = Path('.').resolve().parent

# Directorios principales
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Rutas a archivos de datos de entrada
WINNERS_DATASET_PATH = DATA_DIR / 'winners_dataset.csv'
NOISE_DATASET_PATH = DATA_DIR / 'noise_dataset.csv'
FEATURE_IMPORTANCE_PATH = DATA_DIR / 'feature_importance_report.json'

# Rutas a artefactos generados
HYBRID_FEATURES_PATH = DATA_DIR / 'hybrid_features.npy'
HYBRID_LABELS_PATH = DATA_DIR / 'hybrid_labels.npy'
HYBRID_METADATA_PATH = DATA_DIR / 'hybrid_metadata.json'
TOP10_MAPPING_PATH = RESULTS_DIR / 'top10_feature_mapping.json'
TOP10_ANALYSIS_PNG_PATH = RESULTS_DIR / 'top10_features_analysis.png'

# Rutas a modelos y resultados
TRANSFORMER_OMEGA_BEST_PATH = MODELS_DIR / 'transformer_omega_best.keras'
TRANSFORMER_OMEGA_FINAL_PATH = MODELS_DIR / 'transformer_omega_final.keras'
OMEGA_RESULTS_PATH = RESULTS_DIR / 'transformer_omega_results.json'
OMEGA_PREDICTIONS_PATH = RESULTS_DIR / 'transformer_omega_predictions.npy'
COMPREHENSIVE_EVALUATION_REPORT_PATH = RESULTS_DIR / 'comprehensive_evaluation_report.json'
COMPREHENSIVE_EVALUATION_PNG_PATH = RESULTS_DIR / 'comprehensive_evaluation.png'
COMPREHENSIVE_EVALUATION_MD_PATH = REPORTS_DIR / 'fase5_comprehensive_evaluation.md'

# -----------------------------------------------------------------------------
# 2. CONFIGURACIÓN DEL MODELO TRANSFORMER-OMEGA
# -----------------------------------------------------------------------------
# Dimensiones de las características de entrada
AFFINITY_DIM = 14
CONTEXTUAL_DIM = 60

# Arquitectura del Transformer
MODEL_CONFIG = {
    'affinity_dim': AFFINITY_DIM,
    'contextual_dim': CONTEXTUAL_DIM,
    'd_model': 128,
    'num_heads': 8,
    'num_layers': 3,
    'ff_dim': 256,
    'dropout_rate': 0.1
}

# -----------------------------------------------------------------------------
# 3. HIPERPARÁMETROS DE ENTRENAMIENTO
# -----------------------------------------------------------------------------
TRAINING_CONFIG = {
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 50,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'weight_decay': 1e-5,
    'clipnorm': 1.0
}

# -----------------------------------------------------------------------------
# 4. CONSTANTES DE INGENIERÍA DE CARACTERÍSTICAS
# -----------------------------------------------------------------------------
AFFINITY_LEVELS = [2, 3, 4, 5]
FIBONACCI_NUMBERS = {1, 2, 3, 5, 8, 13, 21, 34} # Hasta 39
NUMBER_RANGE_MIN = 1
NUMBER_RANGE_MAX = 39