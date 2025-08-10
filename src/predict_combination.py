# src/predict_combination.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras # type: ignore
from collections import Counter
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import logging
import os

# Importar configuraciones centralizadas
import config
import utils

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class FeatureExtractor:
    def __init__(self, winners_path=config.WINNERS_DATASET_PATH):
        logging.info("Inicializando extractor de características...")
        self.winners_data = pd.read_csv(winners_path, header=None)
        self.affinity_frequencies = self._calculate_affinity_frequencies(config.AFFINITY_LEVELS)
        self.historical_frequencies = self._precalculate_historical_frequencies()
        self.fibonacci_numbers = config.FIBONACCI_NUMBERS
        logging.info("Extractor listo.")

    def _calculate_affinity_frequencies(self, levels):
        freq_counters = {level: Counter() for level in levels}
        for _, row in self.winners_data.iterrows():
            combo = sorted(row.values.tolist())
            for level in levels:
                if len(combo) >= level:
                    freq_counters[level].update(combinations(combo, level))
        return freq_counters

    def _precalculate_historical_frequencies(self):
        all_numbers = self.winners_data.values.flatten()
        total_draws = len(self.winners_data)
        counts = Counter(all_numbers)
        return {num: count / total_draws for num, count in counts.items()}

    def create_feature_vector(self, combination: list) -> np.ndarray:
        if len(combination) != 6:
            raise ValueError("La combinación debe tener 6 números.")
        affinity_vec = self._get_affinity_features(combination)
        contextual_vec = self._get_contextual_features(combination)
        return np.hstack([affinity_vec, contextual_vec]).reshape(1, -1)

    def _get_affinity_features(self, combination):
        sorted_combo = sorted(combination)
        features = np.zeros(config.AFFINITY_DIM)
        from math import comb
        for i, level in enumerate(config.AFFINITY_LEVELS):
            features[i] = sum(self.affinity_frequencies[level].get(sub, 0) for sub in combinations(sorted_combo, level))
        for i, level in enumerate(config.AFFINITY_LEVELS):
            max_combs = comb(6, level)
            features[i + 4] = features[i] / max_combs if max_combs > 0 else 0
        weights = {2: 0.4, 3: 0.3, 4: 0.2, 5: 0.1}
        features[8] = sum(features[k] * weights[level] for k, level in enumerate(config.AFFINITY_LEVELS))
        features[9] = np.var(features[4:8])
        features[10] = np.max(features[0:4])
        features[11] = (features[2] + features[3]) / (features[0] + features[1] + 1e-8)
        features[12] = np.sum(features[0:4]) / sum(sorted_combo) if sum(sorted_combo) > 0 else 0
        features[13] = np.mean(features[4:8])
        return features

    def _get_contextual_features(self, combination):
        features = np.zeros(config.CONTEXTUAL_DIM)
        combo_mean = np.mean(combination)
        for i, number in enumerate(combination):
            base_idx = i * 10
            features[base_idx + 0] = sum(self.affinity_frequencies[2].get(tuple(sorted((number, other))), 0) for other in combination if other != number)
            features[base_idx + 1] = self.historical_frequencies.get(number, 0)
            features[base_idx + 2] = 1.0 if number % 10 == 6 else 0.0
            features[base_idx + 3] = 1.0 if number % 3 == 0 else 0.0
            features[base_idx + 4] = 1.0 if number % 10 == 9 else 0.0
            features[base_idx + 5] = abs(number - combo_mean) / combo_mean if combo_mean > 0 else 0
            features[base_idx + 6] = 1.0 if 10 <= number <= 19 else 0.0
            features[base_idx + 7] = (number - config.NUMBER_RANGE_MIN) / (config.NUMBER_RANGE_MAX - config.NUMBER_RANGE_MIN)
            features[base_idx + 8] = 0.0 if number in self.fibonacci_numbers else 1.0
            features[base_idx + 9] = 1.0 if number in self.fibonacci_numbers else 0.0
        return features

class Predictor:
    def __init__(self, model_path=config.TRANSFORMER_OMEGA_FINAL_PATH):
        logging.info("Cargando modelo Transformer-Omega...")
        self.model = keras.models.load_model(model_path, compile=False, safe_mode=False)
        self.scaler_affinity, self.scaler_contextual = self._fit_scalers()
        logging.info("Modelo y escaladores listos.")

    def _fit_scalers(self):
        logging.info("Ajustando escaladores de normalización...")
        features = np.load(config.HYBRID_FEATURES_PATH)
        # Las características de afinidad son las primeras 14 (AFFINITY_DIM).
        affinity_features = features[:, :config.AFFINITY_DIM]
        # Las características contextuales son las siguientes 60 (desde la 14 hasta el final).
        contextual_features = features[:, config.AFFINITY_DIM:]

        scaler_affinity = StandardScaler().fit(affinity_features)
        scaler_contextual = StandardScaler().fit(contextual_features)
        logging.info("Escaladores ajustados.")
        return scaler_affinity, scaler_contextual

    def predict(self, feature_vector: np.ndarray) -> float:
        # Dividir correctamente el vector de 74 características.
        affinity_part = feature_vector[:, :config.AFFINITY_DIM]
        contextual_part = feature_vector[:, config.AFFINITY_DIM:]

        # Aplicar la misma normalización que en el entrenamiento
        affinity_scaled = self.scaler_affinity.transform(affinity_part)
        contextual_scaled = self.scaler_contextual.transform(contextual_part)

        # Realizar la predicción
        prediction = self.model.predict([affinity_scaled, contextual_scaled], verbose=0)
        return prediction.flatten()[0]

if __name__ == '__main__':
    utils.ensure_dirs_exist()
    mi_combinacion = [17,19,20,34,35,37]
    print(f"\nEvaluando la combinación: {sorted(mi_combinacion)}")
    extractor = FeatureExtractor()
    feature_vector = extractor.create_feature_vector(mi_combinacion)
    predictor = Predictor()
    probability = predictor.predict(feature_vector)
    print("\n--- RESULTADO DE LA PREDICCIÓN ---")
    print(f"Combinación: {sorted(mi_combinacion)}")
    print(f"Probabilidad de ser ganadora: {probability:.6f} ({probability*100:.2f}%)")
    if probability > 0.5:
        print("Diagnóstico: El modelo considera que esta combinación tiene ALTAS probabilidades de ser ganadora.")
    else:
        print("Diagnóstico: El modelo considera que esta combinación se parece a una ALEATORIA (ruido).")