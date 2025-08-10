# src/predict_combination.py

import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow import keras  # type: ignore
from collections import Counter
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import logging
import os

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class FeatureExtractor:
    """
    Extrae el vector de 74 características híbridas para una combinación.
    """
    def __init__(self, winners_path='../data/winners_dataset.csv'):
        logging.info("Inicializando extractor de características...")
        self.winners_data = pd.read_csv(winners_path, header=None)
        self.affinity_frequencies = self._calculate_affinity_frequencies([2, 3, 4, 5])
        self.historical_frequencies = self._precalculate_historical_frequencies()
        self.fibonacci_numbers = {1, 2, 3, 5, 8, 13, 21, 34}
        logging.info("Extractor listo.")

    def _calculate_affinity_frequencies(self, levels):
        logging.info("Pre-calculando frecuencias de afinidad desde datos históricos...")
        freq_counters = {level: Counter() for level in levels}
        for _, row in self.winners_data.iterrows():
            combo = sorted(row.values.tolist())
            for level in levels:
                if len(combo) >= level:
                    freq_counters[level].update(combinations(combo, level))
        logging.info("Frecuencias de afinidad calculadas.")
        return freq_counters

    def _precalculate_historical_frequencies(self):
        logging.info("Pre-calculando frecuencias históricas de números...")
        all_numbers = self.winners_data.values.flatten()
        total_draws = len(self.winners_data)
        counts = Counter(all_numbers)
        return {num: count / total_draws for num, count in counts.items()}

    def create_feature_vector(self, combination: list) -> np.ndarray:
        """Genera el vector de 74 características para una combinación."""
        if len(combination) != 6:
            raise ValueError("La combinación debe tener 6 números.")
        
        # 1. Calcular características de afinidad (14 dimensiones)
        affinity_vec = self._get_affinity_features(combination)
        
        # 2. Calcular características contextuales (60 dimensiones)
        contextual_vec = self._get_contextual_features(combination)
        
        # 3. Combinar y devolver
        return np.hstack([affinity_vec, contextual_vec]).reshape(1, -1)

    def _get_affinity_features(self, combination):
        sorted_combo = sorted(combination)
        features = np.zeros(14)
        
        # Niveles 2, 3, 4, 5
        for i, level in enumerate([2, 3, 4, 5]):
            total_aff = sum(self.affinity_frequencies[level].get(sub, 0) for sub in combinations(sorted_combo, level))
            features[i] = total_aff

        # Normalizadas
        from math import comb
        for i, level in enumerate([2, 3, 4, 5]):
            max_combs = comb(6, level)
            features[i + 4] = features[i] / max_combs if max_combs > 0 else 0
        
        # Derivadas
        weights = {2: 0.4, 3: 0.3, 4: 0.2, 5: 0.1}
        features[8] = sum(features[k] * weights[level] for k, level in enumerate([2, 3, 4, 5]))
        features[9] = np.var(features[4:8]) # Coherencia
        features[10] = np.max(features[0:4]) # Máxima
        features[11] = (features[2] + features[3]) / (features[0] + features[1] + 1e-8) # Ratio
        features[12] = np.sum(features[0:4]) / sum(sorted_combo) # Densidad
        features[13] = np.mean(features[4:8]) # Promedio Normalizada

        return features

    def _get_contextual_features(self, combination):
        features = np.zeros(60)
        combo_mean = np.mean(combination)

        for i, number in enumerate(combination):
            base_idx = i * 10
            # 1. Afinidad de pares individual
            pair_aff = sum(self.affinity_frequencies[2].get(tuple(sorted((number, other))), 0) for other in combination if other != number)
            features[base_idx + 0] = pair_aff
            # 2. Frecuencia histórica
            features[base_idx + 1] = self.historical_frequencies.get(number, 0)
            # 3. Termina en 6
            features[base_idx + 2] = 1.0 if number % 10 == 6 else 0.0
            # 4. Múltiplo de 3
            features[base_idx + 3] = 1.0 if number % 3 == 0 else 0.0
            # 5. Termina en 9
            features[base_idx + 4] = 1.0 if number % 10 == 9 else 0.0
            # 6. Desviación de la media
            features[base_idx + 5] = abs(number - combo_mean) / combo_mean if combo_mean > 0 else 0
            # 7. Década 10-19
            features[base_idx + 6] = 1.0 if 10 <= number <= 19 else 0.0
            # 8. Valor normalizado
            features[base_idx + 7] = (number - 1) / 38.0
            # 9. No Fibonacci
            features[base_idx + 8] = 0.0 if number in self.fibonacci_numbers else 1.0
            # 10. Es Fibonacci
            features[base_idx + 9] = 1.0 if number in self.fibonacci_numbers else 0.0

        return features

class Predictor:
    """
    Carga el modelo y los escaladores para hacer predicciones.
    """
    def __init__(self, model_path='../models/transformer_omega_final.keras'):
        logging.info("Cargando modelo Transformer-Omega...")
        self.model = keras.models.load_model(model_path, compile=False)
        self.scaler_affinity, self.scaler_contextual = self._fit_scalers()
        logging.info("Modelo y escaladores listos.")

    def _fit_scalers(self):
        """
        Los escaladores deben ajustarse con los mismos datos que en el entrenamiento.
        """
        logging.info("Ajustando escaladores de normalización...")
        features = np.load('../data/hybrid_features.npy')
        affinity_features = features[:, :14]
        contextual_features = features[:, 14:]
        
        scaler_affinity = StandardScaler().fit(affinity_features)
        scaler_contextual = StandardScaler().fit(contextual_features)
        logging.info("Escaladores ajustados.")
        return scaler_affinity, scaler_contextual
    
    def predict(self, feature_vector: np.ndarray) -> float:
        """Realiza una predicción sobre un vector de características ya procesado."""
        affinity_part = feature_vector[:, :14]
        contextual_part = feature_vector[:, 14:]

        # Aplicar la misma normalización que en el entrenamiento
        affinity_scaled = self.scaler_affinity.transform(affinity_part)
        contextual_scaled = self.scaler_contextual.transform(contextual_part)

        # Realizar la predicción
        prediction = self.model.predict([affinity_scaled, contextual_scaled], verbose=0)
        return prediction.flatten()[0]

if __name__ == '__main__':
    # --- EJEMPLO DE USO ---
    
    # Combinación que quieres evaluar
    # Puedes cambiar estos números
    mi_combinacion = [8, 15, 16, 17, 18, 19] # Una combinación que fue ganadora
    #mi_combinacion = [1, 2, 3, 4, 5, 6]      # Una combinación muy improbable

    print(f"\nEvaluando la combinación: {sorted(mi_combinacion)}")

    # 1. Extraer características
    extractor = FeatureExtractor()
    feature_vector = extractor.create_feature_vector(mi_combinacion)
    
    # 2. Cargar modelo y predecir
    predictor = Predictor()
    probability = predictor.predict(feature_vector)

    # 3. Mostrar resultado
    print("\n--- RESULTADO DE LA PREDICCIÓN ---")
    print(f"Combinación: {sorted(mi_combinacion)}")
    print(f"Probabilidad de ser ganadora: {probability:.6f} ({probability*100:.2f}%)")
    
    if probability > 0.5:
        print("Diagnóstico: El modelo considera que esta combinación tiene ALTAS probabilidades de ser ganadora.")
    else:
        print("Diagnóstico: El modelo considera que esta combinación se parece a una ALEATORIA (ruido).")