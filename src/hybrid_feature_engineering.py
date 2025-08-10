#!/usr/bin/env python3
"""
Proyecto Crisálida - Fase 2: Ingeniería de Características Híbrida (Versión Final Optimizada)
Fusión eficiente del Top 10 contextual con métricas de Afinidad Combinatoria.
Versión oficial del proyecto, integrada con config.py y utils.py.
"""

import numpy as np
import pandas as pd
import json
from collections import Counter
from itertools import combinations
from typing import List, Dict, Any
import logging
from math import comb
import time

# --- CAMBIO 1: Importar la configuración central y las utilidades ---
import config
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridFeatureEngineer:
    """
    Ingeniero de características híbridas, optimizado y totalmente integrado.
    """
    
    # --- CAMBIO 2: El constructor ahora usa las rutas de config.py por defecto ---
    def __init__(self,
                 winners_path=config.WINNERS_DATASET_PATH,
                 noise_path=config.NOISE_DATASET_PATH):
        
        self.winners_path = winners_path
        self.noise_path = noise_path
        
        # Datos
        self.winners_data = None
        self.noise_data = None
        
        # Estructuras pre-calculadas para optimización
        self.affinity_frequencies: Dict[int, Counter] = {}
        self.historical_frequencies: Dict[int, float] = {}
        
        # Características finales
        self.hybrid_features = None
        self.labels = None

    def load_data(self):
        """Carga los datasets desde las rutas de configuración."""
        logging.info("Cargando datasets...")
        self.winners_data = pd.read_csv(self.winners_path, header=None).values.tolist()
        self.noise_data = pd.read_csv(self.noise_path, header=None).values.tolist()
        logging.info(f"Cargadas {len(self.winners_data)} combinaciones ganadoras y {len(self.noise_data)} aleatorias.")

    def precompute_structures(self):
        """Pre-calcula estructuras de datos para acelerar la ingeniería de características."""
        logging.info("Pre-calculando estructuras para optimización...")
        
        # 1. Frecuencias de afinidad
        levels = config.AFFINITY_LEVELS
        self.affinity_frequencies = {level: Counter() for level in levels}
        for combo in self.winners_data:
            sorted_combo = sorted(combo)
            for level in levels:
                for sub_combo in combinations(sorted_combo, level):
                    self.affinity_frequencies[level][sub_combo] += 1
        
        # 2. Frecuencias históricas de números individuales
        all_winner_numbers = [num for combo in self.winners_data for num in combo]
        number_counts = Counter(all_winner_numbers)
        total_draws = len(self.winners_data)
        self.historical_frequencies = {num: count / total_draws for num, count in number_counts.items()}
        
        logging.info("Estructuras de afinidad y frecuencias históricas pre-calculadas.")

    def _create_affinity_vector(self, combination: List[int]) -> np.ndarray:
        """Crea el vector de características de afinidad para una combinación."""
        sorted_combo = sorted(combination)
        features = np.zeros(config.AFFINITY_DIM)
        
        # Afinidades básicas (suma de frecuencias)
        affinities = [sum(self.affinity_frequencies[level].get(sub, 0) for sub in combinations(sorted_combo, level)) for level in config.AFFINITY_LEVELS]
        features[0:4] = affinities
        
        # Afinidades normalizadas
        norm_affinities = [aff / comb(6, level) for aff, level in zip(affinities, config.AFFINITY_LEVELS)]
        features[4:8] = norm_affinities
        
        # Métricas derivadas
        weights = {2: 0.4, 3: 0.3, 4: 0.2, 5: 0.1}
        features[8] = sum(affinities[i] * weights[level] for i, level in enumerate(config.AFFINITY_LEVELS)) # Weighted Avg
        features[9] = np.var(norm_affinities) # Coherence
        features[10] = np.max(affinities) # Max Affinity
        features[11] = (affinities[2] + affinities[3]) / (affinities[0] + affinities[1] + 1e-8) # High/Low Ratio
        features[12] = sum(affinities) / sum(sorted_combo) if sum(sorted_combo) > 0 else 0 # Density
        features[13] = np.mean(norm_affinities) # Avg Normalized
        
        return features

    def _create_contextual_vector(self, combination: List[int]) -> np.ndarray:
        """Crea el vector de características contextuales para una combinación."""
        features = np.zeros(config.CONTEXTUAL_DIM)
        combo_mean = np.mean(combination)
        
        for i, number in enumerate(combination):
            base_idx = i * 10
            
            # 1. Afinidad de pares individual
            features[base_idx + 0] = sum(self.affinity_frequencies[2].get(tuple(sorted((number, other))), 0) for other in combination if other != number)
            # 2. Frecuencia histórica (pre-calculada)
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
            # 8. Valor normalizado (usando constantes de config)
            features[base_idx + 7] = (number - config.NUMBER_RANGE_MIN) / (config.NUMBER_RANGE_MAX - config.NUMBER_RANGE_MIN)
            # 9 & 10. Fibonacci
            is_fib = number in config.FIBONACCI_NUMBERS
            features[base_idx + 8] = 0.0 if is_fib else 1.0 # no_fibonacci
            features[base_idx + 9] = 1.0 if is_fib else 0.0 # es_fibonacci

        return features

    def process_combinations(self, batch_size: int = 1000):
        """Procesa todas las combinaciones en lotes para crear el dataset híbrido."""
        logging.info("Procesando todas las combinaciones en lotes...")
        
        all_combinations = self.winners_data + self.noise_data
        all_labels = [1] * len(self.winners_data) + [0] * len(self.noise_data)
        total_combinations = len(all_combinations)
        
        feature_vectors = []
        
        for i in range(0, total_combinations, batch_size):
            batch_combos = all_combinations[i:i + batch_size]
            
            if i % (batch_size * 5) == 0:
                logging.info(f"  Procesando lote {i//batch_size + 1}/{(total_combinations-1)//batch_size + 1}...")
            
            batch_vectors = [
                np.hstack([self._create_affinity_vector(combo), self._create_contextual_vector(combo)])
                for combo in batch_combos
            ]
            feature_vectors.extend(batch_vectors)
            
        self.hybrid_features = np.array(feature_vectors)
        self.labels = np.array(all_labels)
        
        logging.info(f"Dataset híbrido creado: {self.hybrid_features.shape}")
        
    def analyze_feature_separability(self) -> Dict[str, Any]:
        """Analiza la capacidad de las características para separar las clases."""
        logging.info("Analizando separabilidad de características (Cohen's d)...")
        winners_features = self.hybrid_features[self.labels == 1]
        noise_features = self.hybrid_features[self.labels == 0]
        
        separability_scores = []
        for i in range(self.hybrid_features.shape[1]):
            mean_w, std_w = np.mean(winners_features[:, i]), np.std(winners_features[:, i])
            mean_n, std_n = np.mean(noise_features[:, i]), np.std(noise_features[:, i])
            
            # Cohen's d
            pooled_std = np.sqrt(((len(winners_features) - 1) * std_w**2 + (len(noise_features) - 1) * std_n**2) / (len(self.hybrid_features) - 2))
            cohens_d = abs(mean_w - mean_n) / pooled_std if pooled_std > 0 else 0
            separability_scores.append(cohens_d)
        
        stats = {
            'mean_separability': np.mean(separability_scores),
            'max_separability': np.max(separability_scores),
            'top10_separable_indices': np.argsort(separability_scores)[-10:][::-1].tolist()
        }
        logging.info(f"Separabilidad promedio: {stats['mean_separability']:.4f}, Máxima: {stats['max_separability']:.4f}")
        return stats

    def save_artifacts(self):
        """Guarda el dataset híbrido y los metadatos usando rutas de config."""
        logging.info("Guardando artefactos de ingeniería de características...")
        
        # Guardar dataset
        np.save(config.HYBRID_FEATURES_PATH, self.hybrid_features)
        np.save(config.HYBRID_LABELS_PATH, self.labels)
        
        # Crear y guardar metadatos
        metadata = {
            'creation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': self.hybrid_features.shape[0],
            'feature_dimensions': self.hybrid_features.shape[1],
            'affinity_dim': config.AFFINITY_DIM,
            'contextual_dim': config.CONTEXTUAL_DIM,
            'feature_names': { # --- CAMBIO 3: Nombres de características más explícitos y correctos ---
                'affinity': [
                    'affinity_level_2', 'affinity_level_3', 'affinity_level_4', 'affinity_level_5',
                    'affinity_level_2_norm', 'affinity_level_3_norm', 'affinity_level_4_norm', 'affinity_level_5_norm',
                    'affinity_weighted_avg', 'affinity_coherence', 'affinity_max',
                    'affinity_ratio_high_low', 'affinity_density', 'affinity_avg_norm'
                ],
                'contextual_per_number': [ # Top 10 aplicados a cada número
                    'afinidad_pares_individual', 'frecuencia_historica', 'term_6', 'mult_3',
                    'term_9', 'desv_media', 'decada_10_19', 'valor_normalizado',
                    'no_fibonacci', 'es_fibonacci'
                ]
            }
        }
        with open(config.HYBRID_METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logging.info(f"Artefactos guardados en el directorio '{config.DATA_DIR}'.")

# --- CAMBIO 4: La función main ahora es un orquestador limpio y directo ---
def main():
    """Función principal para ejecutar la ingeniería de características híbridas."""
    logging.info("=== PROYECTO CRISÁLIDA - FASE 2: INGENIERÍA HÍBRIDA (OPTIMIZADA) ===")
    
    # 1. Asegurar que los directorios necesarios existen
    utils.ensure_dirs_exist()
    
    # 2. Inicializar el ingeniero
    engineer = HybridFeatureEngineer()
    
    # 3. Ejecutar los pasos en orden
    engineer.load_data()
    engineer.precompute_structures()
    engineer.process_combinations(batch_size=1000)
    stats = engineer.analyze_feature_separability()
    engineer.save_artifacts()
    
    logging.info("=== FASE 2 COMPLETADA EXITOSAMENTE ===")
    logging.info(f"Dataset híbrido final: {engineer.hybrid_features.shape}")
    logging.info(f"Separabilidad promedio (Cohen's d): {stats['mean_separability']:.4f}")
    logging.info("¡Listo para Fase 3: Entrenamiento del Transformer-Omega!")

if __name__ == "__main__":
    main()