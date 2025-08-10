#!/usr/bin/env python3
"""
Proyecto Crisálida - Fase 2: Ingeniería de Características Híbrida (Versión Corregida)
Fusión del Top 10 contextual con métricas de Afinidad Combinatoria (pares, tercias, cuartetos, quintetos)
"""

import numpy as np
import pandas as pd
import json
from collections import Counter
from itertools import combinations
from typing import List, Dict, Tuple, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridFeatureEngineer:
    """Ingeniero de características híbridas para el Transformer-Omega"""
    
    def __init__(self, winners_path: str, noise_path: str, top10_mapping_path: str):
        self.winners_path = winners_path
        self.noise_path = noise_path
        self.top10_mapping_path = top10_mapping_path
        
        # Datos
        self.winners_data = None
        self.noise_data = None
        self.top10_mapping = None
        
        # Frecuencias de afinidad
        self.affinity_frequencies = {}
        
        # Características finales
        self.hybrid_features = None
        
    def load_data(self):
        """Carga los datasets y el mapeo Top 10"""
        logging.info("Cargando datasets y configuración...")
        
        # Cargar datasets (sin headers, 6 columnas)
        self.winners_data = pd.read_csv(self.winners_path, header=None, 
                                      names=['num1', 'num2', 'num3', 'num4', 'num5', 'num6'])
        self.noise_data = pd.read_csv(self.noise_path, header=None,
                                    names=['num1', 'num2', 'num3', 'num4', 'num5', 'num6'])
        
        logging.info(f"Combinaciones ganadoras: {len(self.winners_data)}")
        logging.info(f"Combinaciones aleatorias: {len(self.noise_data)}")
        
        # Cargar mapeo Top 10
        with open(self.top10_mapping_path, 'r') as f:
            self.top10_mapping = json.load(f)
        
        logging.info("Top 10 características contextuales cargadas")
        
    def calculate_affinity_frequencies(self, levels: List[int] = [2, 3, 4, 5]):
        """Calcula frecuencias de afinidad para todos los niveles"""
        logging.info(f"Calculando frecuencias de afinidad para niveles: {levels}")
        
        # Convertir combinaciones ganadoras a listas
        winners_combinations = []
        for _, row in self.winners_data.iterrows():
            combo = [row['num1'], row['num2'], row['num3'], row['num4'], row['num5'], row['num6']]
            winners_combinations.append(sorted(combo))
        
        # Calcular frecuencias por nivel
        self.affinity_frequencies = {level: Counter() for level in levels}
        
        for combo in winners_combinations:
            for level in levels:
                if len(combo) >= level:
                    # Generar todas las combinaciones de ese nivel
                    for sub_combo in combinations(combo, level):
                        self.affinity_frequencies[level][sub_combo] += 1
        
        # Estadísticas
        for level in levels:
            total_combinations = len(self.affinity_frequencies[level])
            max_frequency = max(self.affinity_frequencies[level].values()) if self.affinity_frequencies[level] else 0
            logging.info(f"  Nivel {level}: {total_combinations} combinaciones únicas, frecuencia máxima: {max_frequency}")
        
        logging.info("Frecuencias de afinidad calculadas exitosamente")
        
    def calculate_affinity_for_combination(self, combination: List[int]) -> Dict[str, float]:
        """Calcula todas las métricas de afinidad para una combinación"""
        sorted_combo = sorted(combination)
        affinities = {}
        
        # Afinidades básicas (suma de frecuencias)
        for level in [2, 3, 4, 5]:
            total_affinity = 0
            if len(sorted_combo) >= level:
                for sub_combo in combinations(sorted_combo, level):
                    total_affinity += self.affinity_frequencies[level].get(sub_combo, 0)
            affinities[f'affinity_level_{level}'] = total_affinity
        
        # Afinidades normalizadas (por número de combinaciones posibles)
        from math import comb
        for level in [2, 3, 4, 5]:
            if len(sorted_combo) >= level:
                max_combinations = comb(len(sorted_combo), level)
                normalized_affinity = affinities[f'affinity_level_{level}'] / max_combinations if max_combinations > 0 else 0
                affinities[f'affinity_level_{level}_normalized'] = normalized_affinity
            else:
                affinities[f'affinity_level_{level}_normalized'] = 0
        
        # Métricas derivadas de la Clase Omega
        
        # 1. Afinidad promedio ponderada
        weights = {2: 0.4, 3: 0.3, 4: 0.2, 5: 0.1}  # Pesos decrecientes por nivel
        weighted_affinity = sum(affinities[f'affinity_level_{level}'] * weights[level] for level in [2, 3, 4, 5])
        affinities['affinity_weighted_average'] = weighted_affinity
        
        # 2. Score de coherencia (varianza de afinidades normalizadas)
        normalized_values = [affinities[f'affinity_level_{level}_normalized'] for level in [2, 3, 4, 5]]
        coherence_score = np.var(normalized_values)
        affinities['affinity_coherence'] = coherence_score
        
        # 3. Afinidad máxima por nivel
        max_affinity = max(affinities[f'affinity_level_{level}'] for level in [2, 3, 4, 5])
        affinities['affinity_max'] = max_affinity
        
        # 4. Ratio de afinidad alta vs baja
        high_levels = affinities['affinity_level_4'] + affinities['affinity_level_5']
        low_levels = affinities['affinity_level_2'] + affinities['affinity_level_3']
        ratio_high_low = high_levels / (low_levels + 1e-8)  # Evitar división por cero
        affinities['affinity_ratio_high_low'] = ratio_high_low
        
        # 5. Score de densidad (afinidad total / suma de números)
        total_affinity = sum(affinities[f'affinity_level_{level}'] for level in [2, 3, 4, 5])
        number_sum = sum(sorted_combo)
        density_score = total_affinity / number_sum if number_sum > 0 else 0
        affinities['affinity_density'] = density_score
        
        return affinities
    
    def calculate_top10_contextual_features(self, combination: List[int]) -> Dict[str, List[float]]:
        """Calcula las características contextuales Top 10 para una combinación"""
        features = {f'feature_{i+1}': [] for i in range(10)}
        
        for number in combination:
            # 1. afinidad_pares (ya calculada en afinidades, pero necesaria para contexto individual)
            pair_affinity = 0
            for other_num in combination:
                if other_num != number:
                    pair = tuple(sorted([number, other_num]))
                    pair_affinity += self.affinity_frequencies[2].get(pair, 0)
            features['feature_1'].append(pair_affinity)
            
            # 2. frecuencia_historica
            # Calcular frecuencia del número en todas las combinaciones ganadoras
            freq = 0
            for _, row in self.winners_data.iterrows():
                combo = [row['num1'], row['num2'], row['num3'], row['num4'], row['num5'], row['num6']]
                if number in combo:
                    freq += 1
            features['feature_2'].append(freq / len(self.winners_data))
            
            # 3. term_6 (termina en 6)
            features['feature_3'].append(1.0 if number % 10 == 6 else 0.0)
            
            # 4. mult_3 (múltiplo de 3)
            features['feature_4'].append(1.0 if number % 3 == 0 else 0.0)
            
            # 5. term_9 (termina en 9)
            features['feature_5'].append(1.0 if number % 10 == 9 else 0.0)
            
            # 6. desv_media (desviación de la media de la combinación)
            combo_mean = np.mean(combination)
            deviation = abs(number - combo_mean) / combo_mean if combo_mean > 0 else 0
            features['feature_6'].append(deviation)
            
            # 7. decada_10_19 (segunda década)
            features['feature_7'].append(1.0 if 10 <= number <= 19 else 0.0)
            
            # 8. valor_normalizado
            features['feature_8'].append((number - 1) / 38.0)  # Normalizar [1,39] a [0,1]
            
            # 9. no_fibonacci
            fibonacci_numbers = {1, 1, 2, 3, 5, 8, 13, 21, 34}  # Fibonacci hasta 39
            features['feature_9'].append(0.0 if number in fibonacci_numbers else 1.0)
            
            # 10. es_fibonacci
            features['feature_10'].append(1.0 if number in fibonacci_numbers else 0.0)
        
        return features
    
    def create_hybrid_feature_vector(self, combination: List[int]) -> np.ndarray:
        """Crea el vector de características híbrido completo para una combinación"""
        
        # 1. Características de afinidad (a nivel de combinación)
        affinity_features = self.calculate_affinity_for_combination(combination)
        affinity_vector = list(affinity_features.values())
        
        # 2. Características contextuales Top 10 (a nivel de número individual)
        contextual_features = self.calculate_top10_contextual_features(combination)
        
        # Aplanar características contextuales (6 números x 10 características = 60 valores)
        contextual_vector = []
        for i in range(6):  # Para cada posición en la combinación
            for feature_key in sorted(contextual_features.keys()):
                contextual_vector.append(contextual_features[feature_key][i])
        
        # 3. Combinar vectores
        hybrid_vector = np.array(affinity_vector + contextual_vector)
        
        return hybrid_vector
    
    def process_all_combinations(self):
        """Procesa todas las combinaciones para crear el dataset híbrido"""
        logging.info("Procesando todas las combinaciones...")
        
        all_combinations = []
        all_labels = []
        
        # Procesar combinaciones ganadoras
        logging.info("Procesando combinaciones ganadoras...")
        for _, row in self.winners_data.iterrows():
            combo = [row['num1'], row['num2'], row['num3'], row['num4'], row['num5'], row['num6']]
            all_combinations.append(combo)
            all_labels.append(1)  # Etiqueta positiva
        
        # Procesar combinaciones aleatorias
        logging.info("Procesando combinaciones aleatorias...")
        for _, row in self.noise_data.iterrows():
            combo = [row['num1'], row['num2'], row['num3'], row['num4'], row['num5'], row['num6']]
            all_combinations.append(combo)
            all_labels.append(0)  # Etiqueta negativa
        
        # Crear vectores de características híbridas
        logging.info("Generando vectores de características híbridas...")
        hybrid_vectors = []
        
        for i, combo in enumerate(all_combinations):
            if i % 1000 == 0:
                logging.info(f"  Procesando combinación {i+1}/{len(all_combinations)}")
            
            hybrid_vector = self.create_hybrid_feature_vector(combo)
            hybrid_vectors.append(hybrid_vector)
        
        # Convertir a arrays numpy
        self.hybrid_features = np.array(hybrid_vectors)
        self.labels = np.array(all_labels)
        
        logging.info(f"Dataset híbrido creado: {self.hybrid_features.shape}")
        logging.info(f"Dimensiones: {self.hybrid_features.shape[1]} características por combinación")
        
    def analyze_feature_distribution(self):
        """Analiza la distribución de las características híbridas"""
        logging.info("Analizando distribución de características híbridas...")
        
        # Separar por clase
        winners_features = self.hybrid_features[self.labels == 1]
        noise_features = self.hybrid_features[self.labels == 0]
        
        # Estadísticas básicas
        feature_stats = {
            'total_features': self.hybrid_features.shape[1],
            'affinity_features': 14,  # 4 básicas + 4 normalizadas + 6 derivadas
            'contextual_features': 60,  # 6 números x 10 características
            'winners_samples': len(winners_features),
            'noise_samples': len(noise_features)
        }
        
        # Análisis de separabilidad por característica
        separability_scores = []
        for i in range(self.hybrid_features.shape[1]):
            winners_mean = np.mean(winners_features[:, i])
            noise_mean = np.mean(noise_features[:, i])
            winners_std = np.std(winners_features[:, i])
            noise_std = np.std(noise_features[:, i])
            
            # Cohen's d
            pooled_std = np.sqrt(((len(winners_features) - 1) * winners_std**2 + 
                                (len(noise_features) - 1) * noise_std**2) / 
                               (len(winners_features) + len(noise_features) - 2))
            
            cohens_d = abs(winners_mean - noise_mean) / pooled_std if pooled_std > 0 else 0
            separability_scores.append(cohens_d)
        
        feature_stats['separability_scores'] = separability_scores
        feature_stats['mean_separability'] = np.mean(separability_scores)
        feature_stats['max_separability'] = np.max(separability_scores)
        
        # Top 10 características más separables
        top_indices = np.argsort(separability_scores)[-10:][::-1]
        feature_stats['top10_separable'] = [
            {'index': int(idx), 'separability': float(separability_scores[idx])} 
            for idx in top_indices
        ]
        
        logging.info(f"Separabilidad promedio: {feature_stats['mean_separability']:.4f}")
        logging.info(f"Separabilidad máxima: {feature_stats['max_separability']:.4f}")
        
        return feature_stats
    
    def save_hybrid_dataset(self):
        """Guarda el dataset híbrido procesado"""
        logging.info("Guardando dataset híbrido...")
        
        # Guardar características y etiquetas
        np.save('../data/hybrid_features.npy', self.hybrid_features)
        np.save('../data/hybrid_labels.npy', self.labels)
        
        # Crear metadatos
        metadata = {
            'feature_dimensions': int(self.hybrid_features.shape[1]),
            'total_samples': int(self.hybrid_features.shape[0]),
            'winners_samples': int(np.sum(self.labels == 1)),
            'noise_samples': int(np.sum(self.labels == 0)),
            'affinity_features': 14,
            'contextual_features': 60,
            'feature_breakdown': {
                'affinity_basic': [0, 1, 2, 3],  # levels 2,3,4,5
                'affinity_normalized': [4, 5, 6, 7],  # normalized versions
                'affinity_derived': [8, 9, 10, 11, 12, 13],  # weighted, coherence, max, ratio, density
                'contextual_top10': list(range(14, 74))  # 6 numbers x 10 features
            },
            'feature_names': {
                'affinity': [
                    'affinity_level_2', 'affinity_level_3', 'affinity_level_4', 'affinity_level_5',
                    'affinity_level_2_normalized', 'affinity_level_3_normalized', 
                    'affinity_level_4_normalized', 'affinity_level_5_normalized',
                    'affinity_weighted_average', 'affinity_coherence', 'affinity_max',
                    'affinity_ratio_high_low', 'affinity_density'
                ],
                'contextual': [
                    'afinidad_pares_individual', 'frecuencia_historica', 'term_6', 'mult_3',
                    'term_9', 'desv_media', 'decada_10_19', 'valor_normalizado',
                    'no_fibonacci', 'es_fibonacci'
                ]
            }
        }
        
        with open('../data/hybrid_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info("Dataset híbrido guardado exitosamente")
        logging.info(f"  Características: ../data/hybrid_features.npy")
        logging.info(f"  Etiquetas: ../data/hybrid_labels.npy")
        logging.info(f"  Metadatos: ../data/hybrid_metadata.json")
        
    def generate_phase2_report(self):
        """Genera reporte de la Fase 2"""
        logging.info("Generando reporte de la Fase 2...")
        
        stats = self.analyze_feature_distribution()
        
        report = f"""# Proyecto Crisálida - Fase 2: Ingeniería de Características Híbrida

## Resumen Ejecutivo

Se ha completado exitosamente la fusión del Top 10 de características contextuales con las métricas de Afinidad Combinatoria, creando un vector de características híbrido de {stats['total_features']} dimensiones para el modelo Transformer-Omega.

## Arquitectura de Características Híbridas

### Métricas de Afinidad Combinatoria ({stats['affinity_features']} características)

**Afinidades Básicas (4 características):**
- Afinidad de Pares (Nivel 2)
- Afinidad de Tercias (Nivel 3) 
- Afinidad de Cuartetos (Nivel 4)
- **Afinidad de Quintetos (Nivel 5)** - ¡INNOVACIÓN!

**Afinidades Normalizadas (4 características):**
- Versiones normalizadas por número de combinaciones posibles

**Métricas Derivadas de la Clase Omega (6 características):**
- Afinidad promedio ponderada
- Score de coherencia (varianza de afinidades)
- Afinidad máxima por nivel
- Ratio afinidad alta vs baja
- Score de densidad (afinidad/suma de números)

### Características Contextuales Top 10 ({stats['contextual_features']} características)

Aplicadas a cada uno de los 6 números de la combinación:
1. Afinidad de pares individual
2. Frecuencia histórica
3. Terminación en 6
4. Múltiplo de 3 (innovación Manus)
5. Terminación en 9
6. Desviación de la media (innovación Manus)
7. Década 10-19
8. Valor normalizado
9. No Fibonacci
10. Es Fibonacci

## Estadísticas del Dataset Híbrido

- **Total de muestras**: {stats['total_samples']:,}
- **Combinaciones ganadoras**: {stats['winners_samples']:,}
- **Combinaciones aleatorias**: {stats['noise_samples']:,}
- **Dimensiones por muestra**: {stats['total_features']}

## Análisis de Separabilidad

- **Separabilidad promedio**: {stats['mean_separability']:.4f}
- **Separabilidad máxima**: {stats['max_separability']:.4f}

### Top 5 Características Más Discriminativas:
"""
        
        for i, feature in enumerate(stats['top10_separable'][:5]):
            report += f"{i+1}. Característica {feature['index']}: {feature['separability']:.4f}\n"
        
        report += f"""
## Innovaciones Implementadas

1. **Afinidad de Quintetos**: Primera implementación de afinidades de nivel 5
2. **Métricas de Coherencia**: Score que mide consistencia entre niveles de afinidad
3. **Fusión Inteligente**: Combinación de características globales (afinidad) y locales (contextuales)
4. **Normalización Adaptativa**: Afinidades normalizadas por complejidad combinatoria

## Próximos Pasos para la Fase 3

- Diseñar arquitectura Transformer-Omega con mecanismos especializados
- Implementar atención diferenciada para características de afinidad vs contextuales
- Crear sistema de fusión adaptativa con gating inteligente
- Optimizar para el vector híbrido de {stats['total_features']} dimensiones

## Conclusiones

El vector híbrido combina exitosamente la potencia discriminativa de las afinidades (AUC > 0.99) con la riqueza contextual del Top 10. La separabilidad promedio de {stats['mean_separability']:.4f} sugiere un potencial significativo para el modelo Transformer-Omega.
"""
        
        # Guardar reporte
        with open('../reports/fase2_hybrid_engineering.md', 'w') as f:
            f.write(report)
        
        logging.info("Reporte guardado en ../reports/fase2_hybrid_engineering.md")
        
        return report

def main():
    """Función principal de la Fase 2"""
    logging.info("=== PROYECTO CRISÁLIDA - FASE 2: INGENIERÍA HÍBRIDA ===")
    
    # Inicializar ingeniero
    engineer = HybridFeatureEngineer(
        winners_path='../data/winners_dataset.csv',
        noise_path='../data/noise_dataset.csv',
        top10_mapping_path='../results/top10_feature_mapping.json'
    )
    
    # Cargar datos
    engineer.load_data()
    
    # Calcular frecuencias de afinidad
    engineer.calculate_affinity_frequencies(levels=[2, 3, 4, 5])
    
    # Procesar todas las combinaciones
    engineer.process_all_combinations()
    
    # Analizar distribución
    stats = engineer.analyze_feature_distribution()
    
    # Guardar dataset
    engineer.save_hybrid_dataset()
    
    # Generar reporte
    report = engineer.generate_phase2_report()
    
    logging.info("=== FASE 2 COMPLETADA EXITOSAMENTE ===")
    logging.info(f"Dataset híbrido: {engineer.hybrid_features.shape}")
    logging.info(f"Separabilidad promedio: {stats['mean_separability']:.4f}")
    logging.info("Listo para Fase 3: Diseño del Transformer-Omega")

if __name__ == "__main__":
    main()

