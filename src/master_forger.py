# src/master_forger.py

import numpy as np
import random # Lo mantenemos para random.seed en la versión antigua del worker si es necesario.
import logging
import os
import time
from typing import List, Tuple, Dict, Set, Optional
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from collections import Counter

# --- Módulos del Proyecto ---
import config
import utils
from predict_combination import FeatureExtractor, Predictor

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# --- Lógica del Worker (fuera de la clase para ser "picklable") ---
worker_context = {}

def initialize_worker(extractor: FeatureExtractor, predictor: Predictor):
    # Gestión de aleatoriedad robusta: generador independiente por worker
    pid = os.getpid()
    seed = (int(time.time() * 1000) + pid) % (2**32)
    
    # --- LA CORRECCIÓN CLAVE #1 ---
    # Usamos np.random.Generator, NO random.Generator
    worker_context['rng'] = np.random.Generator(np.random.PCG64(seed))
    
    worker_context['feature_extractor'] = extractor
    worker_context['predictor'] = predictor

def evaluate_fitness_task(combination: List[int]) -> Tuple[Tuple[int, ...], float]:
    extractor = worker_context['feature_extractor']
    predictor = worker_context['predictor']
    combo_tuple = tuple(sorted(combination))
    feature_vector = extractor.create_feature_vector(list(combo_tuple))
    score = predictor.predict(feature_vector)
    return combo_tuple, score

# --- Clase Principal del Algoritmo Genético ---
class MasterForger:
    def __init__(self, forger_config: Dict):
        self.config = forger_config
        # --- LA CORRECCIÓN CLAVE #2 ---
        # Usamos np.random.Generator aquí también
        self.rng = np.random.Generator(np.random.PCG64())
        
        self.cache: Dict[Tuple[int, ...], float] = {}
        self.historical_winners: Set[Tuple[int, ...]] = self._load_historical_winners()

    def _load_historical_winners(self) -> Set[Tuple[int, ...]]:
        winners_df = pd.read_csv(config.WINNERS_DATASET_PATH, header=None)
        return {tuple(sorted(row)) for row in winners_df.itertuples(index=False)}

    def evaluate_population_parallel(self, population: List[List[int]], executor: ProcessPoolExecutor) -> List[float]:
        fitness_scores_map = {}
        # Convertimos a tuplas ordenadas para que sean "hashables" y consistentes
        population_tuples = [tuple(sorted(c)) for c in population]
        unique_combos_to_eval = []

        for combo_tuple in population_tuples:
            if combo_tuple in self.cache:
                fitness_scores_map[combo_tuple] = self.cache[combo_tuple]
            elif combo_tuple in self.historical_winners:
                fitness_scores_map[combo_tuple] = 0.0
            else:
                if combo_tuple not in [tuple(c) for c in unique_combos_to_eval]: # Evitar duplicados en el lote
                    unique_combos_to_eval.append(list(combo_tuple))
        
        if unique_combos_to_eval:
            results = list(executor.map(evaluate_fitness_task, unique_combos_to_eval))
            for combo_tuple_res, score in results:
                fitness_scores_map[combo_tuple_res] = score
                self.cache[combo_tuple_res] = score
        
        return [fitness_scores_map.get(combo_tuple, 0.0) for combo_tuple in population_tuples]

    def _calculate_diversity(self, population: List[List[int]]) -> Tuple[float, int]:
        sample_size = min(len(population), 50)
        # --- LA CORRECCIÓN CLAVE #3 ---
        # Usamos self.rng.choice, que es el método del nuevo generador de NumPy
        sample_indices = self.rng.choice(len(population), size=sample_size, replace=False)
        sample = [population[i] for i in sample_indices]
        
        total_distance = 0
        pairs = 0
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                total_distance += len(set(sample[i]) ^ set(sample[j]))
                pairs += 1
        avg_hamming_dist = total_distance / pairs if pairs > 0 else 0

        unique_genes = len(set(g for combo in population for g in combo))
        return avg_hamming_dist, unique_genes

    def _create_individual(self) -> List[int]:
        while True:
            # --- LA CORRECCIÓN CLAVE #4 ---
            # Usamos self.rng.choice del generador de NumPy
            individual = sorted(list(self.rng.choice(
                range(config.NUMBER_RANGE_MIN, config.NUMBER_RANGE_MAX + 1), 
                6, 
                replace=False
            )))
            if tuple(individual) not in self.historical_winners:
                return individual

    def _crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        child_genes = set()
        for i in range(6):
            # --- LA CORRECCIÓN CLAVE #5 ---
            # Usamos self.rng.random() del generador de NumPy
            gene = p1[i] if self.rng.random() < 0.5 else p2[i]
            child_genes.add(gene)
        
        while len(child_genes) < 6:
            # Usamos self.rng.integers del generador de NumPy
            new_gene = self.rng.integers(config.NUMBER_RANGE_MIN, config.NUMBER_RANGE_MAX + 1)
            if new_gene not in child_genes: child_genes.add(new_gene)
        return sorted(list(child_genes))

    def _mutate(self, combo: List[int], mutation_rate: float) -> List[int]:
        if self.rng.random() < mutation_rate:
            mutated = list(combo)
            idx = self.rng.integers(0, 6) # El límite superior es exclusivo en NumPy
            while True:
                new_gene = self.rng.integers(config.NUMBER_RANGE_MIN, config.NUMBER_RANGE_MAX + 1)
                if new_gene not in mutated:
                    mutated[idx] = new_gene
                    return sorted(mutated)
        return combo

    def run_evolution(self, executor: ProcessPoolExecutor, initial_population: Optional[List[List[int]]] = None):
        # ... El resto de la clase es idéntica a la anterior y ya era correcta ...
        population = initial_population if initial_population else [self._create_individual() for _ in range(self.config['population_size'])]
        
        best_in_run_combo: Optional[List[int]] = None
        best_in_run_score = -1.0
        
        base_mutation_rate = self.config['mutation_rate']
        current_mutation_rate = base_mutation_rate

        for gen in (pbar := tqdm(range(self.config['generations']), desc="Evolución Genética")):
            fitness_scores = self.evaluate_population_parallel(population, executor)
            
            best_idx = np.argmax(fitness_scores)
            current_best_score = fitness_scores[best_idx]
            
            if current_best_score > best_in_run_score:
                best_in_run_score = current_best_score
                best_in_run_combo = population[best_idx]

            avg_dist, unique_genes = self._calculate_diversity(population)
            if avg_dist < self.config.get('diversity_threshold', 2.5):
                current_mutation_rate = self.config.get('high_mutation_rate', 0.5)
            else:
                current_mutation_rate = base_mutation_rate
            
            pbar.set_postfix({
                "Mejor (Run)": f"{best_in_run_score:.6f}", 
                "Diversidad": f"{avg_dist:.2f}", 
                "Mutación": f"{current_mutation_rate:.2f}"
            })

            if best_in_run_score >= self.config['early_stopping_threshold']: break

            new_population = [best_in_run_combo] # Elitismo
            while len(new_population) < self.config['population_size']:
                tournament1_indices = self.rng.choice(len(population), self.config['tournament_size'], replace=False)
                parent1 = population[max(tournament1_indices, key=lambda i: fitness_scores[i])]
                
                tournament2_indices = self.rng.choice(len(population), self.config['tournament_size'], replace=False)
                parent2 = population[max(tournament2_indices, key=lambda i: fitness_scores[i])]

                child = self._crossover(parent1, parent2)
                child = self._mutate(child, current_mutation_rate)
                new_population.append(child)
            
            population = new_population

        return best_in_run_combo, best_in_run_score, population

def main():
    # ... La función main no necesita cambios, ya que los cambios son internos a la clase ...
    utils.ensure_dirs_exist()
    
    forger_cfg = {
        'population_size': 200,
        'generations': 200,
        'mutation_rate': 0.1,
        'high_mutation_rate': 0.5,
        'diversity_threshold': 3.0,
        'tournament_size': 5,
        'early_stopping_threshold': 0.99,
        'num_restarts': 5,
        'stagnation_limit': 2,
        'elite_size': 10,
        'hypermutation_rate': 0.8
    }
    num_workers = os.cpu_count() or 4
    
    logging.info("--- INICIANDO EL FALSIFICADOR MAESTRO (v-Phoenix.2) ---")
    
    main_extractor = FeatureExtractor()
    main_predictor = Predictor()

    forger = MasterForger(forger_cfg)
    
    best_overall_combo: Optional[List[int]] = None
    best_overall_score = -1.0
    elite_population: Optional[List[List[int]]] = None
    stagnation_counter = 0

    with ProcessPoolExecutor(max_workers=num_workers, initializer=initialize_worker, initargs=(main_extractor, main_predictor)) as executor:
        for i in range(forger_cfg['num_restarts']):
            print("\n" + "="*60)
            logging.info(f"             INICIANDO CICLO DE EVOLUCIÓN {i+1}/{forger_cfg['num_restarts']}             ")
            print("="*60)

            next_initial_population = None
            if stagnation_counter >= forger_cfg['stagnation_limit'] and elite_population:
                logging.warning(f"Estancamiento detectado. Activando Hipermutación en la élite.")
                next_initial_population = []
                while len(next_initial_population) < forger_cfg['population_size']:
                    parent = random.choice(elite_population) # Usamos random aquí ya que es una operación simple.
                    mutated_clone = forger._mutate(parent, forger_cfg['hypermutation_rate'])
                    next_initial_population.append(mutated_clone)
                stagnation_counter = 0
            elif elite_population:
                new_randoms = [forger._create_individual() for _ in range(forger_cfg['population_size'] - forger_cfg['elite_size'])]
                next_initial_population = elite_population + new_randoms
            
            run_best_combo, run_best_score, final_population = forger.run_evolution(executor, initial_population=next_initial_population)
            
            if run_best_score > best_overall_score:
                best_overall_score = run_best_score
                best_overall_combo = run_best_combo
                logging.info(f"¡NUEVO MEJOR SCORE GLOBAL ENCONTRADO!: {best_overall_score:.6f}")
                stagnation_counter = 0
            else:
                logging.info(f"Fin del ciclo {i+1}. No se encontró un nuevo mejor global. (Estancamiento: {stagnation_counter+1}/{forger_cfg['stagnation_limit']})")
                stagnation_counter += 1

            final_fitness_scores = forger.evaluate_population_parallel(final_population, executor)
            sorted_indices = np.argsort(final_fitness_scores)[::-1]
            elite_population = [final_population[k] for k in sorted_indices[:forger_cfg['elite_size']]]
            
            if best_overall_score >= forger_cfg['early_stopping_threshold']: break

    print("\n" + "="*60)
    print("         ¡FALSIFICACIÓN MAESTRA PHOENIX COMPLETADA!         ")
    print("="*60)
    if best_overall_combo:
        print(f"\nLa mejor combinación **NUEVA** tras {i+1} ciclos de evolución es:")
        print(f"  > Combinación: {best_overall_combo}")
        print(f"  > Score de 'Ganadora' (predicción): {best_overall_score:.6f} ({best_overall_score*100:.2f}%)")
    else:
        print("\nNo se encontró una combinación satisfactoria.")
    print("="*60)

if __name__ == '__main__':
    main()