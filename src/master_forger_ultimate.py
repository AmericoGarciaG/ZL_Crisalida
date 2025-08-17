# src/master_forger_ultimate.py

import numpy as np
import logging
import os
import time
from typing import List, Tuple, Dict, Set, Optional
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- Módulos del Proyecto ---
import config
import utils
from predict_combination import FeatureExtractor, Predictor
from diversity_metrics import DiversityAnalyzer
from advanced_strategies import AdvancedGeneticStrategies

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# --- Lógica del Worker ---
worker_context = {}

def initialize_worker(extractor: FeatureExtractor, predictor: Predictor):
    pid = os.getpid()
    seed = (int(time.time() * 1000) + pid) % (2**32)
    worker_context['rng'] = np.random.default_rng(seed)
    worker_context['feature_extractor'] = extractor
    worker_context['predictor'] = predictor

def evaluate_fitness_task(combination: List[int]) -> Tuple[Tuple[int, ...], float]:
    extractor = worker_context['feature_extractor']
    predictor = worker_context['predictor']
    combo_tuple = tuple(sorted(combination))
    feature_vector = extractor.create_feature_vector(list(combo_tuple))
    score = predictor.predict(feature_vector)
    return combo_tuple, score

# --- Clase Principal del Algoritmo Genético Definitivo ---
class MasterForgerUltimate:
    def __init__(self, forger_config: Dict):
        self.config = forger_config
        self.rng = np.random.default_rng()
        self.cache: Dict[Tuple[int, ...], float] = {}
        self.historical_winners: Set[Tuple[int, ...]] = self._load_historical_winners()
        
        self.diversity_analyzer = DiversityAnalyzer()
        self.advanced_strategies = AdvancedGeneticStrategies(forger_config)
        self.generation_counter = 0

    def _load_historical_winners(self) -> Set[Tuple[int, ...]]:
        df = pd.read_csv(config.WINNERS_DATASET_PATH, header=None)
        return {tuple(sorted(row)) for row in df.itertuples(index=False)}

    def evaluate_population_parallel(self, population: List[List[int]], executor: ProcessPoolExecutor) -> List[float]:
        fitness_scores_map = {}
        population_tuples = [tuple(sorted(c)) for c in population]
        unique_combos_to_eval = []

        for combo_tuple in population_tuples:
            if combo_tuple in self.cache:
                fitness_scores_map[combo_tuple] = self.cache[combo_tuple]
            elif combo_tuple in self.historical_winners:
                fitness_scores_map[combo_tuple] = 0.0
            else:
                if combo_tuple not in [tuple(c) for c in unique_combos_to_eval]:
                    unique_combos_to_eval.append(list(combo_tuple))
        
        if unique_combos_to_eval:
            results = list(executor.map(evaluate_fitness_task, unique_combos_to_eval))
            for combo_tuple_res, score in results:
                fitness_scores_map[combo_tuple_res] = score
                self.cache[combo_tuple_res] = score
        
        return [fitness_scores_map.get(combo_tuple, 0.0) for combo_tuple in population_tuples]

    # --- LA CORRECCIÓN CLAVE ESTÁ AQUÍ: Mover la lógica a sus propios métodos ---
    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Crossover uniforme."""
        child_genes = {parent1[i] if self.rng.random() < 0.5 else parent2[i] for i in range(6)}
        while len(child_genes) < 6:
            child_genes.add(self.rng.integers(1, 40))
        return sorted(list(child_genes))

    def _mutate(self, child: List[int], mutation_rate: float) -> List[int]:
        """Mutación simple."""
        if self.rng.random() < mutation_rate:
            mutated_child = list(child)
            idx_to_mutate = self.rng.integers(0, 6)
            while True:
                new_gene = self.rng.integers(1, 40)
                if new_gene not in mutated_child:
                    mutated_child[idx_to_mutate] = new_gene
                    break
            return sorted(mutated_child)
        return child

    def run_evolution_cycle(self, executor: ProcessPoolExecutor, initial_population: Optional[List[List[int]]] = None, cycle_id: int = 1) -> Tuple[Optional[List[int]], float, List[List[int]]]:
        population = initial_population or self.advanced_strategies.create_diverse_initial_population(self.config['population_size'], self.historical_winners)
        
        best_in_cycle_combo: Optional[List[int]] = None
        best_in_cycle_score = -1.0
        
        mutation_rate = self.config['mutation_rate']

        pbar_desc = f"Ciclo {cycle_id} - Evolución Ultimate"
        pbar = tqdm(range(self.config['generations']), desc=pbar_desc)

        for gen in pbar:
            self.generation_counter += 1
            fitness_scores = self.evaluate_population_parallel(population, executor)
            
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_in_cycle_score:
                best_in_cycle_score = fitness_scores[best_idx]
                best_in_cycle_combo = population[best_idx]

            analysis = self.diversity_analyzer.analyze_population(population, self.generation_counter, best_in_cycle_score)
            if analysis.get('hamming_diversity', 1.0) < self.config.get('diversity_threshold', 0.15):
                mutation_rate = self.config.get('high_mutation_rate', 0.4)
            else:
                mutation_rate = self.config['mutation_rate']

            pbar.set_postfix({
                "Mejor(Ciclo)": f"{best_in_cycle_score:.4f}",
                "Div": f"{analysis.get('hamming_diversity', 0):.3f}",
                "Mut": f"{mutation_rate:.2f}"
            })
            
            if best_in_cycle_score >= self.config['early_stopping_threshold']: break

            elite_size = max(1, int(self.config['population_size'] * 0.05))
            sorted_indices = np.argsort(fitness_scores)[::-1]
            new_population = [population[i] for i in sorted_indices[:elite_size]]
            
            while len(new_population) < self.config['population_size']:
                # Selección por Torneo
                p1_indices = self.rng.choice(len(population), self.config['tournament_size'], replace=False)
                parent1 = population[max(p1_indices, key=lambda i: fitness_scores[i])]
                
                p2_indices = self.rng.choice(len(population), self.config['tournament_size'], replace=False)
                parent2 = population[max(p2_indices, key=lambda i: fitness_scores[i])]
                
                # --- LA CORRECCIÓN CLAVE: Llamar a los métodos de la clase ---
                child = self._crossover(parent1, parent2)
                child = self._mutate(child, mutation_rate)

                new_population.append(child)
            population = new_population

        pbar.close()
        return best_in_cycle_combo, best_in_cycle_score, population

def main():
    # ... (La función main no cambia)
    logging.info("=== MASTER FORGER ULTIMATE - VERSIÓN DEFINITIVA ===")
    utils.ensure_dirs_exist()
    
    ultimate_config = config.FORGER_CONFIG.copy()
    
    logging.info("Inicializando componentes del sistema...")
    main_extractor = FeatureExtractor()
    main_predictor = Predictor()
    logging.info("✓ Componentes inicializados correctamente")

    forger = MasterForgerUltimate(ultimate_config)
    
    best_overall_combo: Optional[List[int]] = None
    best_overall_score = -1.0
    elite_population: Optional[List[List[int]]] = None
    stagnation_counter = 0

    with ProcessPoolExecutor(max_workers=ultimate_config['num_workers'], initializer=initialize_worker, initargs=(main_extractor, main_predictor)) as executor:
        for cycle in range(ultimate_config['num_restarts']):
            print(f"\n{'='*80}\n    CICLO DE EVOLUCIÓN {cycle+1}/{ultimate_config['num_restarts']}\n{'='*80}")

            initial_pop = None
            if elite_population:
                if stagnation_counter >= ultimate_config.get('stagnation_limit', 2):
                    logging.warning("Estancamiento detectado. Activando reinicio con hipermutación.")
                    initial_pop = forger.advanced_strategies.adaptive_restart_population(elite_population, ultimate_config['population_size'], forger.historical_winners)
                    stagnation_counter = 0
                else:
                    new_random_size = ultimate_config['population_size'] - len(elite_population)
                    new_randoms = forger.advanced_strategies.create_diverse_initial_population(new_random_size, forger.historical_winners)
                    initial_pop = elite_population + new_randoms

            cycle_best, cycle_score, final_pop = forger.run_evolution_cycle(executor, initial_pop, cycle + 1)
            
            if cycle_score > best_overall_score:
                best_overall_score = cycle_score
                best_overall_combo = cycle_best
                stagnation_counter = 0
                logging.info(f"🎯 NUEVO RÉCORD GLOBAL: {best_overall_score:.6f} con {best_overall_combo}")
            else:
                stagnation_counter += 1
                logging.info(f"Fin de ciclo. No se mejoró el récord. Estancamiento: {stagnation_counter}/{ultimate_config.get('stagnation_limit', 2)}")
            
            final_fitness = forger.evaluate_population_parallel(final_pop, executor)
            sorted_indices = np.argsort(final_fitness)[::-1]
            elite_population = [final_pop[i] for i in sorted_indices[:ultimate_config['elite_size']]]

            if best_overall_score >= ultimate_config['early_stopping_threshold']: break
    
    output_dir = 'results/ultimate_analysis'
    os.makedirs(output_dir, exist_ok=True)
    forger.diversity_analyzer.plot_evolution_analysis(os.path.join(output_dir, 'evolution_analysis.png'))
    report_df = forger.diversity_analyzer.get_diversity_report_df()
    report_df.to_csv(os.path.join(output_dir, 'evolution_report.csv'), index=False)

    print(f"\n{'='*80}\n    🏆 MISIÓN COMPLETADA 🏆\n{'='*80}")
    if best_overall_combo:
        print(f"\n🎯 MEJOR COMBINACIÓN ENCONTRADA:\n   Números: {best_overall_combo}\n   Score: {best_overall_score:.6f} ({best_overall_score*100:.2f}%)")
    else:
        print("\n❌ No se encontró una combinación satisfactoria.")
    print(f"\n📁 Resultados detallados y gráficos guardados en: {output_dir}\n{'='*80}")

if __name__ == '__main__':
    main()