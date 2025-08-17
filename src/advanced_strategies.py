# src/advanced_strategies.py

import numpy as np
import random
from typing import List, Dict, Tuple, Set

class AdvancedGeneticStrategies:
    """
    Estrategias avanzadas para mejorar la exploración y evitar convergencia prematura.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.rng = np.random.default_rng()

    def create_diverse_initial_population(self, size: int, historical_winners: Set[Tuple[int, ...]]) -> List[List[int]]:
        """Crea una población inicial maximizando la dispersión de números."""
        population = []
        number_counts = {i: 0 for i in range(1, 40)}
        
        while len(population) < size:
            # Seleccionar números con menos apariciones
            sorted_counts = sorted(number_counts.items(), key=lambda item: item[1])
            least_used_numbers = [num for num, count in sorted_counts]
            
            # Crear un individuo con los números menos usados
            new_combo_set = set()
            # Tomar los 4 menos usados
            for num in least_used_numbers:
                if len(new_combo_set) < 4:
                    new_combo_set.add(num)
            # Rellenar con aleatorios
            while len(new_combo_set) < 6:
                candidate = self.rng.integers(1, 40)
                if candidate not in new_combo_set:
                    new_combo_set.add(candidate)

            individual = sorted(list(new_combo_set))
            if tuple(individual) not in historical_winners:
                population.append(individual)
                for num in individual:
                    number_counts[num] += 1
        return population

    def adaptive_restart_population(self, elite: List[List[int]], size: int, historical_winners: Set[Tuple[int, ...]]) -> List[List[int]]:
        """Crea una población para reinicio, mezclando élite con nuevos individuos hipermutados."""
        elite_size = len(elite)
        new_random_size = size - elite_size
        
        # Clonar la élite y mutarla agresivamente (hipermutación)
        hypermutated_elite = []
        for individual in elite:
            mutated_clone = list(individual)
            # Mutar 2 o 3 genes
            num_mutations = self.rng.integers(2, 4)
            indices_to_mutate = self.rng.choice(6, num_mutations, replace=False)
            
            for idx in indices_to_mutate:
                while True:
                    new_gene = self.rng.integers(1, 40)
                    if new_gene not in mutated_clone:
                        mutated_clone[idx] = new_gene
                        break
            hypermutated_elite.append(sorted(mutated_clone))

        # Rellenar con nuevos individuos diversos
        new_diverse = self.create_diverse_initial_population(new_random_size, historical_winners)
        
        return hypermutated_elite + new_diverse
    