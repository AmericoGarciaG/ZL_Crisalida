# affinity_logic.py

from collections import Counter
from itertools import combinations
from typing import List, Dict, Tuple

def calculate_frequencies(
    winners_data: List[List[int]], 
    levels: List[int] = [2, 3, 4, 5]
) -> Dict[int, Counter]:
    """
    Calcula las frecuencias de subsecuencias para los niveles especificados.
    
    Args:
        winners_data: Una lista de listas, donde cada lista interna es una combinación ganadora.
        levels: Los tamaños de las subsecuencias a calcular (e.g., [2, 3] para pares y tercias).

    Returns:
        Un diccionario donde la clave es el nivel y el valor es un objeto Counter con las frecuencias.
    """
    freq_counters: Dict[int, Counter] = {level: Counter() for level in levels}
    for combo in winners_data:
        sorted_combo = sorted(combo)
        for level in levels:
            if len(sorted_combo) >= level:
                freq_counters[level].update(combinations(sorted_combo, level))
    return freq_counters

def calculate_affinities_for_combo(
    combination: List[int], 
    freq_counters: Dict[int, Counter]
) -> Dict[int, int]:
    """
    Calcula las afinidades de diferentes niveles para una única combinación.

    Args:
        combination: La combinación a evaluar (lista de 6 enteros).
        freq_counters: El diccionario de frecuencias pre-calculado por calculate_frequencies.

    Returns:
        Un diccionario donde la clave es el nivel y el valor es la afinidad calculada.
    """
    sorted_combo = sorted(combination)
    affinities: Dict[int, int] = {}
    for level, counter in freq_counters.items():
        total_affinity = 0
        if len(sorted_combo) >= level:
            for sub in combinations(sorted_combo, level):
                total_affinity += counter.get(sub, 0)
        affinities[level] = total_affinity
    return affinities

if __name__ == '__main__':
    # Ejemplo de uso
    
    # Supongamos que estos son nuestros ganadores históricos
    sample_winners = [
        [1, 2, 3, 10, 20, 30],
        [1, 2, 4, 11, 21, 31],
        [1, 3, 5, 10, 22, 32],
        [8, 15, 16, 17, 18, 19]
    ]
    
    print("--- 1. Calculando Frecuencias Globales ---")
    all_frequencies = calculate_frequencies(sample_winners, levels=[2, 3, 4, 5])
    
    print("\nFrecuencias de Pares (Top 3):")
    for pair, freq in all_frequencies[2].most_common(3):
        print(f"  {pair}: {freq}")

    print("\nFrecuencias de Tercias (Top 3):")
    for triple, freq in all_frequencies[3].most_common(3):
        print(f"  {triple}: {freq}")
        
    # Combinación a evaluar
    test_combo = [1, 2, 3, 11, 22, 33]
    print(f"\n--- 2. Calculando Afinidades para la combinación {test_combo} ---")
    
    combo_affinities = calculate_affinities_for_combo(test_combo, all_frequencies)
    
    print(f"  Afinidad de Pares (Nivel 2): {combo_affinities.get(2, 0)}")
    print(f"  Afinidad de Tercias (Nivel 3): {combo_affinities.get(3, 0)}")
    print(f"  Afinidad de Cuartetos (Nivel 4): {combo_affinities.get(4, 0)}")
    print(f"  Afinidad de Quintetos (Nivel 5): {combo_affinities.get(5, 0)}")