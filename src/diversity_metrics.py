# src/diversity_metrics.py

import numpy as np
import pandas as pd
from typing import List, Dict
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import entropy
import logging

class DiversityAnalyzer:
    """
    Analizador de diversidad genética para poblaciones del Master Forger.
    """
    def __init__(self):
        self.history = []

    def analyze_population(self, population: List[List[int]], generation: int, best_fitness: float) -> Dict:
        """Calcula un conjunto de métricas y las guarda en el historial."""
        if not population: return {}

        # 1. Diversidad de Hamming
        max_num = 39
        binary_matrix = np.zeros((len(population), max_num))
        for i, combo in enumerate(population):
            for num in combo:
                if 1 <= num <= max_num: binary_matrix[i, num-1] = 1
        hamming_diversity = np.mean(pdist(binary_matrix, metric='hamming')) if len(population) > 1 else 0.0

        # 2. Entropía de alelos (números)
        all_numbers = [num for combo in population for num in combo]
        if not all_numbers:
             number_entropy = 0
        else:
            _, counts = np.unique(all_numbers, return_counts=True)
            number_entropy = entropy(counts, base=2)

        # Guardar historial
        record = {
            'generation': generation,
            'best_fitness': best_fitness,
            'hamming_diversity': hamming_diversity,
            'number_entropy': number_entropy,
        }
        self.history.append(record)
        return record

    def get_diversity_report_df(self) -> pd.DataFrame:
        """Devuelve el historial como un DataFrame de Pandas."""
        return pd.DataFrame(self.history)

    def plot_evolution_analysis(self, save_path: str):
        """Crea y guarda una visualización del análisis de evolución."""
        if not self.history:
            logging.warning("No hay datos de historial para graficar.")
            return
            
        df = pd.DataFrame(self.history)
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle('Análisis de Evolución del Master Forger Ultimate', fontsize=16, fontweight='bold')
        
        # Gráfico de Fitness
        axes[0].plot(df['generation'], df['best_fitness'], 'b-', label='Mejor Fitness', linewidth=2)
        axes[0].set_xlabel('Generación')
        axes[0].set_ylabel('Fitness')
        axes[0].set_title('Evolución del Fitness')
        axes[0].grid(True, alpha=0.5)
        
        # Gráfico de Diversidad
        ax2 = axes[1]
        ax2.plot(df['generation'], df['hamming_diversity'], 'g-', label='Diversidad Hamming', linewidth=2)
        ax2.set_xlabel('Generación')
        ax2.set_ylabel('Diversidad Hamming', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        ax3 = ax2.twinx()
        ax3.plot(df['generation'], df['number_entropy'], 'r--', label='Entropía de Números', alpha=0.7)
        ax3.set_ylabel('Entropía (bits)', color='r')
        ax3.tick_params(axis='y', labelcolor='r')
        
        axes[1].set_title('Evolución de la Diversidad Genética')
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax2.transAxes)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        logging.info(f"Gráfico de evolución guardado en: {save_path}")