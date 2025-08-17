# src/test_runner.py

import sys
import os
sys.path.append('.')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
from master_forger_ultimate import MasterForgerUltimate, initialize_worker
from predict_combination import FeatureExtractor, Predictor
import config
import utils
from concurrent.futures import ProcessPoolExecutor

def quick_test():
    print("🧪 INICIANDO PRUEBA RÁPIDA DEL MASTER FORGER ULTIMATE")
    
    test_config = {
        'population_size': 50, 'generations': 20, 'num_restarts': 2,
        'elite_size': 5, 'mutation_rate': 0.2, 'tournament_size': 3,
        'early_stopping_threshold': 0.95, 'num_workers': 2,
        'stagnation_limit': 1, 'high_mutation_rate': 0.5, 'diversity_threshold': 0.15
    }
    
    try:
        print("📦 Inicializando componentes...")
        extractor = FeatureExtractor()
        predictor = Predictor()
        
        forger = MasterForgerUltimate(test_config)
        
        print(f"\n🚀 Ejecutando prueba...")
        best_combo, best_score, elite_pop = None, -1.0, None

        with ProcessPoolExecutor(max_workers=test_config['num_workers'], initializer=initialize_worker, initargs=(extractor, predictor)) as executor:
            for cycle in range(test_config['num_restarts']):
                print(f"\n🔄 Ciclo {cycle + 1}/{test_config['num_restarts']}")
                
                cycle_best, cycle_score, final_pop = forger.run_evolution_cycle(executor, elite_pop, cycle + 1)
                
                if cycle_score > best_score:
                    best_score = cycle_score
                    best_combo = cycle_best
                
                final_fitness = forger.evaluate_population_parallel(final_pop, executor)
                sorted_indices = sorted(range(len(final_fitness)), key=lambda i: final_fitness[i], reverse=True)
                elite_pop = [final_pop[i] for i in sorted_indices[:test_config['elite_size']]]
        
        print(f"\n🎯 RESULTADOS DE LA PRUEBA:\n   Mejor combinación: {best_combo}\n   Mejor score: {best_score:.6f}")
        print(f"\n✅ PRUEBA COMPLETADA EXITOSAMENTE")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR EN LA PRUEBA: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    utils.ensure_dirs_exist()
    success = quick_test()
    if not success: sys.exit(1)