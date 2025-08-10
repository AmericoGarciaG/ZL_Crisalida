# high_performance_search.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras # # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import logging
import os
import argparse
import time
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

# --- Configuración Inicial ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# --- Módulos Reutilizables ---

def load_and_preprocess_data(noise_multiplier: int = 1) -> Optional[Tuple[np.ndarray, ...]]:
    # ... (Sin cambios en esta función) ...
    logging.info(f"Cargando datos con un multiplicador de ruido de {noise_multiplier}x...")
    try:
        winners_df = pd.read_csv(os.path.join('data', 'winners_dataset.csv'), header=None)
        noise_df_base = pd.read_csv(os.path.join('data', 'noise_dataset.csv'), header=None)
    except FileNotFoundError as e:
        logging.error(f"Error: {e}. Asegúrate de que los archivos CSV están en la carpeta 'data'.")
        return None
    noise_df = pd.concat([noise_df_base] * noise_multiplier, ignore_index=True) if noise_multiplier > 1 else noise_df_base
    winners_sorted = np.sort(winners_df.values, axis=1)
    noise_sorted = np.sort(noise_df.values, axis=1)
    winners_labels = np.ones(len(winners_sorted))
    noise_labels = np.zeros(len(noise_sorted))
    all_data = np.vstack([winners_sorted, noise_sorted])
    all_labels = np.concatenate([winners_labels, noise_labels])
    normalized_data = (all_data - 1) / 38.0
    sequence_data = normalized_data.reshape(-1, 6, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        sequence_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    logging.info(f"Datos listos. Forma de X_train: {X_train.shape}, Forma de X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def transformer_encoder(inputs: tf.Tensor, head_size: int, num_heads: int, ff_dim: int, dropout: float = 0.1) -> tf.Tensor:
    # --- CÓDIGO CORREGIDO ---
    x = keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(inputs + x)
    ff_out = keras.layers.Dense(ff_dim, activation="relu")(x)
    ff_out = keras.layers.Dropout(dropout)(ff_out)
    ff_out = keras.layers.Dense(inputs.shape[-1])(ff_out)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x + ff_out)
    return x

def build_transformer_autoencoder_v2(input_shape: Tuple[int, int], num_heads: int, d_model: int, ff_dim: int, num_blocks: int, embedding_dim: int, dropout_rate: float) -> Tuple[keras.Model, keras.Model]:
    # --- CÓDIGO CORREGIDO ---
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Dense(d_model)(inputs)
    for _ in range(num_blocks):
        x = transformer_encoder(x, d_model // num_heads, num_heads, ff_dim, dropout_rate)
    sequence_representation = keras.layers.GlobalAveragePooling1D()(x)
    x_dense = keras.layers.Dense(d_model, activation='relu')(sequence_representation)
    x_dense = keras.layers.Dropout(dropout_rate)(x_dense)
    decoder_output = keras.layers.Dense(1, name='decoder_output')(x_dense)
    encoder_output = keras.layers.Dense(embedding_dim, name='encoder_output')(sequence_representation)
    autoencoder = keras.Model(inputs, decoder_output, name="AutoencoderV2")
    encoder = keras.Model(inputs, encoder_output, name="EncoderV2")
    return autoencoder, encoder

# --- ORQUESTADOR DE ALTO RENDIMIENTO (sin cambios en la lógica) ---

def main(args: argparse.Namespace):
    strategy = tf.distribute.MirroredStrategy()
    logging.info(f"Estrategia de distribución iniciada. Número de réplicas (núcleos a usar): {strategy.num_replicas_in_sync}")

    GLOBAL_BATCH_SIZE = args.batch_size * strategy.num_replicas_in_sync
    logging.info(f"Tamaño de batch por réplica: {args.batch_size}. Tamaño de batch global: {GLOBAL_BATCH_SIZE}")

    data_tuple = load_and_preprocess_data(noise_multiplier=args.sample_multiplier)
    if data_tuple is None: return
    X_train, X_test, y_train, y_test = data_tuple
    y_train_autoencoder = np.mean(X_train, axis=1)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_autoencoder)).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    search_space: Dict[str, List[Any]] = {
        'lr': [1e-3, 5e-4],
        'd_model': [32, 64],
        'num_heads': [4, 8],
        'num_blocks': [2, 4],
        'embedding_dim': [8, 16],
        'dropout': [0.1, 0.2]
    }
    
    keys, values = zip(*search_space.items())
    configs = [dict(zip(keys, v)) for v in product(*values)]
    
    logging.info(f"Se probarán {len(configs)} configuraciones de hiperparámetros en serie (cada una entrenada en paralelo).")
    
    results = []
    total_start_time = time.time()
    
    for i, config in enumerate(configs):
        logging.info("\n" + f"--- INICIANDO EXPERIMENTO {i+1}/{len(configs)} ---")
        logging.info(f"Configuración: {config}")
        exp_start_time = time.time()
        
        with strategy.scope():
            autoencoder, encoder = build_transformer_autoencoder_v2(
                input_shape=(6, 1), num_heads=config['num_heads'], d_model=config['d_model'],
                ff_dim=config['d_model'] * 2, num_blocks=config['num_blocks'],
                embedding_dim=config['embedding_dim'], dropout_rate=config['dropout']
            )
            
            optimizer = keras.optimizers.Adam(learning_rate=config['lr'])
            autoencoder.compile(optimizer=optimizer, loss="mse")
        
        autoencoder.fit(
            train_dataset, 
            epochs=args.epochs,
            verbose=1
        )
        
        embeddings_test = encoder.predict(X_test)
        scores = np.linalg.norm(embeddings_test, axis=1)
        auc_score = roc_auc_score(y_test, scores)
        
        duration = time.time() - exp_start_time
        logging.info(f"Experimento {i+1} finalizado en {duration:.2f}s. AUC: {auc_score:.4f}")
        
        result = config.copy()
        result['auc_score'] = auc_score
        result['duration_sec'] = duration
        results.append(result)

    total_duration = time.time() - total_start_time
    logging.info(f"Búsqueda en grid completada en {total_duration/60:.2f} minutos.")
    
    results_df = pd.DataFrame(results)
    results_df.sort_values(by='auc_score', ascending=False, inplace=True)
    
    logging.info("\n\n" + "="*80)
    logging.info("--- RESULTADOS DE LA BÚSQUEDA DE HIPERPARÁMETROS DE ALTO RENDIMIENTO ---")
    logging.info("="*80)
    
    print(results_df.to_string())
    
    if not results_df.empty:
        best_config = results_df.iloc[0].to_dict()
        logging.info("\n\n--- MEJOR CONFIGURACIÓN ENCONTRADA ---")
        print(best_config)
        logging.info(f"\nMEJOR AUC SCORE ALCANZADO: {best_config['auc_score']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Búsqueda de hiperparámetros de alto rendimiento para el Transformer.")
    parser.add_argument('--sample_multiplier', type=int, default=1, help='Multiplicador para el tamaño del dataset de ruido.')
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas de entrenamiento para cada experimento.')
    parser.add_argument('--batch_size', type=int, default=64, help='Tamaño del batch POR RÉPLICA (por núcleo de CPU).')
    
    args = parser.parse_args()
    main(args)