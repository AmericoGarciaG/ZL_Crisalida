#!/usr/bin/env python3
"""
Proyecto Crisálida - Fase 3: Transformer-Omega (Versión Corregida e Integrada)
Arquitectura híbrida avanzada que fusiona características de afinidad con contextuales.
Versión oficial del proyecto, integrada con config.py y utils.py.
"""

import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow import keras, metrics
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import os

# --- CAMBIO 1: Importar la configuración central y las utilidades ---
import config
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TransformerOmega:
    """
    Transformer-Omega: Arquitectura híbrida avanzada para discriminación de combinaciones ganadoras
    """
    
    # --- CAMBIO 2: Modificar el constructor para usar config.py por defecto ---
    def __init__(self, model_config: Dict[str, Any] = config.MODEL_CONFIG):
        self.config = model_config
        self.model = None
        self.scaler_affinity = None
        self.scaler_contextual = None
        self.history = None
        
        # Configuración de arquitectura leída desde el diccionario de config
        self.affinity_dim = self.config['affinity_dim']
        self.contextual_dim = self.config['contextual_dim']
        self.d_model = self.config['d_model']
        self.num_heads = self.config['num_heads']
        self.num_layers = self.config['num_layers']
        self.ff_dim = self.config['ff_dim']
        self.dropout_rate = self.config['dropout_rate']
        
    def affinity_encoder_block(self, inputs: tf.Tensor, name_prefix: str) -> tf.Tensor:
        """Encoder especializado para características de afinidad"""
        # Proyección inicial
        x = layers.Dense(self.d_model, activation='gelu', name=f'{name_prefix}_projection')(inputs)
        x = layers.LayerNormalization(name=f'{name_prefix}_norm1')(x)
        x = layers.Dropout(self.dropout_rate, name=f'{name_prefix}_dropout1')(x)
        
        # Capas de transformación profunda para afinidades
        for i in range(2):
            residual = x
            ff1 = layers.Dense(self.ff_dim, activation='gelu', name=f'{name_prefix}_ff1_{i}')(x)
            ff2 = layers.Dense(self.d_model, name=f'{name_prefix}_ff2_{i}')(ff1)
            gate = layers.Dense(self.d_model, activation='sigmoid', name=f'{name_prefix}_gate_{i}')(x)
            x = gate * ff2 + (1 - gate) * residual
            x = layers.LayerNormalization(name=f'{name_prefix}_norm_{i+2}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'{name_prefix}_dropout_{i+2}')(x)
        
        return x
    
    def contextual_encoder_block(self, inputs: tf.Tensor, name_prefix: str) -> tf.Tensor:
        """Encoder contextual para características Top 10"""
        # Reshape para secuencia: (batch, 60) -> (batch, 6, 10)
        batch_size = tf.shape(inputs)[0]
        x = tf.reshape(inputs, (batch_size, 6, 10))
        
        # Proyección a d_model
        x = layers.Dense(self.d_model, activation='gelu', name=f'{name_prefix}_projection')(x)
        
        # Codificación posicional simple
        pos_encoding = layers.Embedding(6, self.d_model, name=f'{name_prefix}_pos_embedding')(tf.range(6))
        x = x + pos_encoding
        
        x = layers.LayerNormalization(name=f'{name_prefix}_norm_input')(x)
        x = layers.Dropout(self.dropout_rate, name=f'{name_prefix}_dropout_input')(x)
        
        # Capas de atención multi-head
        for i in range(self.num_layers):
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                dropout=self.dropout_rate,
                name=f'{name_prefix}_attention_{i}'
            )(x, x)
            x = layers.Add(name=f'{name_prefix}_add1_{i}')([x, attention_output])
            x = layers.LayerNormalization(name=f'{name_prefix}_norm1_{i}')(x)
            
            ff_output = layers.Dense(self.ff_dim, activation='gelu', name=f'{name_prefix}_ff1_{i}')(x)
            ff_output = layers.Dense(self.d_model, name=f'{name_prefix}_ff2_{i}')(ff_output)
            ff_output = layers.Dropout(self.dropout_rate, name=f'{name_prefix}_ff_dropout_{i}')(ff_output)
            
            x = layers.Add(name=f'{name_prefix}_add2_{i}')([x, ff_output])
            x = layers.LayerNormalization(name=f'contextual_norm2_{i}')(x) # Corregido nombre de capa para ser único
        
        return x
    
    def adaptive_fusion_layer(self, affinity_features: tf.Tensor, contextual_features: tf.Tensor) -> tf.Tensor:
        """Capa de fusión adaptativa"""
        # Pooling adaptativo para características contextuales
        attention_weights = layers.Dense(1, activation='softmax', name='contextual_attention_pooling')(contextual_features)
        contextual_pooled = tf.reduce_sum(contextual_features * attention_weights, axis=1)
        
        affinity_proj = layers.Dense(self.d_model, activation='gelu', name='affinity_fusion_proj')(affinity_features)
        contextual_proj = layers.Dense(self.d_model, activation='gelu', name='contextual_fusion_proj')(contextual_pooled)
        
        concat_features = layers.Concatenate(name='fusion_concat')([affinity_proj, contextual_proj])
        
        fusion_gate = layers.Dense(self.d_model * 2, activation='sigmoid', name='fusion_gate')(concat_features)
        gated_features = fusion_gate * concat_features
        
        fused_features = layers.Dense(self.d_model, activation='gelu', name='fusion_final')(gated_features)
        
        return fused_features
    
    def build_model(self) -> keras.Model:
        """Construye la arquitectura completa del Transformer-Omega"""
        logging.info("Construyendo arquitectura Transformer-Omega...")
        
        affinity_input = layers.Input(shape=(self.affinity_dim,), name='affinity_input')
        contextual_input = layers.Input(shape=(self.contextual_dim,), name='contextual_input')
        
        affinity_encoded = self.affinity_encoder_block(affinity_input, 'affinity_encoder')
        contextual_encoded = self.contextual_encoder_block(contextual_input, 'contextual_encoder')
        
        fused_features = self.adaptive_fusion_layer(affinity_encoded, contextual_encoded)
        
        x = layers.LayerNormalization(name='final_norm')(fused_features)
        x = layers.Dropout(self.dropout_rate, name='final_dropout1')(x)
        x = layers.Dense(self.ff_dim, activation='gelu', name='final_dense1')(x)
        x = layers.Dropout(self.dropout_rate, name='final_dropout2')(x)
        x = layers.Dense(self.d_model // 2, activation='gelu', name='final_dense2')(x)
        x = layers.Dropout(self.dropout_rate, name='final_dropout3')(x)
        
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        model = keras.Model(inputs=[affinity_input, contextual_input], outputs=output, name='TransformerOmega')
        
        logging.info(f"Transformer-Omega construido exitosamente. Parámetros totales: {model.count_params():,}")
        return model
    
    # --- CAMBIO 3: Modificar prepare_data para usar rutas de config.py ---
    def prepare_data(self) -> Tuple[Any, Any, Any, Any]:
        """Prepara los datos para entrenamiento usando rutas de config"""
        logging.info("Preparando datos para Transformer-Omega...")
        
        features = np.load(config.HYBRID_FEATURES_PATH)
        labels = np.load(config.HYBRID_LABELS_PATH)
        with open(config.HYBRID_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        
        affinity_features = features[:, :self.affinity_dim]
        contextual_features = features[:, self.affinity_dim:]
        
        self.scaler_affinity = StandardScaler()
        self.scaler_contextual = StandardScaler()
        affinity_features = self.scaler_affinity.fit_transform(affinity_features)
        contextual_features = self.scaler_contextual.fit_transform(contextual_features)
        
        indices = np.arange(len(labels))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42, stratify=labels[train_idx])
        
        X_train = [affinity_features[train_idx], contextual_features[train_idx]]
        X_val = [affinity_features[val_idx], contextual_features[val_idx]]
        X_test = [affinity_features[test_idx], contextual_features[test_idx]]
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        y_test = labels[test_idx]
        
        logging.info(f"Entrenamiento: {len(y_train)}, Validación: {len(y_val)}, Prueba: {len(y_test)}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata
    
    # --- CAMBIO 4: Modificar compile_model para usar config.py ---
    def compile_model(self):
        """Compila el modelo con optimizador y métricas de config"""
        optimizer = keras.optimizers.AdamW(
            learning_rate=config.TRAINING_CONFIG['learning_rate'],
            weight_decay=config.TRAINING_CONFIG['weight_decay'],
            clipnorm=config.TRAINING_CONFIG['clipnorm']
        )
        
        model_metrics = [
            metrics.BinaryAccuracy(name='accuracy'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall')
        ]
        
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=model_metrics)
        logging.info(f"Modelo compilado con learning_rate={config.TRAINING_CONFIG['learning_rate']}")
    
    # --- CAMBIO 5: Modificar train_model para usar config.py ---
    def train_model(self, train_data: Tuple, val_data: Tuple) -> keras.callbacks.History:
        """Entrena el modelo Transformer-Omega usando config"""
        epochs = config.TRAINING_CONFIG['epochs']
        logging.info(f"Iniciando entrenamiento por {epochs} épocas...")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.TRAINING_CONFIG['early_stopping_patience'],
                restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config.TRAINING_CONFIG['reduce_lr_patience'],
                min_lr=1e-7, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                str(config.TRANSFORMER_OMEGA_BEST_PATH),
                monitor='val_loss',
                save_best_only=True, verbose=1
            )
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=config.TRAINING_CONFIG['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        logging.info("Entrenamiento completado")
        return self.history
    
    def evaluate_model(self, test_data: Tuple) -> Tuple[Dict[str, float], np.ndarray]:
        """Evalúa el modelo en el conjunto de prueba"""
        logging.info("Evaluando Transformer-Omega...")
        
        X_test, y_test = test_data
        y_pred_proba = self.model.predict(X_test)
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
        
        results = {
            'auc': auc_score,
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': f1_score,
            'loss': test_loss
        }
        
        logging.info(f"Resultados de evaluación:")
        for metric, value in results.items():
            logging.info(f"  {metric}: {value:.4f}")
        
        return results, y_pred_proba
    
    def save_model(self, path: Path):
        """Guarda el modelo entrenado"""
        self.model.save(path)
        logging.info(f"Modelo guardado en {path}")

# --- CAMBIO 6: Actualizar la función main para ser el orquestador limpio ---
def main():
    """Función principal de la Fase 3, ahora usando la clase integrada"""
    logging.info("=== PROYECTO CRISÁLIDA - FASE 3: TRANSFORMER-OMEGA (INTEGRADO) ===")
    
    # Asegurar que los directorios existan
    utils.ensure_dirs_exist()
    
    # Inicializar Transformer-Omega (usará la configuración de config.py por defecto)
    omega = TransformerOmega()
    
    # Preparar datos (las rutas se obtienen de config.py)
    train_data, val_data, test_data, metadata = omega.prepare_data()
    
    # Construir modelo
    omega.model = omega.build_model()
    omega.model.summary()
    
    # Compilar modelo (los hiperparámetros se obtienen de config.py)
    omega.compile_model()
    
    # Entrenar modelo (los hiperparámetros se obtienen de config.py)
    history = omega.train_model(train_data, val_data)
    
    # Evaluar modelo
    results, predictions = omega.evaluate_model(test_data)
    
    # Guardar modelo final
    omega.save_model(config.TRANSFORMER_OMEGA_FINAL_PATH)
    
    # Guardar resultados y predicciones
    results_data = {
        'config': config.MODEL_CONFIG, # Usar el diccionario, no el módulo
        'training_config': config.TRAINING_CONFIG,
        'results': utils.convert_numpy_types(results), # Usar la utilidad para seguridad
        'training_history': {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy']
        }
    }
    
    with open(config.OMEGA_RESULTS_PATH, 'w') as f:
        json.dump(results_data, f, indent=2)
    logging.info(f"Resultados guardados en {config.OMEGA_RESULTS_PATH}")
    
    np.save(config.OMEGA_PREDICTIONS_PATH, predictions)
    logging.info(f"Predicciones guardadas en {config.OMEGA_PREDICTIONS_PATH}")
    
    logging.info("=== FASE 3 COMPLETADA EXITOSAMENTE ===")
    logging.info(f"AUC Score Final: {results['auc']:.6f}")
    logging.info("¡TRANSFORMER-OMEGA LISTO!")

if __name__ == "__main__":
    main()