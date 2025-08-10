#!/usr/bin/env python3
"""
Proyecto Crisálida - Fase 3: Transformer-Omega (Versión Simplificada)
Arquitectura híbrida funcional que fusiona características de afinidad con contextuales
"""

import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow import keras, metrics
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import os

import config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TransformerOmegaSimple:
    """
    Transformer-Omega Simplificado: Arquitectura híbrida funcional
    """
    
    def __init__(self, model_config: Dict[str, Any] = config.MODEL_CONFIG):
        self.config = model_config
        self.model = None
        self.scaler_affinity = None
        self.scaler_contextual = None   
        self.history = None
        
        # Configuración de arquitectura
        self.affinity_dim = self.config['affinity_dim']
        self.contextual_dim = self.config['contextual_dim']
        self.d_model = self.config['d_model']
        self.num_heads = self.config['num_heads']
        self.num_layers = self.config['num_layers']
        self.ff_dim = self.config['ff_dim']
        self.dropout_rate = self.config['dropout_rate']
        
    def build_model(self) -> keras.Model:
        """Construye la arquitectura simplificada del Transformer-Omega"""
        logging.info("Construyendo arquitectura Transformer-Omega simplificada...")
        
        # Entradas
        affinity_input = layers.Input(shape=(self.affinity_dim,), name='affinity_input')
        contextual_input = layers.Input(shape=(self.contextual_dim,), name='contextual_input')
        
        # === ENCODER DE AFINIDAD ===
        # Procesamiento especializado para características de afinidad (globales)
        affinity_x = layers.Dense(self.d_model, activation='gelu', name='affinity_proj1')(affinity_input)
        affinity_x = layers.LayerNormalization(name='affinity_norm1')(affinity_x)
        affinity_x = layers.Dropout(self.dropout_rate, name='affinity_dropout1')(affinity_x)
        
        # Capas profundas para afinidades
        for i in range(2):
            residual = affinity_x
            affinity_x = layers.Dense(self.ff_dim, activation='gelu', name=f'affinity_ff1_{i}')(affinity_x)
            affinity_x = layers.Dense(self.d_model, name=f'affinity_ff2_{i}')(affinity_x)
            
            # Gating mechanism
            gate = layers.Dense(self.d_model, activation='sigmoid', name=f'affinity_gate_{i}')(residual)
            affinity_x = gate * affinity_x + (1 - gate) * residual
            
            affinity_x = layers.LayerNormalization(name=f'affinity_norm_{i+2}')(affinity_x)
            affinity_x = layers.Dropout(self.dropout_rate, name=f'affinity_dropout_{i+2}')(affinity_x)
        
        # === ENCODER CONTEXTUAL ===
        # Reshape contextual: (60,) -> (6, 10) para procesar como secuencia
        contextual_x = layers.Reshape((6, 10), name='contextual_reshape')(contextual_input)
        
        # Proyección a d_model
        contextual_x = layers.Dense(self.d_model, activation='gelu', name='contextual_proj')(contextual_x)
        
        # Codificación posicional
        pos_embedding = layers.Embedding(6, self.d_model, name='pos_embedding')
        positions = tf.range(6)
        contextual_x = contextual_x + pos_embedding(positions)
        
        contextual_x = layers.LayerNormalization(name='contextual_norm_input')(contextual_x)
        contextual_x = layers.Dropout(self.dropout_rate, name='contextual_dropout_input')(contextual_x)
        
        # Capas de atención
        for i in range(self.num_layers):
            # Multi-head self-attention
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                dropout=self.dropout_rate,
                name=f'contextual_attention_{i}'
            )(contextual_x, contextual_x)
            
            # Residual + norm
            contextual_x = layers.Add(name=f'contextual_add1_{i}')([contextual_x, attention_output])
            contextual_x = layers.LayerNormalization(name=f'contextual_norm1_{i}')(contextual_x)
            
            # Feed-forward
            ff_output = layers.Dense(self.ff_dim, activation='gelu', name=f'contextual_ff1_{i}')(contextual_x)
            ff_output = layers.Dense(self.d_model, name=f'contextual_ff2_{i}')(ff_output)
            ff_output = layers.Dropout(self.dropout_rate, name=f'contextual_ff_dropout_{i}')(ff_output)
            
            # Residual + norm
            contextual_x = layers.Add(name=f'contextual_add2_{i}')([contextual_x, ff_output])
            contextual_x = layers.LayerNormalization(name=f'contextual_norm2_{i}')(contextual_x)
        
        # Pooling adaptativo para secuencia contextual
        attention_weights = layers.Dense(1, activation='softmax', name='contextual_pool_attention')(contextual_x)
        contextual_pooled = layers.Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1), 
                                        name='contextual_pooled')([contextual_x, attention_weights])
        
        # === FUSIÓN ADAPTATIVA ===
        # Proyectar ambas características al mismo espacio
        affinity_proj = layers.Dense(self.d_model, activation='gelu', name='fusion_affinity_proj')(affinity_x)
        contextual_proj = layers.Dense(self.d_model, activation='gelu', name='fusion_contextual_proj')(contextual_pooled)
        
        # Concatenar características
        fused = layers.Concatenate(name='fusion_concat')([affinity_proj, contextual_proj])
        
        # Gating para fusión adaptativa
        fusion_gate = layers.Dense(self.d_model * 2, activation='sigmoid', name='fusion_gate')(fused)
        gated_fused = layers.Multiply(name='fusion_gated')([fusion_gate, fused])
        
        # Proyección final de fusión
        fused_final = layers.Dense(self.d_model, activation='gelu', name='fusion_final')(gated_fused)
        
        # === CLASIFICADOR FINAL ===
        x = layers.LayerNormalization(name='final_norm')(fused_final)
        x = layers.Dropout(self.dropout_rate, name='final_dropout1')(x)
        
        # Capas densas finales
        x = layers.Dense(self.ff_dim, activation='gelu', name='final_dense1')(x)
        x = layers.Dropout(self.dropout_rate, name='final_dropout2')(x)
        x = layers.Dense(self.d_model // 2, activation='gelu', name='final_dense2')(x)
        x = layers.Dropout(self.dropout_rate, name='final_dropout3')(x)
        
        # Salida
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Crear modelo
        model = keras.Model(
            inputs=[affinity_input, contextual_input],
            outputs=output,
            name='TransformerOmegaSimple'
        )
        
        logging.info(f"Transformer-Omega construido exitosamente")
        logging.info(f"Parámetros totales: {model.count_params():,}")
        
        return model
    
    def prepare_data(self) -> Tuple[Any, Any, Any, Any]:
        """Prepara los datos para entrenamiento"""
        logging.info("Preparando datos para Transformer-Omega...")
        
        # Cargar datos
        features = np.load(config.HYBRID_FEATURES_PATH)
        labels = np.load(config.HYBRID_LABELS_PATH)
        
        with open(config.HYBRID_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        
        # Separar características de afinidad y contextuales
        affinity_features = features[:, :self.affinity_dim]
        contextual_features = features[:, self.affinity_dim:]
        
        logging.info(f"Características de afinidad: {affinity_features.shape}")
        logging.info(f"Características contextuales: {contextual_features.shape}")
        
        # Normalización
        self.scaler_affinity = StandardScaler()
        self.scaler_contextual = StandardScaler()
        
        affinity_features = self.scaler_affinity.fit_transform(affinity_features)
        contextual_features = self.scaler_contextual.fit_transform(contextual_features)
        
        # División train/validation/test
        indices = np.arange(len(labels))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42, stratify=labels[train_idx])
        
        # Crear conjuntos de datos
        X_train = [affinity_features[train_idx], contextual_features[train_idx]]
        X_val = [affinity_features[val_idx], contextual_features[val_idx]]
        X_test = [affinity_features[test_idx], contextual_features[test_idx]]
        
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        y_test = labels[test_idx]
        
        logging.info(f"Entrenamiento: {len(y_train)} muestras")
        logging.info(f"Validación: {len(y_val)} muestras")
        logging.info(f"Prueba: {len(y_test)} muestras")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata
    
    def compile_model(self, learning_rate: float = config.TRAINING_CONFIG['learning_rate']):
        """Compila el modelo"""
        optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=config.TRAINING_CONFIG['weight_decay'],
        clipnorm=config.TRAINING_CONFIG['clipnorm']
    )
        
        model_metrics = [
        metrics.BinaryAccuracy(name='accuracy'),
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall')
    ]
                
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=model_metrics
        )
        
        logging.info(f"Modelo compilado con learning_rate={learning_rate}")
    
    def train_model(self, train_data: Tuple, val_data: Tuple, epochs: int = config.TRAINING_CONFIG['epochs']) -> keras.callbacks.History:
        """Entrena el modelo"""
        logging.info(f"Iniciando entrenamiento por {epochs} épocas...")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Crear directorios
        os.makedirs('../models', exist_ok=True)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.TRAINING_CONFIG['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config.TRAINING_CONFIG['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                str(config.TRANSFORMER_OMEGA_BEST_PATH),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entrenamiento
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
    
    def evaluate_model(self, test_data: Tuple) -> Dict[str, float]:
        """Evalúa el modelo"""
        logging.info("Evaluando Transformer-Omega...")
        
        X_test, y_test = test_data
        
        # Predicciones
        y_pred_proba = self.model.predict(X_test)
        
        # Métricas
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

def main():
    # utils.ensure_dirs_exist() # Puedes mantener o quitar esta línea si ya no usas utils
    """Función principal de la Fase 3"""
    logging.info("=== PROYECTO CRISÁLIDA - FASE 3: TRANSFORMER-OMEGA ===")
    
    # --- CAMBIO 1: Definir la configuración como un diccionario local ---
    # En lugar de usar el módulo 'config', definimos los parámetros aquí.
    # Esto evita el error de serialización JSON.
    model_config = {
        'affinity_dim': 14,
        'contextual_dim': 60,
        'd_model': 128,
        'num_heads': 8,
        'num_layers': 3,
        'ff_dim': 256,
        'dropout_rate': 0.1
    }
    
    training_config = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'clipnorm': 1.0,
        'epochs': 50,
        'early_stopping_patience': 15,
        'reduce_lr_patience': 8,
        'batch_size': 32
    }

    # --- CAMBIO 2: Pasar el diccionario a la clase ---
    # Inicializar Transformer-Omega con el diccionario de configuración
    omega = TransformerOmegaSimple(model_config=model_config)
    
    # Preparar datos
    train_data, val_data, test_data, metadata = omega.prepare_data()
    
    # Construir modelo
    omega.model = omega.build_model()
    
    # Mostrar arquitectura
    omega.model.summary()
    
    # Compilar modelo
    omega.compile_model(learning_rate=training_config['learning_rate'])
    
    # Entrenar modelo
    history = omega.train_model(train_data, val_data, epochs=training_config['epochs'])
    
    # Evaluar modelo
    results, predictions = omega.evaluate_model(test_data)
    
    # --- CAMBIO 3: Definir las rutas de guardado directamente ---
    # Ya que no usamos el módulo 'config', especificamos las rutas aquí.
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    final_model_path = '../models/transformer_omega_simple_final.keras'
    results_path = '../results/transformer_omega_simple_results.json'
    predictions_path = '../results/transformer_omega_simple_predictions.npy'

    # Guardar modelo
    omega.model.save(final_model_path)
    logging.info(f"Modelo final guardado en {final_model_path}")
    
    # Guardar resultados
    results_data = {
        # --- ¡LA CORRECCIÓN CLAVE! ---
        # Ahora 'model_config' es un diccionario, no un módulo.
        'config': model_config, 
        'training_config': training_config,
        'results': results,
        'training_history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    logging.info(f"Resultados guardados en {results_path}")
    
    # Guardar predicciones
    np.save(predictions_path, predictions)
    logging.info(f"Predicciones guardadas en {predictions_path}")
    
    logging.info("=== FASE 3 COMPLETADA EXITOSAMENTE ===")
    logging.info(f"AUC Score: {results['auc']:.4f}")
    logging.info("¡TRANSFORMER-OMEGA LISTO!")

if __name__ == "__main__":
    main()

