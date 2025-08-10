#!/usr/bin/env python3
"""
Proyecto Crisálida - Fase 3: Transformer-Omega (Versión Corregida)
Arquitectura híbrida avanzada que fusiona características de afinidad con contextuales
"""

import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TransformerOmega:
    """
    Transformer-Omega: Arquitectura híbrida avanzada para discriminación de combinaciones ganadoras
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler_affinity = None
        self.scaler_contextual = None
        self.history = None
        
        # Configuración de arquitectura
        self.affinity_dim = config.get('affinity_dim', 14)
        self.contextual_dim = config.get('contextual_dim', 60)
        self.d_model = config.get('d_model', 128)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 4)
        self.ff_dim = config.get('ff_dim', 256)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        
    def affinity_encoder_block(self, inputs: tf.Tensor, name_prefix: str) -> tf.Tensor:
        """Encoder especializado para características de afinidad"""
        # Proyección inicial
        x = layers.Dense(self.d_model, activation='gelu', name=f'{name_prefix}_projection')(inputs)
        x = layers.LayerNormalization(name=f'{name_prefix}_norm1')(x)
        x = layers.Dropout(self.dropout_rate, name=f'{name_prefix}_dropout1')(x)
        
        # Capas de transformación profunda para afinidades
        for i in range(2):
            # Residual connection
            residual = x
            
            # Feed-forward con gating
            ff1 = layers.Dense(self.ff_dim, activation='gelu', name=f'{name_prefix}_ff1_{i}')(x)
            ff2 = layers.Dense(self.d_model, name=f'{name_prefix}_ff2_{i}')(ff1)
            
            # Gating mechanism para afinidades
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
            # Multi-head self-attention
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                dropout=self.dropout_rate,
                name=f'{name_prefix}_attention_{i}'
            )(x, x)
            
            # Residual connection y normalización
            x = layers.Add(name=f'{name_prefix}_add1_{i}')([x, attention_output])
            x = layers.LayerNormalization(name=f'{name_prefix}_norm1_{i}')(x)
            
            # Feed-forward
            ff_output = layers.Dense(self.ff_dim, activation='gelu', name=f'{name_prefix}_ff1_{i}')(x)
            ff_output = layers.Dense(self.d_model, name=f'{name_prefix}_ff2_{i}')(ff_output)
            ff_output = layers.Dropout(self.dropout_rate, name=f'{name_prefix}_ff_dropout_{i}')(ff_output)
            
            # Residual connection y normalización
            x = layers.Add(name=f'{name_prefix}_add2_{i}')([x, ff_output])
            x = layers.LayerNormalization(name=f'{name_prefix}_norm2_{i}')(x)
        
        return x
    
    def adaptive_fusion_layer(self, affinity_features: tf.Tensor, contextual_features: tf.Tensor) -> tf.Tensor:
        """Capa de fusión adaptativa"""
        # Pooling adaptativo para características contextuales
        attention_weights = layers.Dense(1, activation='softmax', name='contextual_attention')(contextual_features)
        contextual_pooled = tf.reduce_sum(contextual_features * attention_weights, axis=1)
        
        # Proyectar ambas características al mismo espacio
        affinity_proj = layers.Dense(self.d_model, activation='gelu', name='affinity_fusion_proj')(affinity_features)
        contextual_proj = layers.Dense(self.d_model, activation='gelu', name='contextual_fusion_proj')(contextual_pooled)
        
        # Concatenar y procesar
        concat_features = layers.Concatenate(name='fusion_concat')([affinity_proj, contextual_proj])
        
        # Gating mechanism para fusión adaptativa
        fusion_gate = layers.Dense(self.d_model * 2, activation='sigmoid', name='fusion_gate')(concat_features)
        gated_features = fusion_gate * concat_features
        
        # Proyección final
        fused_features = layers.Dense(self.d_model, activation='gelu', name='fusion_final')(gated_features)
        
        return fused_features
    
    def build_model(self) -> keras.Model:
        """Construye la arquitectura completa del Transformer-Omega"""
        logging.info("Construyendo arquitectura Transformer-Omega...")
        
        # Entradas
        affinity_input = layers.Input(shape=(self.affinity_dim,), name='affinity_input')
        contextual_input = layers.Input(shape=(self.contextual_dim,), name='contextual_input')
        
        # Encoders especializados
        affinity_encoded = self.affinity_encoder_block(affinity_input, 'affinity_encoder')
        contextual_encoded = self.contextual_encoder_block(contextual_input, 'contextual_encoder')
        
        # Fusión adaptativa
        fused_features = self.adaptive_fusion_layer(affinity_encoded, contextual_encoded)
        
        # Capas de clasificación final
        x = layers.LayerNormalization(name='final_norm')(fused_features)
        x = layers.Dropout(self.dropout_rate, name='final_dropout1')(x)
        
        # Capas densas finales con regularización
        x = layers.Dense(self.ff_dim, activation='gelu', name='final_dense1')(x)
        x = layers.Dropout(self.dropout_rate, name='final_dropout2')(x)
        x = layers.Dense(self.d_model // 2, activation='gelu', name='final_dense2')(x)
        x = layers.Dropout(self.dropout_rate, name='final_dropout3')(x)
        
        # Salida final
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Crear modelo
        model = keras.Model(
            inputs=[affinity_input, contextual_input],
            outputs=output,
            name='TransformerOmega'
        )
        
        logging.info(f"Transformer-Omega construido exitosamente")
        logging.info(f"Parámetros totales: {model.count_params():,}")
        
        return model
    
    def prepare_data(self, features_path: str, labels_path: str, metadata_path: str) -> Tuple[Any, Any, Any, Any]:
        """Prepara los datos para entrenamiento"""
        logging.info("Preparando datos para Transformer-Omega...")
        
        # Cargar datos
        features = np.load(features_path)
        labels = np.load(labels_path)
        
        with open(metadata_path, 'r') as f:
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
    
    def compile_model(self, learning_rate: float = 1e-4):
        """Compila el modelo con optimizador y métricas"""
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-5,
            clipnorm=1.0
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logging.info(f"Modelo compilado con learning_rate={learning_rate}")
    
    def train_model(self, train_data: Tuple, val_data: Tuple, epochs: int = 100) -> keras.callbacks.History:
        """Entrena el modelo Transformer-Omega"""
        logging.info(f"Iniciando entrenamiento por {epochs} épocas...")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Crear directorio de modelos
        import os
        os.makedirs('../models', exist_ok=True)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                '../models/transformer_omega_best.keras',
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
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        logging.info("Entrenamiento completado")
        return self.history
    
    def evaluate_model(self, test_data: Tuple) -> Dict[str, float]:
        """Evalúa el modelo en el conjunto de prueba"""
        logging.info("Evaluando Transformer-Omega...")
        
        X_test, y_test = test_data
        
        # Predicciones
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
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
    
    def save_model(self, path: str):
        """Guarda el modelo entrenado"""
        self.model.save(path)
        logging.info(f"Modelo guardado en {path}")

def main():
    """Función principal de la Fase 3"""
    logging.info("=== PROYECTO CRISÁLIDA - FASE 3: TRANSFORMER-OMEGA ===")
    
    # Configuración del modelo
    config = {
        'affinity_dim': 14,
        'contextual_dim': 60,
        'd_model': 128,
        'num_heads': 8,
        'num_layers': 3,  # Reducido para mayor estabilidad
        'ff_dim': 256,
        'dropout_rate': 0.1
    }
    
    # Inicializar Transformer-Omega
    omega = TransformerOmega(config)
    
    # Preparar datos
    train_data, val_data, test_data, metadata = omega.prepare_data(
        '../data/hybrid_features.npy',
        '../data/hybrid_labels.npy',
        '../data/hybrid_metadata.json'
    )
    
    # Construir modelo
    omega.model = omega.build_model()
    
    # Mostrar arquitectura
    omega.model.summary()
    
    # Compilar modelo
    omega.compile_model(learning_rate=1e-4)
    
    # Entrenar modelo
    history = omega.train_model(train_data, val_data, epochs=50)  # Reducido para prueba
    
    # Evaluar modelo
    results, predictions = omega.evaluate_model(test_data)
    
    # Guardar modelo
    omega.save_model('../models/transformer_omega_final.keras')
    
    # Crear directorio de resultados
    import os
    os.makedirs('../results', exist_ok=True)
    
    # Guardar resultados
    results_data = {
        'config': config,
        'results': results,
        'training_history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
    }
    
    with open('../results/transformer_omega_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Guardar predicciones
    np.save('../results/transformer_omega_predictions.npy', predictions)
    
    logging.info("=== FASE 3 COMPLETADA EXITOSAMENTE ===")
    logging.info(f"AUC Score: {results['auc']:.4f}")
    logging.info("¡TRANSFORMER-OMEGA LISTO!")

if __name__ == "__main__":
    main()

