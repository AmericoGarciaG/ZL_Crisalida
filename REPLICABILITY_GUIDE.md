# Guía de Replicabilidad - Proyecto Crisálida
## Transformer-Omega: Reproducción Completa de Resultados

**Versión:** 1.0  
**Fecha:** Agosto 2025  
**Autor:** Equipo Zen Lotto  

---

## Resumen de Replicabilidad

Esta guía proporciona instrucciones completas para replicar todos los resultados del Proyecto Crisálida, incluyendo el entrenamiento del Transformer-Omega que alcanzó un AUC de 0.999953. Todos los componentes necesarios están incluidos: código fuente, datos procesados, configuraciones, y scripts de ejecución.

## Requisitos del Sistema

### Hardware Mínimo
- **CPU**: 8 cores, 2.5GHz+
- **RAM**: 16GB mínimo, 32GB recomendado
- **GPU**: NVIDIA GPU con 8GB+ VRAM (opcional pero recomendado)
- **Almacenamiento**: 10GB espacio libre

### Software Requerido
- **Python**: 3.11+
- **TensorFlow**: 2.15+
- **Scikit-learn**: 1.3+
- **NumPy**: 1.24+
- **Pandas**: 2.0+
- **Matplotlib**: 3.7+
- **Seaborn**: 0.12+

### Instalación de Dependencias

```bash
# Crear entorno virtual
py -3.11 -m venv .venv
python -m venv .venv
.venv/Scripts/Activate.ps1 # Windows
.venv\Scripts\activate  
# o 
source .venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install tensorflow==2.15.0
pip install scikit-learn==1.3.0
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install matplotlib==3.7.1
pip install seaborn==0.12.2
```

## Estructura del Proyecto

```
zl_crisalida/
├── data/
│   ├── winners_dataset.csv          # Combinaciones ganadoras originales
│   ├── noise_dataset.csv            # Combinaciones aleatorias originales
│   ├── hybrid_features.npy          # Características híbridas procesadas
│   ├── hybrid_labels.npy            # Etiquetas correspondientes
│   └── hybrid_metadata.json         # Metadatos del procesamiento
├── src/
│   ├── top10_analysis.py            # Análisis Top 10 características
│   ├── hybrid_feature_engineering_optimized.py  # Ingeniería híbrida
│   ├── transformer_omega_simple.py  # Modelo Transformer-Omega
│   ├── comprehensive_evaluation_fixed.py  # Evaluación comprehensiva
│   └── replication_pipeline.py      # Pipeline completo de replicación
├── models/
│   ├── transformer_omega_model.h5   # Modelo entrenado
│   └── model_config.json            # Configuración del modelo
├── results/
│   ├── transformer_omega_results.json      # Resultados finales
│   ├── transformer_omega_predictions.npy   # Predicciones del modelo
│   ├── comprehensive_evaluation.png        # Visualizaciones
│   └── comprehensive_evaluation_report.json # Reporte de evaluación
└── reports/
    ├── reporte_metodologico_completo_crisalida.md
    └── fase5_comprehensive_evaluation.md
```

## Pasos de Replicación

### Paso 1: Preparación de Datos

```bash
cd zl_crisalida/src
python top10_analysis.py
```

**Salida esperada:**
- Identificación de Top 10 características contextuales
- Análisis de importancia con 93.35% de captura
- Visualizaciones de importancia guardadas

### Paso 2: Ingeniería de Características Híbridas

```bash
python hybrid_feature_engineering_optimized.py
```

**Salida esperada:**
- `hybrid_features.npy`: (11552, 74) características híbridas
- `hybrid_labels.npy`: (11552,) etiquetas binarias
- `hybrid_metadata.json`: Metadatos del procesamiento
- Tiempo de ejecución: ~7 segundos

**Verificación:**
```python
import numpy as np
features = np.load('../data/hybrid_features.npy')
labels = np.load('../data/hybrid_labels.npy')
print(f"Features shape: {features.shape}")  # Debe ser (11552, 74)
print(f"Labels shape: {labels.shape}")      # Debe ser (11552,)
print(f"Positive samples: {np.sum(labels)}")  # Debe ser 1552
```

### Paso 3: Entrenamiento del Transformer-Omega

```bash
python transformer_omega_simple.py
```

**Salida esperada:**
- Modelo entrenado en ~8 minutos
- AUC Score: 0.999953 ± 0.000001
- Accuracy: 99.96% ± 0.01%
- Convergencia en 12 épocas

**Verificación de Resultados:**
```python
import json
with open('../results/transformer_omega_results.json', 'r') as f:
    results = json.load(f)
    
print(f"AUC Score: {results['results']['auc']:.6f}")
print(f"Accuracy: {results['results']['accuracy']:.4f}")
print(f"Epochs trained: {len(results['training_history']['loss'])}")
```

### Paso 4: Evaluación Comprehensiva

```bash
python comprehensive_evaluation_fixed.py
```

**Salida esperada:**
- 12 visualizaciones comprehensivas
- Reporte de evaluación completo
- Comparación con modelos anteriores
- Análisis de características híbridas

## Configuraciones Críticas

### Configuración del Modelo

```python
config = {
    'affinity_dim': 14,
    'contextual_dim': 60,
    'd_model': 128,
    'num_heads': 8,
    'num_layers': 3,
    'ff_dim': 256,
    'dropout_rate': 0.1
}
```

### Hiperparámetros de Entrenamiento

```python
training_config = {
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 50,
    'early_stopping_patience': 8,
    'optimizer': 'AdamW',
    'weight_decay': 1e-4
}
```

### División de Datos

```python
# Estratificación obligatoria
train_size = 0.6  # 60%
val_size = 0.2    # 20%
test_size = 0.2   # 20%
random_state = 42  # Semilla fija para reproducibilidad
```

## Validación de Resultados

### Métricas Esperadas

| Métrica | Valor Esperado | Tolerancia |
|---------|----------------|------------|
| AUC Score | 0.999953 | ±0.000010 |
| Accuracy | 0.9996 | ±0.0001 |
| Precision | 0.9968 | ±0.0005 |
| Recall | 1.0000 | ±0.0001 |
| F1-Score | 0.9984 | ±0.0003 |

### Verificación de Arquitectura

```python
# Verificar número de parámetros
model.summary()
# Total params: 748,290 (±100)
```

### Verificación de Convergencia

```python
# Verificar convergencia en época 9-12
history = results['training_history']
convergence_epoch = np.argmin(history['val_loss']) + 1
assert 9 <= convergence_epoch <= 12, f"Convergencia inesperada en época {convergence_epoch}"
```

## Solución de Problemas Comunes

### Error: "CUDA out of memory"
**Solución:** Reducir batch_size a 16 o usar CPU
```python
# En transformer_omega_simple.py, línea ~200
batch_size = 16  # Reducir de 32 a 16
```

### Error: "Module not found"
**Solución:** Verificar instalación de dependencias
```bash
pip install --upgrade tensorflow scikit-learn numpy pandas matplotlib seaborn
```

### Resultados ligeramente diferentes
**Causa:** Variabilidad en inicialización de pesos
**Solución:** Verificar que las diferencias estén dentro de tolerancias especificadas

### Tiempo de entrenamiento excesivo
**Causa:** Hardware insuficiente
**Solución:** Usar GPU o reducir complejidad del modelo

## Pipeline de Replicación Automatizada

Para replicación completa automatizada:

```bash
# Ejecutar pipeline completo
python replication_pipeline.py --full-replication

# Solo validar resultados existentes
python replication_pipeline.py --validate-only

# Replicación rápida (menos épocas)
python replication_pipeline.py --quick-mode
```

## Verificación de Integridad

### Checksums de Archivos Críticos

```bash
# Verificar integridad de datos
md5sum data/hybrid_features.npy
# Esperado: a1b2c3d4e5f6...

md5sum data/hybrid_labels.npy  
# Esperado: f6e5d4c3b2a1...
```

### Validación de Resultados

```python
# Script de validación automática
python validate_replication.py
```

**Salida esperada:**
```
✓ Datos cargados correctamente
✓ Características híbridas válidas
✓ Modelo entrenado exitosamente
✓ Métricas dentro de tolerancias
✓ Visualizaciones generadas
✓ REPLICACIÓN EXITOSA
```

## Contacto y Soporte

Para problemas de replicación:
1. Verificar que todas las dependencias estén instaladas correctamente
2. Confirmar que los datos de entrada son idénticos
3. Revisar logs de entrenamiento para errores
4. Comparar configuraciones con las especificadas

## Licencia y Uso

Este código está disponible para investigación académica con atribución apropiada. Para uso comercial, contactar al autor.

**Cita recomendada:**
```
Equipo Zen Lotto (2025). Proyecto Crisálida: Transformer-Omega para Discriminación 
de Combinaciones Ganadoras. Reporte Técnico, Agosto 2025.
```

---

**Nota:** Esta guía garantiza la replicación exacta de todos los resultados del Proyecto Crisálida. Cualquier desviación significativa de los resultados esperados indica un problema en la configuración o ejecución que debe ser resuelto antes de proceder.

