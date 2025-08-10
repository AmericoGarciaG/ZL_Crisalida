#!/usr/bin/env python3
"""
Proyecto Crisálida - Fase 1: Análisis del Top 10 de Características Contextuales
Identificación y preparación de las características más prometedoras del Context-Aware Transformer
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

import config
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Top10FeatureAnalyzer:
    """Analizador para identificar y preparar el Top 10 de características contextuales"""
    
    def __init__(self, feature_importance_path: str = config.FEATURE_IMPORTANCE_PATH):
        self.feature_importance_path = feature_importance_path
        self.feature_importance = None
        self.top10_features = None
        
    def load_feature_importance(self):
        """Carga el reporte de importancia de características"""
        logging.info("Cargando reporte de importancia de características...")
        
        with open(self.feature_importance_path, 'r') as f:
            self.feature_importance = json.load(f)
        
        logging.info(f"Modelo: {self.feature_importance['model_name']}")
        logging.info(f"Métrica: {self.feature_importance['metric']}")
        logging.info(f"Total de características: {len(self.feature_importance['feature_importance'])}")
        
    def identify_top10(self):
        """Identifica las Top 10 características más importantes"""
        logging.info("Identificando Top 10 características...")
        
        # Extraer características ordenadas por importancia
        features = self.feature_importance['feature_importance']
        
        # Tomar las primeras 10
        self.top10_features = features[:10]
        
        logging.info("Top 10 características identificadas:")
        for i, feature in enumerate(self.top10_features, 1):
            logging.info(f"  {i:2d}. {feature['feature_name']}: {feature['importance_score']:.4f}")
        
        return self.top10_features
    
    def analyze_feature_categories(self):
        """Analiza las categorías de las características Top 10"""
        logging.info("Analizando categorías de características...")
        
        categories = {
            'Afinidad': [],
            'Frecuencia/Historia': [],
            'Terminación': [],
            'Matemática': [],
            'Estadística': [],
            'Década': [],
            'Paridad': [],
            'Valor': []
        }
        
        for feature in self.top10_features:
            name = feature['feature_name']
            
            if 'afinidad' in name:
                categories['Afinidad'].append(feature)
            elif 'frecuencia' in name or 'historica' in name:
                categories['Frecuencia/Historia'].append(feature)
            elif 'term_' in name:
                categories['Terminación'].append(feature)
            elif 'mult_' in name or 'fibonacci' in name or 'primo' in name:
                categories['Matemática'].append(feature)
            elif 'desv_' in name or 'media' in name:
                categories['Estadística'].append(feature)
            elif 'decada_' in name:
                categories['Década'].append(feature)
            elif 'par' in name or 'impar' in name:
                categories['Paridad'].append(feature)
            elif 'valor' in name or 'normalizado' in name:
                categories['Valor'].append(feature)
        
        # Mostrar análisis por categorías
        logging.info("\nAnálisis por categorías:")
        for category, features in categories.items():
            if features:
                logging.info(f"\n{category} ({len(features)} características):")
                for feature in features:
                    logging.info(f"  - {feature['feature_name']}: {feature['importance_score']:.4f}")
        
        return categories
    
    def create_visualization(self):
        """Crea visualización del Top 10"""
        logging.info("Creando visualización del Top 10...")
        
        # Preparar datos
        names = [f['feature_name'] for f in self.top10_features]
        scores = [f['importance_score'] for f in self.top10_features]
        
        # Crear figura
        plt.figure(figsize=(12, 8))
        
        # Crear colores por categoría
        colors = []
        for name in names:
            if 'afinidad' in name:
                colors.append('#FF6B6B')  # Rojo para afinidad
            elif 'frecuencia' in name:
                colors.append('#4ECDC4')  # Verde azulado para frecuencia
            elif 'term_' in name:
                colors.append('#45B7D1')  # Azul para terminación
            elif 'mult_' in name or 'fibonacci' in name:
                colors.append('#96CEB4')  # Verde para matemática
            elif 'desv_' in name:
                colors.append('#FFEAA7')  # Amarillo para estadística
            elif 'decada_' in name:
                colors.append('#DDA0DD')  # Púrpura para década
            elif 'par' in name or 'impar' in name:
                colors.append('#FFB347')  # Naranja para paridad
            else:
                colors.append('#B0B0B0')  # Gris para otros
        
        # Crear gráfico de barras horizontal
        bars = plt.barh(range(len(names)), scores, color=colors, alpha=0.8)
        
        # Personalizar
        plt.yticks(range(len(names)), names)
        plt.xlabel('Importancia (Cohen\'s d)', fontsize=12)
        plt.title('Top 10 Características Contextuales - Proyecto Crisálida', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Añadir valores en las barras
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(score + 0.002, bar.get_y() + bar.get_height()/2, 
                    f'{score:.4f}', ha='left', va='center', fontsize=10)
        
        # Añadir línea de referencia
        plt.axvline(x=0.01, color='red', linestyle='--', alpha=0.5, label='Umbral de relevancia')
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        # Guardar
        plt.savefig(config.TOP10_ANALYSIS_PNG_PATH, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Visualización guardada en ../results/top10_features_analysis.png")
    
    def generate_feature_mapping(self):
        """Genera mapeo de características para implementación"""
        logging.info("Generando mapeo de características...")
        
        feature_mapping = {
            'top10_contextual': {},
            'implementation_notes': {},
            'categories': {}
        }
        
        for i, feature in enumerate(self.top10_features):
            name = feature['feature_name']
            feature_mapping['top10_contextual'][f'feature_{i+1}'] = {
                'name': name,
                'importance': feature['importance_score'],
                'description': feature['description'],
                'rank': i + 1
            }
            
            # Notas de implementación
            if 'afinidad' in name:
                feature_mapping['implementation_notes'][name] = "Ya implementada en proyecto anterior - reutilizar"
            elif 'term_' in name:
                feature_mapping['implementation_notes'][name] = "One-hot encoding de terminación"
            elif 'mult_' in name:
                feature_mapping['implementation_notes'][name] = "Innovación de Manus - verificar múltiplos"
            elif 'desv_' in name:
                feature_mapping['implementation_notes'][name] = "Estadística emergente - calcular por combinación"
            else:
                feature_mapping['implementation_notes'][name] = "Característica estándar - implementar según descripción"
        
        # Guardar mapeo
        with open(config.TOP10_MAPPING_PATH, 'w') as f:
            json.dump(feature_mapping, f, indent=2)
        
        logging.info("Mapeo guardado en ../results/top10_feature_mapping.json")
        
        return feature_mapping
    
    def calculate_cumulative_importance(self):
        """Calcula la importancia acumulativa del Top 10"""
        logging.info("Calculando importancia acumulativa...")
        
        # Calcular importancia total de todas las características
        all_features = self.feature_importance['feature_importance']
        total_importance = sum(f['importance_score'] for f in all_features)
        
        # Calcular importancia del Top 10
        top10_importance = sum(f['importance_score'] for f in self.top10_features)
        
        # Porcentajes
        top10_percentage = (top10_importance / total_importance) * 100
        
        logging.info(f"Importancia total (todas las características): {total_importance:.4f}")
        logging.info(f"Importancia Top 10: {top10_importance:.4f}")
        logging.info(f"Porcentaje capturado por Top 10: {top10_percentage:.2f}%")
        
        # Análisis de dominancia de afinidad
        afinidad_importance = sum(f['importance_score'] for f in self.top10_features 
                                if 'afinidad' in f['feature_name'])
        afinidad_percentage = (afinidad_importance / top10_importance) * 100
        
        logging.info(f"Importancia de afinidad en Top 10: {afinidad_percentage:.2f}%")
        
        return {
            'total_importance': total_importance,
            'top10_importance': top10_importance,
            'top10_percentage': top10_percentage,
            'afinidad_dominance': afinidad_percentage
        }
    
    def generate_summary_report(self):
        """Genera reporte resumen de la Fase 1"""
        logging.info("Generando reporte resumen...")
        
        categories = self.analyze_feature_categories()
        stats = self.calculate_cumulative_importance()
        
        report = f"""# Proyecto Crisálida - Fase 1: Análisis Top 10 Características Contextuales

## Resumen Ejecutivo

Se ha completado el análisis de las características contextuales del Context-Aware Transformer para identificar las 10 más prometedoras que servirán como base para el modelo híbrido Transformer-Omega.

## Top 10 Características Seleccionadas

"""
        
        for i, feature in enumerate(self.top10_features, 1):
            report += f"{i:2d}. **{feature['feature_name']}**: {feature['importance_score']:.4f}\n"
            report += f"    - {feature['description']}\n\n"
        
        report += f"""
## Análisis Estadístico

- **Importancia total capturada**: {stats['top10_percentage']:.2f}% de todas las características
- **Dominancia de afinidad**: {stats['afinidad_dominance']:.2f}% del Top 10
- **Característica más importante**: {self.top10_features[0]['feature_name']} ({self.top10_features[0]['importance_score']:.4f})

## Distribución por Categorías

"""
        
        for category, features in categories.items():
            if features:
                report += f"- **{category}**: {len(features)} características\n"
        
        report += f"""
## Conclusiones para la Fase 2

1. **Afinidad de pares** sigue siendo la característica más discriminativa (0.3289)
2. Las características de **terminación** (6 y 9) muestran importancia significativa
3. Las **innovaciones de Manus** (múltiplo de 3, desviación de media) están en el Top 10
4. El Top 10 captura {stats['top10_percentage']:.1f}% de la importancia total, justificando la selección

## Próximos Pasos

- Implementar cálculo de afinidades de pares, tercias, cuartetos y quintetos
- Combinar Top 10 contextual con métricas de afinidad
- Diseñar arquitectura Transformer-Omega con fusión inteligente
"""
        
        # Guardar reporte
        with open(config.REPORTS_DIR / 'fase1_top10_analysis.md', 'w') as f:
            f.write(report)
        
        logging.info("Reporte guardado en ../reports/fase1_top10_analysis.md")
        
        return report

def main():
    utils.ensure_dirs_exist()
    """Función principal de la Fase 1"""
    logging.info("=== PROYECTO CRISÁLIDA - FASE 1: ANÁLISIS TOP 10 ===")
    
    # Inicializar analizador
    analyzer = Top10FeatureAnalyzer()
    
    # Cargar datos
    analyzer.load_feature_importance()
    
    # Identificar Top 10
    top10 = analyzer.identify_top10()
    
    # Análisis por categorías
    categories = analyzer.analyze_feature_categories()
    
    # Crear visualización
    analyzer.create_visualization()
    
    # Generar mapeo
    mapping = analyzer.generate_feature_mapping()
    
    # Estadísticas
    stats = analyzer.calculate_cumulative_importance()
    
    # Reporte final
    report = analyzer.generate_summary_report()
    
    logging.info("=== FASE 1 COMPLETADA EXITOSAMENTE ===")
    logging.info(f"Top 10 características identificadas y analizadas")
    logging.info(f"Capturan {stats['top10_percentage']:.1f}% de la importancia total")
    logging.info(f"Afinidad de pares domina con {top10[0]['importance_score']:.4f}")

if __name__ == "__main__":
    main()

