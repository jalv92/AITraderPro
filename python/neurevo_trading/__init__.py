# python/neurevo_trading/__init__.py
"""
NeurEvo Trading - Sistema de trading algorítmico que combina redes neuronales, 
detección de patrones y estrategias de reversión de precios.

Este paquete contiene módulos para:
- Detección de patrones de precios mediante técnicas de aprendizaje profundo
- Implementación de agentes de trading basados en aprendizaje por refuerzo
- Entornos de simulación de mercados financieros
- Herramientas para backtesting y trading en vivo
- Utilidades para visualización y procesamiento de datos
- Integración con el framework NeurEvo para aprendizaje por refuerzo avanzado
"""

__version__ = "0.1.0"
__author__ = "NeurEvo Trading Team"

from neurevo_trading.config import Config

# Configuración global
config = Config()

# Intentar importar NeurEvo (no falla si no está disponible)
try:
    import neurevo
    has_neurevo = True
except ImportError:
    has_neurevo = False