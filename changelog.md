# Changelog del Proyecto AITraderPro

Este archivo registra todos los cambios significativos realizados en el proyecto.

## [Unreleased]

### Añadido
- Creación de la estructura inicial del proyecto
  - Estructura de directorios para NinjaTrader (indicators y strategies)
  - Estructura de directorios para Python (neurevo_trading con environment, agents, models, utils)
  - Estructura de directorios para notebooks
- Movimiento de archivos existentes a su ubicación adecuada:
  - RLExtractor.cs → ninjatrader/indicators/AITraderDataExtractor.cs
  - RLExecutor.cs → ninjatrader/strategies/AITraderExecutor.cs
- Mejoras en la integración de detección de patrones en AITraderExecutor.cs:
  - Nuevos campos para seguimiento de patrones (tipo, confianza, ratio riesgo/recompensa)
  - Propiedades de gestión de riesgo (MaxRiskPercent, MinRiskReward)
  - Mejora del procesamiento de señales de trading para incluir información de patrones
  - Actualización del sistema de envío de datos de ejecución para incluir información de patrones

### Modificado
- AITraderExecutor.cs: Actualizado para soportar detección de patrones y gestión de riesgo mejorada
  - Modificación del formato de mensajes para incluir tipo de patrón, confianza y ratio riesgo/recompensa
  - Actualización de la validación de señales para incluir verificación de ratio riesgo/recompensa mínimo
  - Mejora de los mensajes de log para incluir información de patrones

### Eliminado

## [0.0.1] - 2025-03-08
- Configuración inicial del proyecto 