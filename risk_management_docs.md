# Mejoras en la Gestión de Riesgo para AITraderPro

Este documento describe las mejoras implementadas en el sistema de gestión de riesgo de AITraderPro, específicamente en lo relacionado con Stop Loss, Take Profit y penalizaciones por drawdowns excesivos.

## Características Implementadas

### 1. Sistema de Stop Loss y Take Profit

Se ha implementado un sistema configurable de Stop Loss (SL) y Take Profit (TP) que permite al agente de trading aprender a utilizar estos mecanismos de gestión de riesgo:

- **Activación opcional**: Se puede habilitar o deshabilitar mediante el parámetro `--enable-sl-tp`
- **Valores predeterminados configurables**: Mediante los parámetros `--default-sl` y `--default-tp`
- **Configuraciones múltiples**: El agente puede elegir entre 5 configuraciones diferentes de SL/TP para adaptar el riesgo según el contexto de mercado:
  - Conservador (SL 1%, TP 2%): Relación riesgo/recompensa 1:2
  - Equilibrado (SL 2%, TP 3%): Relación riesgo/recompensa 1:1.5
  - Agresivo (SL 3%, TP 6%): Relación riesgo/recompensa 1:2
  - Scalping (SL 1%, TP 1%): Relación riesgo/recompensa 1:1
  - Muy agresivo (SL 5%, TP 10%): Relación riesgo/recompensa 1:2

- **Espacio de acción ampliado**: El espacio de acción se amplía cuando SL/TP está activado:
  - 0: No hacer nada
  - 1: Comprar (Long) con SL/TP por defecto
  - 2: Vender (Short) con SL/TP por defecto
  - 3-7: Comprar con 5 configuraciones diferentes de SL/TP
  - 8-12: Vender con 5 configuraciones diferentes de SL/TP

### 2. Penalizaciones Progresivas por Drawdowns Excesivos

El sistema ahora incluye penalizaciones progresivas por drawdowns (caídas desde máximos) que superan ciertos umbrales:

- **Umbrales configurables**: Se definen mediante el parámetro `--dd-thresholds` (valor por defecto: "0.05,0.10,0.15,0.20")
- **Penalización base**: Para cualquier drawdown, proporcional a su magnitud
- **Penalización progresiva**: Por cada umbral superado, se aplica una penalización adicional con factor exponencial
- **Factor de severidad creciente**: Cada umbral superado multiplica la penalización por un factor de 1.5

Esto incentiva al agente a mantener drawdowns controlados y evitar grandes pérdidas de capital.

## Beneficios

1. **Aprendizaje de gestión de riesgo**: El agente puede aprender cuándo usar configuraciones agresivas o conservadoras
2. **Protección de capital**: Se limitan las pérdidas máximas por operación mediante SL
3. **Aseguramiento de beneficios**: Se capturan beneficios automáticamente mediante TP
4. **Curvas de equity más estables**: La penalización por drawdowns incentiva estrategias con menor volatilidad
5. **Feedback más preciso**: El sistema proporciona señales claras sobre comportamientos indeseados

## Uso Práctico

Para entrenar un modelo con estas mejoras:

```bash
python python/train_optimized.py --episodes 1000 --enable-sl-tp --default-sl 0.02 --default-tp 0.04 --dd-thresholds "0.03,0.07,0.15,0.20"
```

Para evaluar el comportamiento:

```bash
python python/evaluate_model.py --model models/nombre_del_modelo.pt --enable-sl-tp
```

## Estadísticas de Trading

Al habilitar estas mejoras, el sistema ahora registra en cada operación:
- Si el Stop Loss o Take Profit fue activado
- El precio de activación
- La razón exacta del cierre de posición

Esto permite análisis post-entrenamiento más detallados sobre el comportamiento del agente en diferentes condiciones de mercado.

## Próximas Mejoras

Planificadas para futuras versiones:
1. Ajuste dinámico de SL/TP basado en volatilidad de mercado
2. Trailing Stop Loss para asegurar ganancias en tendencias fuertes
3. SL/TP basados en tiempo (expiración automática)
4. Posiciones con tamaño variable basado en nivel de confianza o volatilidad del mercado 