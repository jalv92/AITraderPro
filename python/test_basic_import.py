#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script simple para probar las importaciones básicas y el entorno de entrenamiento.
"""

import os
import sys
import numpy as np
import pandas as pd
import traceback

print("Comprobando importaciones básicas...")

try:
    # Importar módulos de neurevo_trading
    print("Importando módulos de neurevo_trading...")
    from neurevo_trading.environment.trading_env import TradingEnvironment
    from neurevo_trading.environment.data_processor import DataProcessor
    print("Módulos de entorno importados correctamente")
    
    # Importar visualización
    print("Importando visualización...")
    from neurevo_trading.utils.visualization import plot_equity_curve, plot_training_progress
    print("Módulos de visualización importados correctamente")
    
    # Importar neurevo
    print("Importando neurevo...")
    import neurevo
    print("Neurevo importado correctamente")
    
    # Importar adaptador NeurEvo
    print("Importando adaptador NeurEvo...")
    from neurevo_trading.environment.neurevo_adapter import NeurEvoEnvironmentAdapter
    print("Adaptador importado correctamente")
    
    # Todas las importaciones funcionan
    print("\n✅ Todas las importaciones funcionan correctamente!\n")
    
    # Intentar cargar una pequeña muestra de datos
    print("Intentando cargar datos de prueba...")
    data_processor = DataProcessor()
    
    # Usamos una muestra pequeña de los datos para verificar el procesamiento
    data_file = "data/combined_data_optimized.csv"
    if os.path.exists(data_file):
        # Cargar solo 100 filas para una prueba rápida
        data = data_processor.load_csv(data_file, nrows=100)
        print(f"Datos cargados exitosamente: {len(data)} filas")
        
        # Mostrar los nombres de las columnas
        print("Columnas disponibles:")
        for col in data.columns:
            print(f"  - {col}")
        
        # Intentar preprocesar los datos
        print("\nPreprocesando datos...")
        processed_data = data_processor.prepare_data(data)
        print(f"Datos procesados exitosamente: {len(processed_data)} filas")
        
        # Intentar crear un entorno de trading
        print("\nCreando entorno de trading...")
        env = TradingEnvironment(
            data=processed_data,
            window_size=20,
            initial_balance=10000
        )
        print("Entorno creado exitosamente")
        
        # Intentar obtener una observación
        print("\nObteniendo observación inicial...")
        obs = env.reset()
        print(f"Observación obtenida: forma {obs.shape}")
        
        # Intentar ejecutar un paso con acción aleatoria
        print("\nEjecutando paso aleatorio...")
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Paso ejecutado exitosamente. Recompensa: {reward}")
        
        print("\n✅ Pruebas básicas del entorno completadas exitosamente!")
    else:
        print(f"ERROR: No se encuentra el archivo de datos: {data_file}")
        
except Exception as e:
    print(f"\n❌ Error durante las pruebas: {e}")
    traceback.print_exc()
    sys.exit(1) 