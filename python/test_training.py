#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para ejecutar un entrenamiento rápido para probar el sistema
con 100 episodios.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import pickle

# Importar componentes de neurevo_trading
from neurevo_trading.environment.trading_env import TradingEnvironment
from neurevo_trading.environment.data_processor import DataProcessor
from neurevo_trading.environment.neurevo_adapter import NeurEvoEnvironmentAdapter

# Intentar importar neurevo
try:
    import neurevo
    print("Neurevo importado correctamente")
except ImportError:
    print("Error: No se pudo importar neurevo. Asegúrate de que esté instalado.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="AITraderPro - Test de Entrenamiento Rápido")
    
    parser.add_argument("--data", type=str, default="data/raw/MNQ_03-25_Data_20250308_180321.csv", 
                        help="Ruta al archivo CSV de datos")
    parser.add_argument("--episodes", type=int, default=100, 
                        help="Número de episodios para el entrenamiento")
    parser.add_argument("--window", type=int, default=30, 
                        help="Tamaño de la ventana de observación")
    parser.add_argument("--balance", type=float, default=10000, 
                        help="Balance inicial")
    parser.add_argument("--commission", type=float, default=0.0, 
                        help="Comisión por operación")
    parser.add_argument("--eval", type=int, default=5, 
                        help="Número de episodios para evaluación")
    
    return parser.parse_args()

def main():
    print("=== AITraderPro - Test de Entrenamiento Rápido ===")
    
    # Parsear argumentos
    args = parse_args()
    
    # Verificar que el archivo de datos existe
    if not os.path.exists(args.data):
        print(f"Error: No se encuentra el archivo de datos: {args.data}")
        return
    
    try:
        # Cargar y procesar datos
        print(f"Cargando datos desde: {args.data}")
        data_processor = DataProcessor()
        data = data_processor.load_csv(args.data)
        
        # Convertir nombres de columnas a minúsculas
        data.columns = [col.lower() for col in data.columns]
        print("Nombres de columnas convertidos a minúsculas")
        
        prepared_data = data_processor.prepare_data(data, add_features=True)
        
        print(f"Datos cargados: {len(prepared_data)} filas con {len(prepared_data.columns)} características")
        
        # Configuración optimizada para entrenamiento rápido
        config = {
            "hidden_layers": [128, 64, 32],   # Capas más pequeñas para entrenar más rápido
            "learning_rate": 0.001,          # Learning rate más alto para convergencia más rápida
            "batch_size": 64,
            "memory_size": 10000,            # Memoria más pequeña
            "curiosity_weight": 0.1,
            "dynamic_network": True,
            "reward_shaping": True,
            "reward_scale": 0.1,
            "exploration_rate": 0.2,         # Exploración más alta
            "exploration_decay": 0.99,
            "gamma": 0.95                    # Horizonte más corto
        }
        
        # Crear entorno de trading
        print("Creando entorno de trading...")
        env = TradingEnvironment(
            data=prepared_data,
            window_size=args.window,
            initial_balance=args.balance,
            commission=args.commission
        )
        
        # Crear adaptador NeurEvo
        print("Creando adaptador NeurEvo...")
        adapter = NeurEvoEnvironmentAdapter(env)
        
        # Crear cerebro NeurEvo
        print("Creando cerebro NeurEvo...")
        brain = neurevo.create_brain(config)
        
        # Registrar entorno y crear agente
        print("Registrando entorno y creando agente...")
        env_id = adapter.register_with_brain(brain, env_id="TestTradingEnv")
        agent_id = adapter.create_agent()
        
        # Entrenar agente
        print("\n=== Iniciando Entrenamiento de Prueba ===")
        print(f"Entrenando por {args.episodes} episodios...")
        
        start_time = datetime.now()
        results = adapter.train_agent(
            episodes=args.episodes,
            verbose=True
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        print(f"Entrenamiento completado en {training_time:.2f} segundos")
        
        # Evaluar agente
        print("\n=== Evaluando Agente ===")
        rewards = []
        
        for i in range(args.eval):
            reward = adapter.run_episode()
            rewards.append(reward)
        
        avg_reward = np.mean(rewards)
        print(f"Recompensa promedio: {avg_reward:.2f}")
        
        # Guardar modelo en carpeta temporal
        print("\n=== Guardando Modelo de Prueba ===")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("temp_models", exist_ok=True)
        model_path = os.path.join("temp_models", f"test_model_{timestamp}.pt")
        
        with open(model_path, 'wb') as f:
            pickle.dump({'adapter': adapter, 'config': config}, f)
        
        print(f"Modelo guardado en: {model_path}")
        print("=== Test de Entrenamiento Completado ===")
        
    except Exception as e:
        print(f"Error durante el test de entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 