#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para entrenar el cerebro NeurEvo con un conjunto de datos completo.
Optimizado para manejar grandes conjuntos de datos y entrenamientos largos.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pickle
import sys
import time
import gc  # Garbage collector para liberar memoria

# Importar NeurEvo
print("Importando neurevo...", file=sys.stderr)
import neurevo
print("Neurevo importado con éxito", file=sys.stderr)

# Importar componentes de neurevo_trading
from neurevo_trading.environment.trading_env import TradingEnvironment
from neurevo_trading.environment.data_processor import DataProcessor
from neurevo_trading.environment.neurevo_adapter import NeurEvoEnvironmentAdapter
from neurevo_trading.agents.neurevo_agent import NeurEvoTradingAgent
from neurevo_trading.utils.visualization import plot_equity_curve, plot_training_progress

def parse_args():
    parser = argparse.ArgumentParser(description="NeurEvo Trading - Entrenamiento con Datos Completos")
    
    parser.add_argument("--data", type=str, default="data/combined_data.csv",
                       help="Ruta al archivo CSV de datos combinados")
    parser.add_argument("--config", type=str, help="Ruta al archivo de configuración de NeurEvo")
    parser.add_argument("--episodes", type=int, default=10000, help="Número de episodios para entrenar")
    parser.add_argument("--window", type=int, default=100, help="Tamaño de la ventana de observación")
    parser.add_argument("--balance", type=float, default=10000, help="Balance inicial")
    parser.add_argument("--commission", type=float, default=0.0, help="Comisión por operación")
    parser.add_argument("--output", type=str, default="models", help="Directorio de salida para modelos")
    parser.add_argument("--eval", type=int, default=5, help="Número de episodios para evaluación")
    parser.add_argument("--save-interval", type=int, default=1000, 
                       help="Guardar modelo cada N episodios")
    parser.add_argument("--target-angle", type=float, default=45.0, 
                       help="Ángulo objetivo para la curva de equity (en grados)")
    parser.add_argument("--checkpoint", type=str, 
                       help="Archivo de checkpoint para continuar entrenamiento previo")
    parser.add_argument("--gpu", action="store_true", help="Usar GPU para el entrenamiento si está disponible")
    parser.add_argument("--optimize-memory", action="store_true", 
                       help="Optimizar uso de memoria para conjuntos de datos grandes")
    parser.add_argument("--max-data-rows", type=int, default=None,
                       help="Limitar número de filas de datos para pruebas rápidas")
    parser.add_argument("--verbose", action="store_true", help="Mostrar información detallada")
    
    return parser.parse_args()

def load_config(config_path):
    if not config_path or not os.path.exists(config_path):
        # Configuración optimizada para datos completos y entrenamiento largo
        return {
            "hidden_layers": [512, 256, 128, 64],
            "learning_rate": 0.0001,
            "batch_size": 256,
            "memory_size": 500000,
            "curiosity_weight": 0.05,
            "dynamic_network": True,
            "hebbian_learning": True,
            "episodic_memory": True,
            "reward_shaping": True,
            "reward_scale": 0.1,
            "exploration_rate": 0.2,
            "exploration_decay": 0.9995,
            "gamma": 0.99
        }
    
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    start_time_total = time.time()
    print("=== NeurEvo Trading - Entrenamiento con Datos Completos ===")
    
    # Parsear argumentos
    args = parse_args()
    print(f"Archivo de datos: {args.data}")
    print(f"Episodios de entrenamiento: {args.episodes}")
    
    # Verificar existencia del archivo de datos
    if not os.path.exists(args.data):
        print(f"ERROR: No se encuentra el archivo de datos: {args.data}")
        return
    
    # Configurar uso de GPU
    if args.gpu:
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Usando dispositivo: {device}")
            if device.type == "cpu" and args.gpu:
                print("WARNING: GPU solicitada pero no disponible, usando CPU")
        except:
            print("WARNING: No se pudo configurar GPU, usando CPU")
    
    try:
        # Crear directorio de salida si no existe
        os.makedirs(args.output, exist_ok=True)
        
        # Cargar configuración
        config = load_config(args.config)
        print("Configuración cargada")
        
        # Cargar y procesar datos
        print(f"Cargando datos desde {args.data}...")
        data_processor = DataProcessor()
        data = data_processor.load_csv(args.data)
        
        # Limitar datos si se especificó
        if args.max_data_rows and len(data) > args.max_data_rows:
            print(f"Limitando datos a {args.max_data_rows} filas (de {len(data)} disponibles)")
            data = data.iloc[:args.max_data_rows].copy()
        
        # Asegurar que los nombres de columnas estén en minúsculas
        data.columns = [col.lower() for col in data.columns]
        print("Nombres de columnas convertidos a minúsculas")
        
        # Preparar datos
        print("Preparando datos...")
        prepared_data = data_processor.prepare_data(data, add_features=True)
        print(f"Datos preparados: {len(prepared_data)} filas con {len(prepared_data.columns)} características")
        
        # Liberar memoria de objetos que ya no necesitamos
        if args.optimize_memory:
            del data
            gc.collect()
            print("Memoria liberada de datos originales")
        
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
        
        # Crear cerebro NeurEvo con configuración optimizada
        print("Creando cerebro NeurEvo...")
        brain = neurevo.create_brain(config)
        
        # Registrar entorno con cerebro
        print("Registrando entorno con cerebro...")
        env_id = adapter.register_with_brain(brain, env_id="TradingEnv-Full")
        
        # Crear agente
        print("Creando agente...")
        agent_id = adapter.create_agent()
        
        # Cargar checkpoint si se especificó
        if args.checkpoint and os.path.exists(args.checkpoint):
            print(f"Cargando checkpoint desde {args.checkpoint}...")
            try:
                with open(args.checkpoint, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    if 'adapter' in checkpoint_data:
                        # Intentar transferir solo los pesos del modelo anterior
                        adapter.brain = checkpoint_data['adapter'].brain
                        print("Checkpoint cargado exitosamente")
                    else:
                        print("WARNING: Formato de checkpoint no reconocido, iniciando entrenamiento desde cero")
            except Exception as e:
                print(f"ERROR al cargar checkpoint: {e}")
                print("Iniciando entrenamiento desde cero")
        
        # Iniciar entrenamiento
        print("\n=== Iniciando Entrenamiento ===")
        print(f"Objetivo: Optimizar PnL con ángulo de aproximadamente {args.target_angle}° sin grandes drawdowns")
        print(f"Entrenando por {args.episodes} episodios...")
        
        # Estructura para almacenar métricas durante el entrenamiento
        training_metrics = {
            "rewards": [],
            "drawdowns": [],
            "timestamps": []
        }
        
        start_time = time.time()
        
        # Guardar checkpoint inicial
        timestamp_init = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path_init = os.path.join(args.output, f"checkpoint_init_{timestamp_init}.pt")
        
        try:
            with open(checkpoint_path_init, 'wb') as f:
                pickle.dump({'adapter': adapter, 'config': config}, f)
            print(f"Checkpoint inicial guardado en: {checkpoint_path_init}")
        except Exception as e:
            print(f"WARNING: No se pudo guardar checkpoint inicial: {e}")
        
        # Entrenar agente
        try:
            results = adapter.train_agent(
                episodes=args.episodes,
                verbose=args.verbose
            )
            
            # Guardar métricas de entrenamiento
            training_metrics["rewards"] = results.get("rewards", [])
            training_metrics["drawdowns"] = results.get("avg_drawdowns", [])
        except KeyboardInterrupt:
            print("\nEntrenamiento interrumpido por el usuario")
            print("Guardando modelo parcial...")
            
            # Guardar el estado actual
            timestamp_interrupted = datetime.now().strftime("%Y%m%d_%H%M%S")
            interrupted_path = os.path.join(args.output, f"neurevo_interrupted_{timestamp_interrupted}.pt")
            
            with open(interrupted_path, 'wb') as f:
                pickle.dump({'adapter': adapter, 'config': config}, f)
            
            print(f"Modelo parcial guardado en: {interrupted_path}")
            results = {"rewards": training_metrics["rewards"]}
        except Exception as e:
            print(f"ERROR durante el entrenamiento: {e}")
            import traceback
            traceback.print_exc()
            
            # Intentar guardar el modelo parcial
            try:
                timestamp_error = datetime.now().strftime("%Y%m%d_%H%M%S")
                error_path = os.path.join(args.output, f"neurevo_error_{timestamp_error}.pt")
                
                with open(error_path, 'wb') as f:
                    pickle.dump({'adapter': adapter, 'config': config}, f)
                
                print(f"Modelo parcial guardado en: {error_path}")
            except:
                print("No se pudo guardar el modelo parcial")
            
            results = {"rewards": training_metrics["rewards"]}
        
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Entrenamiento completado en {training_time:.2f} segundos")
        
        # Guardar modelo final
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(args.output, f"neurevo_full_model_{timestamp}.pt")
        
        with open(model_path, 'wb') as f:
            pickle.dump({'adapter': adapter, 'config': config}, f)
        
        print(f"Modelo guardado en: {model_path}")
        
        # Evaluar agente
        print("\n=== Evaluando Agente ===")
        rewards = []
        equity_curves = []
        
        evaluation_success = True
        
        for i in range(args.eval):
            try:
                print(f"Episodio de evaluación {i+1}/{args.eval}...")
                reward = adapter.run_episode()
                rewards.append(reward)
                
                if hasattr(env, 'equity_curve'):
                    equity_curves.append(env.equity_curve)
            except Exception as e:
                print(f"ERROR en episodio de evaluación {i+1}: {e}")
                evaluation_success = False
                continue
        
        if rewards:
            avg_reward = np.mean(rewards)
            print(f"Recompensa promedio: {avg_reward:.2f}")
        else:
            print("No se completó ningún episodio de evaluación correctamente")
            evaluation_success = False
        
        # Visualizar resultados solo si la evaluación fue exitosa
        if evaluation_success and equity_curves:
            try:
                best_episode = np.argmax(rewards)
                best_equity = equity_curves[best_episode]
                
                # Crear visualización de la curva de equity
                equity_curve_path = os.path.join(args.output, f"equity_curve_{timestamp}.png")
                plot_equity_curve(
                    best_equity, 
                    title=f"Mejor Curva de Equity (Total: {len(prepared_data)} barras)",
                    save_path=equity_curve_path
                )
                print(f"Curva de equity guardada en: {equity_curve_path}")
                
                # Crear visualización del progreso del entrenamiento
                if len(training_metrics["rewards"]) > 0:
                    training_plot_path = os.path.join(args.output, f"training_progress_{timestamp}.png")
                    plot_training_progress(
                        {"rewards": training_metrics["rewards"], "avg_drawdowns": training_metrics["drawdowns"]},
                        save_path=training_plot_path
                    )
                    print(f"Progreso de entrenamiento guardado en: {training_plot_path}")
            except Exception as e:
                print(f"ERROR al crear visualizaciones: {e}")
        
        # Guardar resultados
        results_path = os.path.join(args.output, f"full_training_results_{timestamp}.json")
        
        try:
            with open(results_path, 'w') as f:
                json.dump({
                    "training_results": results,
                    "evaluation": {
                        "episodes": args.eval,
                        "rewards": rewards if rewards else [],
                        "average_reward": float(avg_reward) if rewards else 0.0
                    },
                    "config": config,
                    "training_time": training_time,
                    "data_size": len(prepared_data),
                    "window_size": args.window
                }, f, indent=4)
            
            print(f"Resultados guardados en: {results_path}")
        except Exception as e:
            print(f"ERROR al guardar resultados: {e}")
        
        total_time = time.time() - start_time_total
        print(f"\n=== Entrenamiento y Evaluación Completos ===")
        print(f"Tiempo total: {total_time:.2f} segundos")
        
    except Exception as e:
        print(f"ERROR durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 