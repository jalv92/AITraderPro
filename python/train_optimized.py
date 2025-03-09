#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script optimizado para entrenar el cerebro NeurEvo con conjuntos de datos grandes.
Incluye optimizaciones de rendimiento, manejo robusto de errores y checkpoints automáticos.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pickle
import time
import gc
import warnings
import traceback
import logging
from tqdm import tqdm

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger('neurevo_training')

# Filtrar advertencias menos importantes
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Intentar importar torch para detección de GPU
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch no está disponible. No se podrá usar aceleración GPU.")

# Importar NeurEvo
logger.info("Importando neurevo...")
try:
    import neurevo
    logger.info("Neurevo importado correctamente")
except ImportError as e:
    logger.error(f"Error al importar neurevo: {e}")
    logger.error("Asegúrate de que neurevo está instalado correctamente")
    sys.exit(1)

# Importar componentes de neurevo_trading
try:
    from neurevo_trading.environment.trading_env import TradingEnvironment
    from neurevo_trading.environment.data_processor import DataProcessor
    from neurevo_trading.environment.neurevo_adapter import NeurEvoEnvironmentAdapter
    from neurevo_trading.agents.neurevo_agent import NeurEvoTradingAgent
    from neurevo_trading.utils.visualization import plot_equity_curve, plot_training_progress
except ImportError as e:
    logger.error(f"Error al importar componentes de neurevo_trading: {e}")
    logger.error("Asegúrate de que todos los módulos necesarios están disponibles")
    sys.exit(1)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="NeurEvo Trading - Entrenamiento Optimizado")
    
    # Argumentos básicos
    parser.add_argument("--data", type=str, default="data/combined_data_optimized.csv",
                       help="Ruta al archivo CSV de datos optimizados")
    parser.add_argument("--config", type=str, help="Ruta al archivo de configuración de NeurEvo")
    parser.add_argument("--episodes", type=int, default=10000, help="Número de episodios para entrenar")
    
    # Parámetros del entorno
    parser.add_argument("--window", type=int, default=100, help="Tamaño de la ventana de observación")
    parser.add_argument("--balance", type=float, default=10000, help="Balance inicial")
    parser.add_argument("--commission", type=float, default=0.0, help="Comisión por operación")
    parser.add_argument("--enable-sl-tp", action="store_true", 
                      help="Habilitar Stop Loss y Take Profit automáticos")
    parser.add_argument("--default-sl", type=float, default=0.02, 
                      help="Porcentaje de Stop Loss por defecto (0.02 = 2%)")
    parser.add_argument("--default-tp", type=float, default=0.03, 
                      help="Porcentaje de Take Profit por defecto (0.03 = 3%)")
    parser.add_argument("--dd-thresholds", type=str, default="0.05,0.10,0.15,0.20",
                      help="Umbrales de drawdown para penalizaciones progresivas (separados por comas)")
    
    # Parámetros de salida
    parser.add_argument("--output", type=str, default="models", help="Directorio de salida para modelos")
    parser.add_argument("--model-name", type=str, help="Nombre base para el modelo guardado")
    
    # Parámetros de evaluación
    parser.add_argument("--eval", type=int, default=5, help="Número de episodios para evaluación")
    parser.add_argument("--save-interval", type=int, default=1000, 
                       help="Guardar modelo cada N episodios")
    
    # Parámetros de entrenamiento avanzados
    parser.add_argument("--target-angle", type=float, default=45.0, 
                       help="Ángulo objetivo para la curva de equity (en grados)")
    parser.add_argument("--checkpoint", type=str, 
                       help="Archivo de checkpoint para continuar entrenamiento previo")
    
    # Optimizaciones de rendimiento
    parser.add_argument("--gpu", action="store_true", 
                       help="Usar GPU para el entrenamiento si está disponible")
    parser.add_argument("--optimize-memory", action="store_true", 
                       help="Activar optimizaciones agresivas de memoria")
    parser.add_argument("--max-data-rows", type=int, default=None,
                       help="Limitar número de filas de datos (útil para pruebas)")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Tamaño de batch para entrenar el modelo")
    
    # Opciones de logging
    parser.add_argument("--verbose", action="store_true", help="Mostrar información detallada")
    parser.add_argument("--log-level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help="Nivel de detalle del log")
    parser.add_argument("--progress-bar", action="store_true",
                       help="Mostrar barra de progreso durante entrenamiento")
    
    return parser.parse_args()

def load_config(config_path):
    """Load or create NeurEvo configuration"""
    if not config_path or not os.path.exists(config_path):
        # Configuración optimizada por defecto
        return {
            "hidden_layers": [512, 256, 128, 64],  # Redes más profundas
            "learning_rate": 0.0001,
            "batch_size": 256,
            "memory_size": 500000,
            "curiosity_weight": 0.05,
            "dynamic_network": True,  # Permite evolución de la red
            "hebbian_learning": True,  # Mejora conexiones exitosas
            "episodic_memory": True,   # Usa memoria episódica para transferencia
            "reward_shaping": True,    # Modifica recompensas para mejor convergencia
            "reward_scale": 0.1,       # Escala recompensas para evitar explosiones
            "exploration_rate": 0.2,   # Tasa de exploración inicial
            "exploration_decay": 0.9995, # Decaimiento lento para mantener exploración
            "gamma": 0.99              # Factor de descuento para recompensas futuras
        }
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error al cargar configuración desde {config_path}: {e}")
        logger.warning("Usando configuración por defecto")
        return load_config(None)

def setup_device(use_gpu):
    """Configure device (CPU/GPU) for training"""
    if use_gpu and TORCH_AVAILABLE:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Usando GPU: {device_name}")
            return device
        else:
            logger.warning("GPU solicitada pero no disponible. Usando CPU.")
    
    return torch.device("cpu") if TORCH_AVAILABLE else None

def load_and_process_data(data_path, data_processor, max_rows=None, optimize_memory=False):
    """Load and preprocess data with robust error handling"""
    try:
        logger.info(f"Cargando datos desde {data_path}...")
        data = data_processor.load_csv(data_path)
        
        # Mostrar información de memoria
        if optimize_memory:
            mem_usage = data.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info(f"Uso de memoria del DataFrame: {mem_usage:.2f} MB")
        
        # Limitar filas si se especificó
        if max_rows and len(data) > max_rows:
            logger.info(f"Limitando datos a {max_rows} filas (de {len(data)} totales)")
            data = data.iloc[:max_rows].copy()
        
        # Normalizar nombres de columnas
        data.columns = [col.lower() for col in data.columns]
        
        # Verificar columnas necesarias
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.error(f"Faltan columnas requeridas en los datos: {missing_cols}")
            sys.exit(1)
        
        # Preparar datos
        logger.info("Preparando datos para entrenamiento...")
        prepared_data = data_processor.prepare_data(data, add_features=True)
        
        # Información sobre los datos
        logger.info(f"Datos preparados: {len(prepared_data)} filas, {len(prepared_data.columns)} características")
        
        # Liberar memoria del DataFrame original
        if optimize_memory:
            logger.info("Liberando memoria del DataFrame original...")
            del data
            gc.collect()
        
        return prepared_data
        
    except Exception as e:
        logger.error(f"Error crítico al procesar datos: {e}")
        traceback.print_exc()
        sys.exit(1)

def create_model_path(output_dir, model_name=None):
    """Create output directory and generate model path"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if model_name:
            return os.path.join(output_dir, f"{model_name}_{timestamp}.pt")
        else:
            return os.path.join(output_dir, f"neurevo_model_{timestamp}.pt")
    except Exception as e:
        logger.error(f"Error al crear directorio de salida: {e}")
        return os.path.join(".", f"neurevo_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")

def save_checkpoint(adapter, config, path, is_final=False):
    """Save training checkpoint with error handling"""
    try:
        with open(path, 'wb') as f:
            pickle.dump({'adapter': adapter, 'config': config}, f)
        
        if is_final:
            logger.info(f"Modelo final guardado en: {path}")
        else:
            logger.info(f"Checkpoint guardado en: {path}")
        
        return True
    except Exception as e:
        logger.error(f"Error al guardar checkpoint: {e}")
        return False

def load_checkpoint(checkpoint_path, adapter):
    """Load checkpoint with error handling"""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning("No se especificó checkpoint o el archivo no existe")
        return False
    
    try:
        logger.info(f"Cargando checkpoint desde {checkpoint_path}...")
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        if 'adapter' in checkpoint_data:
            # Transferir el cerebro del checkpoint al adaptador actual
            adapter.brain = checkpoint_data['adapter'].brain
            logger.info("Checkpoint cargado exitosamente")
            return True
        else:
            logger.warning("Formato de checkpoint no reconocido")
            return False
            
    except Exception as e:
        logger.error(f"Error al cargar checkpoint: {e}")
        return False

def log_training_progress(episode, total_episodes, avg_reward, mean_drawdown, elapsed_time):
    """Format and log training progress"""
    progress_pct = episode / total_episodes * 100
    
    # Estimar tiempo restante
    if elapsed_time > 0 and episode > 0:
        time_per_episode = elapsed_time / episode
        remaining_episodes = total_episodes - episode
        eta_seconds = time_per_episode * remaining_episodes
        
        # Formatear tiempo restante
        if eta_seconds < 60:
            eta = f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            eta = f"{eta_seconds/60:.1f}min"
        else:
            eta = f"{eta_seconds/3600:.1f}h"
    else:
        eta = "?"
    
    logger.info(f"Episodio {episode}/{total_episodes} ({progress_pct:.1f}%) - "
                f"Recompensa: {avg_reward:.2f}, DrawDown: {mean_drawdown:.2%}, "
                f"ETA: {eta}")

def evaluate_model(adapter, env, num_episodes, verbose=False):
    """Evaluate trained model with robust error handling"""
    logger.info(f"Evaluando modelo en {num_episodes} episodios...")
    rewards = []
    equity_curves = []
    
    for i in range(num_episodes):
        try:
            if verbose:
                logger.info(f"Episodio de evaluación {i+1}/{num_episodes}...")
            
            # Usar la versión mejorada de run_episode
            reward = adapter.run_episode(verbose=verbose)
            rewards.append(reward)
            
            # Capturar curva de equity si está disponible
            if hasattr(env, 'equity_curve'):
                equity_curves.append(env.equity_curve)
                
        except Exception as e:
            logger.error(f"Error en episodio de evaluación {i+1}: {e}")
            continue
    
    if not rewards:
        logger.warning("No se completó ningún episodio de evaluación correctamente")
        return None, None
    
    avg_reward = np.mean(rewards)
    logger.info(f"Evaluación completada. Recompensa promedio: {avg_reward:.2f}")
    
    if equity_curves:
        best_idx = np.argmax(rewards)
        return avg_reward, equity_curves[best_idx]
    else:
        return avg_reward, None

def main():
    # Registrar tiempo de inicio
    start_time_total = time.time()
    logger.info("=== NeurEvo Trading - Entrenamiento Optimizado ===")
    
    # Parsear argumentos
    args = parse_args()
    
    # Configurar nivel de log
    logger.setLevel(getattr(logging, args.log_level))
    
    # Mostrar configuración
    logger.info(f"Archivo de datos: {args.data}")
    logger.info(f"Episodios: {args.episodes}")
    logger.info(f"Optimización de memoria: {'Activada' if args.optimize_memory else 'Desactivada'}")
    
    # Configurar device
    device = setup_device(args.gpu)
    
    try:
        # Cargar configuración
        config = load_config(args.config)
        if args.batch_size:
            config["batch_size"] = args.batch_size
            logger.info(f"Batch size establecido a: {args.batch_size}")
        
        # Crear procesador de datos
        data_processor = DataProcessor()
        
        # Cargar y procesar datos
        prepared_data = load_and_process_data(
            args.data, 
            data_processor, 
            max_rows=args.max_data_rows,
            optimize_memory=args.optimize_memory
        )
        
        # Crear entorno de trading
        logger.info("Creando entorno de trading...")
        
        # Convertir umbrales de drawdown de string a lista de floats
        drawdown_thresholds = [float(x) for x in args.dd_thresholds.split(",")]
        
        env = TradingEnvironment(
            data=prepared_data,
            window_size=args.window,
            initial_balance=args.balance,
            commission=args.commission,
            enable_sl_tp=args.enable_sl_tp,
            default_sl_pct=args.default_sl,
            default_tp_pct=args.default_tp,
            drawdown_thresholds=drawdown_thresholds
        )
        
        # Crear adaptador NeurEvo
        logger.info("Inicializando adaptador NeurEvo...")
        adapter = NeurEvoEnvironmentAdapter(env)
        
        # Crear cerebro NeurEvo
        logger.info("Creando cerebro NeurEvo...")
        brain = neurevo.create_brain(config)
        
        # Registrar entorno con cerebro
        logger.info("Registrando entorno con cerebro...")
        env_id = adapter.register_with_brain(brain, env_id="TradingEnv-Optimized")
        
        # Crear agente
        logger.info("Creando agente...")
        agent_id = adapter.create_agent()
        
        # Cargar checkpoint si existe
        if args.checkpoint:
            load_checkpoint(args.checkpoint, adapter)
        
        # Crear ruta para el modelo
        model_path = create_model_path(args.output, args.model_name)
        checkpoint_path = model_path.replace(".pt", "_checkpoint.pt")
        
        # Guardar checkpoint inicial
        save_checkpoint(adapter, config, checkpoint_path)
        
        # Iniciar entrenamiento
        logger.info("\n=== Iniciando Entrenamiento ===")
        logger.info(f"Objetivo: Optimizar trading con ángulo de curva ~{args.target_angle}°")
        
        # Métricas de entrenamiento
        training_metrics = {
            "rewards": [],
            "drawdowns": [],
            "timestamps": []
        }
        
        # Registrar tiempo de inicio del entrenamiento
        start_time = time.time()
        
        # Entrenar agente
        try:
            # Inicializar progress bar si se solicitó
            if args.progress_bar:
                progress = tqdm(total=args.episodes, desc="Entrenando")
            
            # Entrenamiento manual con checkpoints
            episode = 0
            while episode < args.episodes:
                try:
                    # Ejecutar un batch de episodios
                    batch_size = min(args.save_interval, args.episodes - episode)
                    
                    results = adapter.train_agent(
                        episodes=batch_size,
                        verbose=args.verbose
                    )
                    
                    # Actualizar métricas
                    if "rewards" in results:
                        training_metrics["rewards"].extend(results["rewards"])
                    if "avg_drawdowns" in results:
                        training_metrics["drawdowns"].extend(results["avg_drawdowns"])
                    
                    # Actualizar episodio
                    episode += batch_size
                    
                    # Calcular métricas de progreso
                    if training_metrics["rewards"]:
                        recent_rewards = training_metrics["rewards"][-batch_size:]
                        avg_reward = np.mean(recent_rewards)
                        
                        if "avg_drawdowns" in results and results["avg_drawdowns"]:
                            recent_drawdowns = training_metrics["drawdowns"][-batch_size:]
                            mean_drawdown = np.mean(recent_drawdowns)
                        else:
                            mean_drawdown = 0
                    else:
                        avg_reward = 0
                        mean_drawdown = 0
                    
                    # Actualizar barra de progreso
                    if args.progress_bar:
                        progress.update(batch_size)
                        progress.set_postfix({
                            'reward': f'{avg_reward:.2f}',
                            'drawdown': f'{mean_drawdown:.2%}'
                        })
                    
                    # Mostrar progreso
                    elapsed = time.time() - start_time
                    log_training_progress(episode, args.episodes, avg_reward, mean_drawdown, elapsed)
                    
                    # Guardar checkpoint
                    save_checkpoint(adapter, config, checkpoint_path)
                    
                    # Liberar memoria si se solicitó
                    if args.optimize_memory:
                        gc.collect()
                        if TORCH_AVAILABLE and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                except KeyboardInterrupt:
                    logger.warning("\nEntrenamiento interrumpido por el usuario")
                    break
                    
                except Exception as e:
                    logger.error(f"Error durante el batch de entrenamiento: {e}")
                    logger.error(traceback.format_exc())
                    logger.info("Intentando continuar con el siguiente batch...")
                    continue
            
            # Cerrar barra de progreso si existe
            if args.progress_bar:
                progress.close()
                
        except Exception as e:
            logger.error(f"Error crítico durante el entrenamiento: {e}")
            logger.error(traceback.format_exc())
        
        # Calcular tiempo de entrenamiento
        training_time = time.time() - start_time
        logger.info(f"Entrenamiento completado en {training_time:.2f} segundos")
        
        # Guardar modelo final
        save_checkpoint(adapter, config, model_path, is_final=True)
        
        # Evaluar modelo
        if episode > 0:  # Solo evaluar si se entrenó al menos un episodio
            avg_reward, best_equity = evaluate_model(
                adapter, 
                env, 
                args.eval, 
                verbose=args.verbose
            )
            
            # Visualizar resultados
            if best_equity is not None:
                try:
                    # Crear visualización de la curva de equity
                    equity_curve_path = model_path.replace(".pt", "_equity.png")
                    plot_equity_curve(
                        best_equity, 
                        title=f"Mejor Curva de Equity (Total: {len(prepared_data)} barras)",
                        save_path=equity_curve_path
                    )
                    logger.info(f"Curva de equity guardada en: {equity_curve_path}")
                    
                    # Crear visualización del progreso del entrenamiento
                    if training_metrics["rewards"]:
                        training_plot_path = model_path.replace(".pt", "_training.png")
                        plot_training_progress(
                            {
                                "rewards": training_metrics["rewards"], 
                                "avg_drawdowns": training_metrics["drawdowns"]
                            },
                            save_path=training_plot_path
                        )
                        logger.info(f"Gráfico de progreso guardado en: {training_plot_path}")
                except Exception as e:
                    logger.error(f"Error al crear visualizaciones: {e}")
            
            # Guardar resultados en formato JSON
            results_path = model_path.replace(".pt", "_results.json")
            try:
                with open(results_path, 'w') as f:
                    json.dump({
                        "training": {
                            "episodes": episode,
                            "target_episodes": args.episodes,
                            "avg_reward_final": float(np.mean(training_metrics["rewards"][-min(100, len(training_metrics["rewards"])):])) if training_metrics["rewards"] else 0.0,
                            "training_time": training_time
                        },
                        "evaluation": {
                            "episodes": args.eval,
                            "avg_reward": float(avg_reward) if avg_reward is not None else 0.0
                        },
                        "config": config,
                        "data_info": {
                            "file": args.data,
                            "rows": len(prepared_data),
                            "columns": len(prepared_data.columns),
                            "window_size": args.window
                        },
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }, f, indent=4)
                
                logger.info(f"Resultados guardados en: {results_path}")
            except Exception as e:
                logger.error(f"Error al guardar resultados: {e}")
        
        # Tiempo total
        total_time = time.time() - start_time_total
        logger.info(f"\n=== Proceso Completo ===")
        logger.info(f"Tiempo total: {total_time:.2f} segundos")
        
    except Exception as e:
        logger.error(f"Error general en el proceso: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 