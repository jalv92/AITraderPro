import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pickle
import sys

# Importar NeurEvo
print("Importando neurevo...", file=sys.stderr)
import neurevo
print("Neurevo importado con éxito", file=sys.stderr)

# Importar componentes de neurevo_trading
from neurevo_trading.environment.trading_env import TradingEnvironment
from neurevo_trading.environment.data_processor import DataProcessor
from neurevo_trading.environment.neurevo_adapter import NeurEvoEnvironmentAdapter
from neurevo_trading.agents.neurevo_agent import NeurEvoTradingAgent
from neurevo_trading.utils.visualization import plot_equity_curve

def parse_args():
    parser = argparse.ArgumentParser(description="NeurEvo Trading - Training Script")
    
    parser.add_argument("--data", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("--config", type=str, help="Path to NeurEvo configuration file")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--window", type=int, default=50, help="Window size for observations")
    parser.add_argument("--balance", type=float, default=10000, help="Initial balance")
    parser.add_argument("--commission", type=float, default=0.0, help="Commission fee")
    parser.add_argument("--output", type=str, default="models", help="Output directory for models")
    parser.add_argument("--eval", type=int, default=10, help="Number of episodes for evaluation")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("--target-angle", type=float, default=45.0, help="Target angle for equity curve (in degrees)")
    
    return parser.parse_args()

def load_config(config_path):
    if not config_path or not os.path.exists(config_path):
        # Configuración optimizada para maximizar PnL estable
        return {
            "hidden_layers": [512, 256, 128, 64],
            "learning_rate": 0.0002,
            "batch_size": 128,
            "memory_size": 200000,
            "curiosity_weight": 0.05,
            "dynamic_network": True,
            "hebbian_learning": True,
            "episodic_memory": True,
            "exploration_rate": 0.15,
            "exploration_decay": 0.995,
            "gamma": 0.99
        }
    
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    print("Iniciando script principal...", file=sys.stderr)
    # Parse arguments
    args = parse_args()
    print("Argumentos parseados", file=sys.stderr)
    
    # Load configuration
    config = load_config(args.config)
    print("Configuración cargada", file=sys.stderr)
    
    print("=== NeurEvo Trading Training ===")
    print(f"Enfoque: Optimización de PnL con crecimiento estable (~{args.target_angle}°)")
    print(f"Loading data from: {args.data}")
    print(f"Training for {args.episodes} episodes")
    
    try:
        # Load and prepare data
        print("Cargando datos...", file=sys.stderr)
        data_processor = DataProcessor()
        data = data_processor.load_csv(args.data)
        prepared_data = data_processor.prepare_data(data, add_features=True)
        print("Datos preparados", file=sys.stderr)
        
        print(f"Data loaded: {len(prepared_data)} rows with {len(prepared_data.columns)} features")
        
        # Create training environment
        print("Creando entorno de trading...", file=sys.stderr)
        env = TradingEnvironment(
            data=prepared_data,
            window_size=args.window,
            initial_balance=args.balance,
            commission=args.commission
        )
        print("Entorno creado", file=sys.stderr)
        
        # Create NeurEvo adapter
        print("Creando adaptador NeurEvo...", file=sys.stderr)
        adapter = NeurEvoEnvironmentAdapter(env)
        print("Adaptador creado", file=sys.stderr)
        
        # Create NeurEvo brain with custom config
        custom_config = config.copy()
        # Ajustar parámetros para optimizar el crecimiento estable
        custom_config["reward_shaping"] = True
        custom_config["reward_scale"] = 0.1
        
        print("Creando cerebro NeurEvo con configuración optimizada...", file=sys.stderr)
        brain = neurevo.create_brain(custom_config)
        print("BrainInterface inicializada", file=sys.stderr)
        
        # Register environment with brain
        print("Registrando entorno con cerebro...", file=sys.stderr)
        env_id = adapter.register_with_brain(brain, env_id="TradingEnv")
        print("Entorno registrado", file=sys.stderr)
        
        # Create agent
        print("Creando agente...", file=sys.stderr)
        agent_id = adapter.create_agent()
        print("Agente creado con ID:", agent_id, file=sys.stderr)
        
        print("\n=== Starting Training ===")
        print("Objetivo: Maximizar PnL con crecimiento estable y drawdowns mínimos")
        start_time = datetime.now()
        
        # Train agent
        print("Iniciando entrenamiento...", file=sys.stderr)
        results = adapter.train_agent(
            episodes=args.episodes,
            verbose=args.verbose
        )
        print("Entrenamiento completado", file=sys.stderr)
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Create output directory if it doesn't exist
        print("Creando directorio de salida...", file=sys.stderr)
        os.makedirs(args.output, exist_ok=True)
        
        # Save model
        print("Guardando modelo...", file=sys.stderr)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(args.output, f"neurevo_pnl_optimized_{timestamp}.pt")
        
        # Como la API ha cambiado, guardamos el adaptador que contiene referencias al entorno y agente
        with open(model_path, 'wb') as f:
            pickle.dump({'adapter': adapter, 'config': custom_config}, f)
        
        print(f"Model saved to: {model_path}")
        print("Modelo guardado", file=sys.stderr)
        
        # Evaluate agent
        print("\n=== Evaluating Agent ===")
        print("Evaluando agente...", file=sys.stderr)
        rewards = []
        equity_curves = []
        max_drawdowns = []
        
        for i in range(args.eval):
            # Ejecutar episodio completo
            total_reward = adapter.run_episode()
            rewards.append(total_reward)
            
            # Recopilar información sobre equity y drawdown
            final_balance = env.balance
            max_drawdown = env.max_drawdown
            equity_curve = env.equity_curve
            
            equity_curves.append(equity_curve)
            max_drawdowns.append(max_drawdown)
            
            print(f"Episode {i+1}/{args.eval}: Reward = {total_reward:.2f}, Final Balance = {final_balance:.2f}, Max Drawdown = {max_drawdown:.2%}")
        
        avg_reward = np.mean(rewards)
        avg_drawdown = np.mean(max_drawdowns)
        print(f"Average reward: {avg_reward:.2f}, Average Max Drawdown: {avg_drawdown:.2%}")
        print("Evaluación completada", file=sys.stderr)
        
        # Visualizar la curva de equity del mejor episodio
        best_episode = np.argmax(rewards)
        best_equity_curve = equity_curves[best_episode]
        
        plt.figure(figsize=(12, 6))
        plt.plot(best_equity_curve)
        plt.title(f"Mejor Curva de Equity (Episodio {best_episode+1})")
        plt.xlabel("Pasos")
        plt.ylabel("Balance")
        plt.grid(True)
        equity_curve_path = os.path.join(args.output, f"equity_curve_{timestamp}.png")
        plt.savefig(equity_curve_path)
        plt.close()
        
        # Save results
        print("Guardando resultados...", file=sys.stderr)
        results_path = os.path.join(args.output, f"neurevo_results_{timestamp}.json")
        
        with open(results_path, 'w') as f:
            json.dump({
                "training_results": results,
                "evaluation": {
                    "episodes": args.eval,
                    "rewards": rewards,
                    "average_reward": float(avg_reward),
                    "average_drawdown": float(avg_drawdown),
                    "best_episode": int(best_episode)
                },
                "config": custom_config,
                "training_time": training_time
            }, f, indent=4)
        
        print(f"Results saved to: {results_path}")
        print(f"Equity curve saved to: {equity_curve_path}")
        print("Resultados guardados", file=sys.stderr)
        print("=== Training Complete ===")
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

if __name__ == "__main__":
    main()