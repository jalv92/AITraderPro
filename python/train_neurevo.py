import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Importar NeurEvo
import neurevo

# Importar componentes de neurevo_trading
from neurevo_trading.environment.trading_env import TradingEnvironment
from neurevo_trading.environment.data_processor import DataProcessor
from neurevo_trading.environment.neurevo_adapter import NeurEvoEnvironmentAdapter
from neurevo_trading.agents.neurevo_agent import NeurEvoTradingAgent
from neurevo_trading.utils.visualization import plot_trades, plot_pattern_distribution

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
    
    return parser.parse_args()

def load_config(config_path):
    if not config_path or not os.path.exists(config_path):
        # Default configuration
        return {
            "hidden_layers": [256, 128, 64],
            "learning_rate": 0.00025,
            "batch_size": 64,
            "curiosity_weight": 0.1,
            "dynamic_network": True
        }
    
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=== NeurEvo Trading Training ===")
    print(f"Loading data from: {args.data}")
    print(f"Training for {args.episodes} episodes")
    
    # Load and prepare data
    data_processor = DataProcessor()
    data = data_processor.load_csv(args.data)
    prepared_data = data_processor.prepare_data(data, add_features=True)
    
    print(f"Data loaded: {len(prepared_data)} rows with {len(prepared_data.columns)} features")
    
    # Create training environment
    env = TradingEnvironment(
        data=prepared_data,
        window_size=args.window,
        initial_balance=args.balance,
        commission=args.commission
    )
    
    # Create NeurEvo adapter
    adapter = NeurEvoEnvironmentAdapter(env)
    
    # Create NeurEvo brain
    brain = neurevo.create_brain(config)
    
    # Register environment with brain
    env_id = adapter.register_with_brain(brain, env_id="TradingEnv")
    
    # Create agent
    agent_id = adapter.create_agent()
    
    print("\n=== Starting Training ===")
    start_time = datetime.now()
    
    # Train agent
    results = adapter.train_agent(
        agent_id=agent_id,
        episodes=args.episodes,
        verbose=args.verbose
    )
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.output, f"neurevo_trading_model_{timestamp}.pt")
    brain.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Evaluate agent
    print("\n=== Evaluating Agent ===")
    rewards = []
    for i in range(args.eval):
        reward = adapter.run_episode(agent_id)
        rewards.append(reward)
        print(f"Episode {i+1}/{args.eval}: Reward = {reward:.2f}")
    
    avg_reward = np.mean(rewards)
    print(f"Average reward: {avg_reward:.2f}")
    
    # Save results
    results_path = os.path.join(args.output, f"neurevo_results_{timestamp}.json")
    
    with open(results_path, 'w') as f:
        json.dump({
            "training_results": results,
            "evaluation": {
                "episodes": args.eval,
                "rewards": rewards,
                "average_reward": float(avg_reward)
            },
            "config": config,
            "training_time": training_time
        }, f, indent=4)
    
    print(f"Results saved to: {results_path}")
    print("=== Training Complete ===")

if __name__ == "__main__":
    main()