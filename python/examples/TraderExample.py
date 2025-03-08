"""
Ejemplo de uso del sistema NeurEvo Trading
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurevo

from neurevo_trading.environment.data_processor import DataProcessor
from neurevo_trading.environment.trading_env import TradingEnvironment
from neurevo_trading.agents.neurevo_agent import NeurEvoTradingAgent
from neurevo_trading.utils.visualization import plot_trades

# Configurar el cerebro NeurEvo
config = {
    "hidden_layers": [128, 64, 32],
    "learning_rate": 0.0003,
    "batch_size": 32,
    "curiosity_weight": 0.2,
    "dynamic_network": True,
    "hebbian_learning": True
}

# Crear cerebro
brain = neurevo.create_brain(config)

# Cargar y preparar datos
print("Cargando datos...")
data_processor = DataProcessor()
data = data_processor.load_csv("data/forex_eurusd_daily.csv")  # Reemplazar con tu archivo
prepared_data = data_processor.prepare_data(data, add_features=True)

# Crear entorno
print("Creando entorno de trading...")
env = TradingEnvironment(
    data=prepared_data,
    window_size=50,
    initial_balance=10000,
    commission=0.0001
)

# Inicializar agente
print("Inicializando agente NeurEvo...")
agent = NeurEvoTradingAgent(env.observation_space, env.action_space, config)
agent.initialize(env)

# Entrenar agente
print("Entrenando agente...")
results = agent.train(episodes=100, verbose=True)

# Guardar modelo entrenado
print("Guardando modelo...")
os.makedirs("models", exist_ok=True)
agent.save("models/neurevo_trading_model.pt")

# Evaluar agente
print("\nEvaluando agente...")
observation = env.reset()
done = False
total_reward = 0

while not done:
    action = agent.act(observation)
    observation, reward, done, info = env.step(action)
    total_reward += reward

print(f"Evaluación completa - Reward: {total_reward}, Balance final: {info['balance']}")

# Visualizar trades
print("Generando visualización...")
trades = env.trade_history
plot_trades(prepared_data, trades, "NeurEvo Trading Performance")
plt.show()

print("Ejemplo completado!")