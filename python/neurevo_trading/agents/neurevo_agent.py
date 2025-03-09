import numpy as np
import torch
from typing import Dict, Tuple, Any, List

# Importar NeurEvo
import neurevo

# Importar componentes de neurevo_trading
from neurevo_trading.agents.trading_agent import TradingAgent


class NeurEvoTradingAgent(TradingAgent):
    """
    Agente de trading que utiliza el cerebro NeurEvo para tomar decisiones.
    Optimizado para maximizar el crecimiento estable de la cuenta con drawdowns mínimos.
    """
    
    def __init__(self, observation_space, action_space, config=None):
        """
        Inicializa el agente NeurEvoTradingAgent.
        
        Args:
            observation_space: Espacio de observación del entorno
            action_space: Espacio de acción del entorno
            config: Configuración para el cerebro NeurEvo
        """
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Configuración por defecto para NeurEvo, optimizada para trading de PnL
        self.default_config = {
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
        
        # Actualizar con la configuración proporcionada
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # Inicializar el cerebro NeurEvo
        self.brain = neurevo.create_brain(self.config)
        self.agent_id = None
        
        # Estado del agente
        self.is_trained = False
        self.current_state = None
        self.last_action = None
        self.training_info = {}
        
        # Historial para análisis
        self.action_history = []
        self.reward_history = []
        self.equity_history = []
        self.drawdown_history = []
    
    def initialize(self, environment, env_id="TradingEnv"):
        """
        Inicializa el agente con el entorno específico.
        
        Args:
            environment: Entorno de trading
            env_id: Identificador del entorno para NeurEvo
        """
        # Adaptar el entorno para NeurEvo
        def reset_fn():
            obs = environment.reset()
            self.current_state = obs
            return obs
        
        def step_fn(action):
            next_state, reward, done, info = environment.step(action)
            self.current_state = next_state
            
            # Registrar métricas de equity
            if 'balance' in info:
                self.equity_history.append(info['balance'])
            if 'max_drawdown' in info:
                self.drawdown_history.append(info['max_drawdown'])
                
            return next_state, reward, done, info
        
        # Registrar el entorno en NeurEvo
        self.brain.register_environment(
            env_id,
            create_custom_environment=True,
            reset_fn=reset_fn,
            step_fn=step_fn,
            observation_shape=self.observation_space.shape,
            action_size=self.action_space.shape[0] if hasattr(self.action_space, 'shape') else self.action_space.n
        )
        
        # Crear agente en NeurEvo
        self.agent_id = self.brain.create_for_environment(env_id)
        return self.agent_id
    
    def train(self, episodes=1000, verbose=True):
        """
        Entrena al agente usando el cerebro NeurEvo.
        
        Args:
            episodes: Número de episodios de entrenamiento
            verbose: Si es True, muestra progreso del entrenamiento
            
        Returns:
            Resultados del entrenamiento
        """
        if not self.agent_id:
            raise ValueError("Agent not initialized. Call initialize() first.")
        
        print(f"Training NeurEvo agent for {episodes} episodes...")
        
        # Entrenar usando cerebro NeurEvo
        results = self.brain.train(
            agent_id=self.agent_id,
            episodes=episodes,
            verbose=verbose
        )
        
        self.is_trained = True
        self.training_info = results
        
        # Calcular métricas finales
        if len(self.equity_history) > 0:
            final_equity = self.equity_history[-1]
            initial_equity = self.equity_history[0]
            profit_factor = final_equity / initial_equity if initial_equity > 0 else 0
            max_drawdown = max(self.drawdown_history) if len(self.drawdown_history) > 0 else 0
            
            print(f"Training completed. Final equity: {final_equity:.2f}, Profit factor: {profit_factor:.2f}, Max drawdown: {max_drawdown:.2%}")
        else:
            print(f"Training completed. Final reward: {results.get('final_reward', 'N/A')}")
            
        return results
    
    def act(self, observation, reward=0.0, done=False):
        """
        Determina la acción a tomar basada en la observación actual.
        
        Args:
            observation: Estado actual del entorno
            reward: Recompensa recibida (usado durante entrenamiento)
            done: Si el episodio ha terminado
            
        Returns:
            Acción a tomar
        """
        if not self.agent_id:
            raise ValueError("Agent not initialized. Call initialize() first.")
        
        # Usar cerebro NeurEvo para predecir la acción
        self.current_state = observation
        action = self.brain.predict(self.agent_id, observation)
        
        # Guardar para análisis
        self.last_action = action
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """
        Actualiza el agente con nueva experiencia.
        
        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Siguiente estado
            done: Si el episodio ha terminado
        """
        # Durante entrenamiento, el cerebro NeurEvo gestiona esto internamente
        pass
    
    def save(self, filepath):
        """
        Guarda el agente entrenado.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        if not self.is_trained:
            print("Warning: Saving an untrained agent")
        
        return self.brain.save(filepath)
    
    def load(self, filepath):
        """
        Carga un agente previamente entrenado.
        
        Args:
            filepath: Ruta del modelo guardado
        """
        success = self.brain.load(filepath)
        if success:
            self.is_trained = True
        return success
    
    def get_performance_metrics(self):
        """
        Obtiene métricas de rendimiento del agente.
        
        Returns:
            Dict: Métricas de rendimiento
        """
        if len(self.equity_history) == 0:
            return {"error": "No performance data available"}
            
        initial_equity = self.equity_history[0]
        final_equity = self.equity_history[-1]
        
        # Calcular métricas
        total_return = final_equity / initial_equity - 1
        max_drawdown = max(self.drawdown_history) if len(self.drawdown_history) > 0 else 0
        
        # Calcular calidad de la pendiente (qué tan cerca está de 45 grados)
        if len(self.equity_history) > 30:
            recent_equity = self.equity_history[-30:]
            x = np.arange(len(recent_equity))
            y = np.array(recent_equity)
            slope = np.polyfit(x, y, 1)[0]
            ideal_slope = initial_equity * 0.01  # ~1% por periodo
            slope_quality = 1.0 - min(1.0, abs(slope - ideal_slope) / ideal_slope)
        else:
            slope_quality = 0
        
        return {
            "initial_equity": initial_equity,
            "final_equity": final_equity,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": total_return / (max_drawdown + 0.001),  # Uso simple de Sharpe ratio
            "slope_quality": slope_quality
        }
    
    def analyze_current_state(self):
        """
        Analiza el estado actual para obtener información detallada.
        
        Returns:
            Información detallada sobre el estado actual y la decisión del agente
        """
        if not self.current_state is not None:
            return {"error": "No current state available"}
        
        if hasattr(self.brain, 'analyze_state'):
            return self.brain.analyze_state(self.agent_id, self.current_state)
        
        return {
            "state_summary": "Basic state information",
            "last_action": self.last_action,
            "performance": self.get_performance_metrics()
        }