import numpy as np
from typing import Callable, Dict, Tuple, Any, List
import gym

# Importar NeurEvo
import neurevo

# Importar componentes de neurevo_trading
from neurevo_trading.environment.trading_env import TradingEnvironment

class NeurEvoEnvironmentAdapter:
    """
    Adaptador que permite usar entornos de trading con NeurEvo.
    """
    
    def __init__(self, trading_env: TradingEnvironment):
        """
        Inicializa el adaptador.
        
        Args:
            trading_env: Entorno de trading a adaptar
        """
        self.env = trading_env
        self.is_registered = False
        self.brain = None
        self.env_id = "TradingEnv"
    
    def register_with_brain(self, brain, env_id=None):
        """
        Registra el entorno adaptado con un cerebro NeurEvo.
        
        Args:
            brain: Cerebro NeurEvo
            env_id: Identificador para el entorno (opcional)
            
        Returns:
            Identificador del entorno registrado
        """
        if env_id:
            self.env_id = env_id
        
        self.brain = brain
        
        # Definir funciones para adaptación
        def reset_fn():
            return self.env.reset()
        
        def step_fn(action):
            return self.env.step(action)
        
        # Registrar el entorno
        self.brain.register_environment(
            self.env_id,
            create_custom_environment=True,
            reset_fn=reset_fn,
            step_fn=step_fn,
            observation_shape=self.env.observation_space.shape,
            action_size=self.env.action_space.shape[0] if hasattr(self.env.action_space, 'shape') else self.env.action_space.n
        )
        
        self.is_registered = True
        return self.env_id
    
    def create_agent(self, config=None):
        """
        Crea un agente NeurEvo para este entorno.
        
        Args:
            config: Configuración para el agente
            
        Returns:
            ID del agente creado
        """
        if not self.is_registered or not self.brain:
            raise ValueError("Environment not registered with a brain. Call register_with_brain() first.")
        
        agent_id = self.brain.create_for_environment(self.env_id, config)
        return agent_id
    
    def train_agent(self, agent_id, episodes=1000, verbose=True):
        """
        Entrena un agente en este entorno.
        
        Args:
            agent_id: ID del agente a entrenar
            episodes: Número de episodios
            verbose: Si es True, muestra progreso
            
        Returns:
            Resultados del entrenamiento
        """
        if not self.is_registered or not self.brain:
            raise ValueError("Environment not registered with a brain.")
        
        results = self.brain.train(
            agent_id=agent_id,
            episodes=episodes,
            verbose=verbose
        )
        
        return results
    
    def run_episode(self, agent_id, render=False):
        """
        Ejecuta un episodio completo con un agente entrenado.
        
        Args:
            agent_id: ID del agente a usar
            render: Si es True, renderiza el entorno
            
        Returns:
            Total de recompensa del episodio
        """
        if not self.is_registered or not self.brain:
            raise ValueError("Environment not registered with a brain.")
        
        state = self.env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = self.brain.predict(agent_id, state)
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            
            if render:
                self.env.render()
        
        return total_reward