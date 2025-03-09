from abc import ABC, abstractmethod

class TradingAgent(ABC):
    """
    Clase base abstracta para agentes de trading.
    Define la interfaz para los agentes de trading.
    """
    
    def __init__(self):
        """
        Inicializa el agente de trading.
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    def save(self, filepath):
        """
        Guarda el agente entrenado.
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        raise NotImplementedError("save method not implemented")
    
    def load(self, filepath):
        """
        Carga un agente previamente entrenado.
        
        Args:
            filepath: Ruta del modelo guardado
        """
        raise NotImplementedError("load method not implemented")
