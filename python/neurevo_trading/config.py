import os
import json
import logging
from typing import Dict, Any, Optional

class Config:
    """
    Clase para manejar la configuración del sistema AITraderPro.
    """
    
    def __init__(self, config_file: Optional[str] = None, default_config: Optional[Dict] = None):
        """
        Inicializa la configuración.
        
        Args:
            config_file: Ruta al archivo de configuración
            default_config: Configuración por defecto
        """
        self.logger = self._setup_logger()
        self.config_file = config_file
        
        # Configuración por defecto
        self.default_config = default_config or {
            # Parámetros de la aplicación
            "app": {
                "name": "AITraderPro",
                "version": "0.1.0",
                "log_level": "INFO",
                "log_dir": "logs"
            },
            
            # Parámetros de NinjaTrader
            "ninjatrader": {
                "host": "127.0.0.1",
                "data_port": 5000,
                "order_port": 5001,
                "reconnect_attempts": 5,
                "reconnect_delay": 3
            },
            
            # Parámetros de trading
            "trading": {
                "initial_capital": 10000,
                "risk_per_trade": 0.02,
                "max_positions": 1,
                "commission": 0.0,
                "slippage": 0.0
            },
            
            # Parámetros de detección de patrones
            "pattern_detection": {
                "min_confidence": 0.7,
                "min_risk_reward": 1.5,
                "window_size": 50,
                "model_path": "models/pattern_detector.pt"
            },
            
            # Directorios
            "directories": {
                "data": "data",
                "models": "models",
                "results": "results",
                "logs": "logs"
            }
        }
        
        # Cargar configuración
        self.config = self.default_config.copy()
        
        if config_file:
            try:
                self.load_config(config_file)
            except Exception as e:
                self.logger.error(f"Error al cargar configuración: {e}")
                self.logger.info("Usando configuración por defecto")
    
    def _setup_logger(self) -> logging.Logger:
        """Configura y devuelve un logger."""
        logger = logging.getLogger("Config")
        logger.setLevel(logging.INFO)
        
        # Crear manejador de consola si no existe
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_config(self, config_file: str) -> None:
        """
        Carga la configuración desde un archivo.
        
        Args:
            config_file: Ruta al archivo de configuración
        """
        try:
            if not os.path.exists(config_file):
                self.logger.warning(f"Archivo de configuración no encontrado: {config_file}")
                return
            
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Actualizar configuración
            self._update_config(self.config, config_data)
            
            self.config_file = config_file
            self.logger.info(f"Configuración cargada desde {config_file}")
        
        except Exception as e:
            self.logger.error(f"Error al cargar configuración desde {config_file}: {e}")
            raise
    
    def _update_config(self, target: Dict, source: Dict) -> None:
        """
        Actualiza la configuración de forma recursiva.
        
        Args:
            target: Diccionario destino
            source: Diccionario fuente
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_config(target[key], value)
            else:
                target[key] = value
    
    def save_config(self, config_file: Optional[str] = None) -> None:
        """
        Guarda la configuración en un archivo.
        
        Args:
            config_file: Ruta al archivo de configuración
        """
        filepath = config_file or self.config_file
        
        if not filepath:
            self.logger.error("No se ha especificado archivo de configuración")
            return
        
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            self.logger.info(f"Configuración guardada en {filepath}")
        
        except Exception as e:
            self.logger.error(f"Error al guardar configuración en {filepath}: {e}")
            raise
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración.
        
        Args:
            section: Sección de configuración
            key: Clave de configuración
            default: Valor por defecto
            
        Returns:
            Valor de configuración o valor por defecto
        """
        try:
            return self.config[section][key]
        except (KeyError, TypeError):
            return default
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Establece un valor de configuración.
        
        Args:
            section: Sección de configuración
            key: Clave de configuración
            value: Valor a establecer
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def get_all(self) -> Dict:
        """
        Obtiene toda la configuración.
        
        Returns:
            Diccionario con toda la configuración
        """
        return self.config.copy()
    
    def create_default_config(self, filepath: str) -> None:
        """
        Crea un archivo de configuración por defecto.
        
        Args:
            filepath: Ruta al archivo de configuración
        """
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(self.default_config, f, indent=4)
            
            self.logger.info(f"Configuración por defecto creada en {filepath}")
        
        except Exception as e:
            self.logger.error(f"Error al crear configuración por defecto en {filepath}: {e}")
            raise