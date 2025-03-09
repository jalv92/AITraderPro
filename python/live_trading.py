import argparse
import logging
import time
import threading
import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os
import sys

# Agregar directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurevo_trading.utils.socket_client import NinjaTraderClient
from neurevo_trading.agents.pattern_detector import PatternDetector
from neurevo_trading.environment.data_processor import DataProcessor
from neurevo_trading.utils.visualization import plot_trades, plot_pattern_detection
from neurevo_trading.utils.feature_engineering import create_features

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger("live_trading")

class LiveTradingSystem:
    """
    Sistema de trading en vivo que conecta con NinjaTrader y ejecuta estrategias
    basadas en la detección de patrones de precios.
    """
    
    def __init__(self, config_file=None, host="127.0.0.1", data_port=5000, order_port=5001,
                 model_path=None, window_size=50, min_confidence=0.7, min_risk_reward=1.5):
        """
        Inicializa el sistema de trading en vivo.
        
        Args:
            config_file: Ruta al archivo de configuración
            host: Host para conexión con NinjaTrader
            data_port: Puerto para datos
            order_port: Puerto para órdenes
            model_path: Ruta al modelo entrenado de detección de patrones
            window_size: Tamaño de la ventana para analizar patrones
            min_confidence: Confianza mínima para considerar un patrón válido
            min_risk_reward: Ratio mínimo de riesgo/recompensa
        """
        # Cargar configuración si existe
        self.config = self._load_config(config_file)
        
        # Usar valores de configuración si existen, de lo contrario usar los parámetros
        self.host = self.config.get("host", host)
        self.data_port = self.config.get("data_port", data_port)
        self.order_port = self.config.get("order_port", order_port)
        self.model_path = self.config.get("model_path", model_path)
        self.window_size = self.config.get("window_size", window_size)
        self.min_confidence = self.config.get("min_confidence", min_confidence)
        self.min_risk_reward = self.config.get("min_risk_reward", min_risk_reward)
        
        # Inicializar cliente de NinjaTrader
        self.nt_client = NinjaTraderClient(
            data_host=self.host,
            data_port=self.data_port,
            order_host=self.host,
            order_port=self.order_port,
            logger=logger
        )
        
        # Registrar callbacks
        self.nt_client.on_data_received = self._on_data_received
        self.nt_client.on_order_update = self._on_order_update
        self.nt_client.on_trade_executed = self._on_trade_executed
        self.nt_client.on_connection_status_changed = self._on_connection_status_changed
        
        # Inicializar detector de patrones
        self.pattern_detector = PatternDetector(
            input_channels=5,  # OHLCV por defecto
            window_size=self.window_size
        )
        
        # Cargar modelo neurevo si se especifica
        self.neurevo_model = None
        if self.model_path and os.path.exists(self.model_path):
            try:
                # Intentar cargar como modelo neurevo
                # Usamos pickle directamente para evitar problemas con torch.load y weights_only
                # en versiones recientes de PyTorch (2.6+)
                with open(self.model_path, 'rb') as f:
                    try:
                        import pickle
                        model_data = pickle.load(f)
                        self.neurevo_model = model_data.get('adapter')
                        self.neurevo_config = model_data.get('config')
                        logger.info(f"Modelo neurevo cargado desde {self.model_path}")
                        logger.info(f"Configuración del modelo: {self.neurevo_config}")
                    except Exception as pickle_err:
                        logger.error(f"Error al cargar con pickle: {pickle_err}")
                        # Intentar con torch.load como respaldo
                        f.seek(0)  # Volver al inicio del archivo
                        import torch
                        try:
                            # Usar explícitamente weights_only=False
                            model_data = torch.load(f, map_location='cpu', weights_only=False)
                            self.neurevo_model = model_data.get('adapter')
                            self.neurevo_config = model_data.get('config') 
                            logger.info(f"Modelo neurevo cargado con torch.load desde {self.model_path}")
                        except Exception as torch_err:
                            logger.error(f"Error al cargar con torch.load: {torch_err}")
                            raise
            except Exception as e:
                logger.error(f"Error al cargar modelo neurevo: {e}")
                logger.warning("Intentando cargar como modelo de detector de patrones...")
                try:
                    self.pattern_detector.load(self.model_path)
                    logger.info(f"Modelo de detección de patrones cargado desde {self.model_path}")
                except Exception as e2:
                    logger.error(f"También falló la carga como detector de patrones: {e2}")
                    logger.warning("No se ha podido cargar el modelo, usando detector básico")
        else:
            logger.warning("No se ha especificado un modelo válido, usando detector básico")
        
        # Inicializar procesador de datos
        self.data_processor = DataProcessor()
        
        # Buffers de datos
        self.price_data = pd.DataFrame()
        self.last_pattern = None
        self.last_pattern_time = None
        
        # Estado del sistema
        self.running = False
        self.trading_enabled = False
        self.current_position = 0  # 0 = sin posición, 1 = long, -1 = short
        
        # Métricas de rendimiento
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        # Thread para análisis continuo
        self.analysis_thread = None
    
    def _load_config(self, config_file):
        """Carga la configuración desde un archivo JSON."""
        if not config_file or not os.path.exists(config_file):
            return {}
        
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error al cargar configuración: {e}")
            return {}
    
    def start(self):
        """Inicia el sistema de trading."""
        if self.running:
            logger.warning("El sistema ya está en ejecución")
            return False
        
        # Conectar con NinjaTrader
        if not self.nt_client.connect():
            logger.error("No se pudo conectar con NinjaTrader")
            return False
        
        self.running = True
        
        # Iniciar thread de análisis
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        
        logger.info("Sistema de trading iniciado")
        return True
    
    def stop(self):
        """Detiene el sistema de trading."""
        if not self.running:
            logger.warning("El sistema no está en ejecución")
            return False
        
        # Desactivar trading
        self.trading_enabled = False
        
        # Cerrar posiciones abiertas
        if self.current_position != 0:
            self._close_position("System shutdown")
        
        # Detener sistema
        self.running = False
        
        # Desconectar de NinjaTrader
        self.nt_client.disconnect()
        
        # Esperar a que termine el thread de análisis
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(5.0)
        
        logger.info("Sistema de trading detenido")
        return True
    
    def enable_trading(self, enabled=True):
        """Activa o desactiva el trading automático."""
        self.trading_enabled = enabled
        logger.info(f"Trading automático {'activado' if enabled else 'desactivado'}")
    
    def _on_data_received(self, message):
        """Callback cuando se reciben datos de NinjaTrader."""
        # Implementar procesamiento de datos en tiempo real
        # Este método debería convertir los datos a DataFrame y actualizar self.price_data
        pass
    
    def _on_order_update(self, update):
        """Callback cuando se actualiza el estado de una orden."""
        logger.info(f"Actualización de orden: {update}")
        
        if update.get("type") == "confirmation":
            # Actualizar posición actual
            self.current_position = update.get("signal", 0)
    
    def _on_trade_executed(self, trade_info):
        """Callback cuando se ejecuta un trade."""
        logger.info(f"Trade ejecutado: {trade_info}")
        
        # Actualizar métricas
        self.total_trades += 1
        
        if trade_info.get("pnl", 0) > 0:
            self.winning_trades += 1
        
        self.total_pnl += trade_info.get("pnl", 0)
        
        # Registrar en el historial
        self._log_trade(trade_info)
    
    def _on_connection_status_changed(self, connected):
        """Callback cuando cambia el estado de la conexión."""
        logger.info(f"Estado de conexión: {'Conectado' if connected else 'Desconectado'}")
        
        # Si se desconecta, desactivar trading
        if not connected:
            self.trading_enabled = False
    
    def _analysis_loop(self):
        """Loop principal de análisis de patrones y toma de decisiones."""
        loop_count = 0
        
        while self.running:
            try:
                # Esperar acumulación de suficientes datos
                if len(self.price_data) < self.window_size:
                    time.sleep(1)
                    continue
                
                # Analizar cada 5 segundos (o después de cada nueva barra)
                if loop_count % 5 == 0:
                    self._analyze_market()
                
                loop_count += 1
                time.sleep(1)
            
            except Exception as e:
                logger.error(f"Error en loop de análisis: {e}")
                time.sleep(5)  # Esperar antes de reintentar
    
    def _analyze_market(self):
        """Analiza el mercado en busca de patrones y oportunidades de trading."""
        if len(self.price_data) < self.window_size:
            return
        
        # Obtener datos recientes
        recent_data = self.price_data.tail(self.window_size).copy()
        
        # Agregar características para el análisis
        enhanced_data = create_features(recent_data)
        
        # Usar el modelo neurevo si está disponible
        if self.neurevo_model is not None:
            try:
                # Usar el método run_episode que sabemos que funciona
                reward = self.neurevo_model.run_episode(None)
                logger.info(f"Ejecución del modelo neurevo: Reward={reward}")
                
                # Convertir el reward en una señal de trading
                # Nota: Esto es una lógica simple y debería mejorarse
                if reward > 50.0:  # Umbral para señal positiva
                    signal = 1  # Long
                    confidence = min(0.95, 0.7 + (reward - 50.0) / 100)
                elif reward < 50.0:  # Umbral para señal negativa
                    signal = -1  # Short
                    confidence = min(0.95, 0.7 + (50.0 - reward) / 100)
                else:
                    signal = 0  # Neutral
                    confidence = 0.5
                
                pattern_type = "RL_PREDICTION"
                risk_reward = 1.5  # Valor por defecto
                
                logger.info(f"Predicción del modelo neurevo: Signal={signal}, Conf={confidence:.2f}, RR={risk_reward:.2f}")
                
            except Exception as e:
                logger.error(f"Error al usar modelo neurevo: {e}")
                # Si falla, recurrir al detector de patrones
                pattern_info = self.pattern_detector.detect(enhanced_data)
                signal = 1 if pattern_info['pattern_name'] in ['DOUBLE_BOTTOM', 'INV_HEAD_AND_SHOULDERS'] else \
                     -1 if pattern_info['pattern_name'] in ['DOUBLE_TOP', 'HEAD_AND_SHOULDERS'] else 0
                confidence = pattern_info['confidence']
                pattern_type = pattern_info['pattern_name']
                risk_reward = 1.0  # Valor por defecto
                
                if 'entry_price' in pattern_info and 'stop_loss' in pattern_info and 'take_profit' in pattern_info:
                    if pattern_info['stop_loss'] > 0 and pattern_info['take_profit'] > 0:
                        risk = abs(pattern_info['entry_price'] - pattern_info['stop_loss'])
                        reward = abs(pattern_info['entry_price'] - pattern_info['take_profit'])
                        risk_reward = reward / risk if risk > 0 else 1.0
        else:
            # Usar detector de patrones si no hay modelo neurevo
            pattern_info = self.pattern_detector.detect(enhanced_data)
            signal = 1 if pattern_info['pattern_name'] in ['DOUBLE_BOTTOM', 'INV_HEAD_AND_SHOULDERS'] else \
                 -1 if pattern_info['pattern_name'] in ['DOUBLE_TOP', 'HEAD_AND_SHOULDERS'] else 0
            confidence = pattern_info['confidence']
            pattern_type = pattern_info['pattern_name']
            risk_reward = 1.0  # Valor por defecto
            
            if 'entry_price' in pattern_info and 'stop_loss' in pattern_info and 'take_profit' in pattern_info:
                if pattern_info['stop_loss'] > 0 and pattern_info['take_profit'] > 0:
                    risk = abs(pattern_info['entry_price'] - pattern_info['stop_loss'])
                    reward = abs(pattern_info['entry_price'] - pattern_info['take_profit'])
                    risk_reward = reward / risk if risk > 0 else 1.0
        
        # Solo procesar si la señal no es neutral, la confianza y el ratio riesgo/recompensa son suficientes
        if signal != 0 and confidence >= self.min_confidence and risk_reward >= self.min_risk_reward:
            # Evitar repetir señales en poco tiempo
            current_time = time.time()
            if self.last_pattern_time is None or (current_time - self.last_pattern_time) > 60:  # 1 minuto mínimo
                self._process_trading_signal(signal, confidence, risk_reward, pattern_type)
                self.last_pattern_time = current_time
                self.last_pattern = pattern_type
        
    def _process_trading_signal(self, signal, confidence, risk_reward, pattern_type):
        """Procesa una señal de trading basada en el patrón detectado."""
        # Si no está habilitado el trading, solo registrar
        if not self.trading_enabled:
            logger.info(f"Señal detectada pero trading deshabilitado: {signal} ({pattern_type}), "
                        f"Confianza: {confidence:.2f}, R/R: {risk_reward:.2f}")
            return
        
        # Si ya tenemos una posición en la misma dirección, ignorar
        if signal == self.current_position:
            logger.info(f"Señal ignorada: ya tenemos posición en la misma dirección ({signal})")
            return
        
        # Si tenemos una posición contraria, cerrarla primero
        if self.current_position != 0 and self.current_position != signal:
            self._close_position("Cambio de dirección")
        
        # Calcular tamaño de posición basado en confianza
        position_size = min(0.9, confidence)
        
        # Valores por defecto para stop loss y take profit (en ticks)
        # Estos valores deberían ajustarse según el instrumento y timeframe
        stop_loss = 20
        take_profit = stop_loss * risk_reward
        
        # Enviar señal de trading
        success = self.nt_client.send_trading_signal(
            signal=signal,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pattern_type=pattern_type,
            confidence=confidence,
            risk_reward=risk_reward
        )
        
        if success:
            logger.info(f"Señal enviada: {signal} con tamaño {position_size:.2f}, "
                      f"SL={stop_loss:.0f} ticks, TP={take_profit:.0f} ticks, R/R={risk_reward:.2f}")
        else:
            logger.error(f"Error al enviar señal de trading")
    
    def _close_position(self, reason):
        """Cierra la posición actual."""
        if self.current_position == 0:
            logger.info("No hay posición abierta para cerrar")
            return
        
        # Enviar señal de cierre (0)
        self.nt_client.send_trading_signal(
            signal=0,
            position_size=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            pattern_type="MANUAL_CLOSE",
            confidence=1.0,
            risk_reward=1.0
        )
        
        logger.info(f"Posición cerrada: {reason}")
    
    def _log_trade(self, trade_info):
        """Registra información del trade en archivo."""
        try:
            # Crear directorio de logs si no existe
            log_dir = "trade_logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Nombre del archivo de log
            log_file = os.path.join(log_dir, f"trades_{datetime.now().strftime('%Y%m%d')}.csv")
            
            # Convertir trade_info a DataFrame
            trade_df = pd.DataFrame([trade_info])
            
            # Verificar si existe el archivo para agregar encabezado
            file_exists = os.path.isfile(log_file)
            
            # Guardar en CSV
            trade_df.to_csv(log_file, mode='a', header=not file_exists, index=False)
        
        except Exception as e:
            logger.error(f"Error al registrar trade: {e}")
    
    def get_performance_metrics(self):
        """Obtiene métricas de rendimiento del sistema."""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl
        }

def main():
    """Función principal para ejecutar el sistema de trading en vivo."""
    parser = argparse.ArgumentParser(description="AITraderPro - Sistema de trading en vivo")
    
    parser.add_argument("--config", type=str, help="Ruta al archivo de configuración")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host para conexión con NinjaTrader")
    parser.add_argument("--data-port", type=int, default=5000, help="Puerto para datos")
    parser.add_argument("--order-port", type=int, default=5001, help="Puerto para órdenes")
    parser.add_argument("--model", type=str, help="Ruta al modelo entrenado")
    parser.add_argument("--window", type=int, default=50, help="Tamaño de ventana para análisis")
    parser.add_argument("--confidence", type=float, default=0.7, help="Confianza mínima para señales")
    parser.add_argument("--risk-reward", type=float, default=1.5, help="Ratio mínimo de riesgo/recompensa")
    parser.add_argument("--enable-trading", action="store_true", help="Activar trading automático al inicio")
    
    args = parser.parse_args()
    
    # Inicializar sistema
    system = LiveTradingSystem(
        config_file=args.config,
        host=args.host,
        data_port=args.data_port,
        order_port=args.order_port,
        model_path=args.model,
        window_size=args.window,
        min_confidence=args.confidence,
        min_risk_reward=args.risk_reward
    )
    
    # Iniciar sistema
    if system.start():
        # Activar trading si se especifica
        if args.enable_trading:
            system.enable_trading(True)
        
        try:
            # Mantener el programa en ejecución
            print("Sistema iniciado. Presiona Ctrl+C para detener.")
            while True:
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\nDetención solicitada por el usuario.")
        finally:
            # Asegurar detención correcta
            system.stop()
            
            # Mostrar métricas finales
            metrics = system.get_performance_metrics()
            print("\nMétricas de rendimiento:")
            print(f"Total de trades: {metrics['total_trades']}")
            print(f"Trades ganadores: {metrics['winning_trades']}")
            print(f"Win rate: {metrics['win_rate']:.2f}%")
            print(f"PnL total: {metrics['total_pnl']:.2f}")
    else:
        print("No se pudo iniciar el sistema.")

if __name__ == "__main__":
    main()