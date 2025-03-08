import argparse
import logging
import time
import threading
import json
import pandas as pd
import numpy as np
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
        
        # Cargar modelo si se especifica
        if self.model_path and os.path.exists(self.model_path):
            self.pattern_detector.load(self.model_path)
            logger.info(f"Modelo de detección de patrones cargado desde {self.model_path}")
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
        
        # Detectar patrones
        pattern_info = self.pattern_detector.detect(enhanced_data)
        
        pattern_name = pattern_info.get("pattern_name", "NO_PATTERN")
        confidence = pattern_info.get("confidence", 0.0)
        
        # Si se detecta un patrón con confianza suficiente
        if pattern_name != "NO_PATTERN" and confidence >= self.min_confidence:
            # Si es un patrón nuevo o ha pasado suficiente tiempo desde el último
            current_time = time.time()
            if (self.last_pattern != pattern_name or 
                not self.last_pattern_time or 
                current_time - self.last_pattern_time > 300):  # 5 minutos
                
                logger.info(f"Patrón detectado: {pattern_name} con confianza {confidence:.2f}")
                
                # Actualizar registro de último patrón
                self.last_pattern = pattern_name
                self.last_pattern_time = current_time
                
                # Si el trading está activado, generar señal
                if self.trading_enabled:
                    self._generate_trading_signal(pattern_info, enhanced_data)
    
    def _generate_trading_signal(self, pattern_info, data):
        """Genera una señal de trading basada en el patrón detectado."""
        pattern_name = pattern_info.get("pattern_name")
        confidence = pattern_info.get("confidence", 0.0)
        entry_price = pattern_info.get("entry_price", data["Close"].iloc[-1])
        stop_loss = pattern_info.get("stop_loss", 0.0)
        take_profit = pattern_info.get("take_profit", 0.0)
        
        # Determinar dirección de la señal basada en el patrón
        signal = 0
        if pattern_name in ["DOUBLE_BOTTOM", "INV_HEAD_AND_SHOULDERS"]:
            signal = 1  # Long
        elif pattern_name in ["DOUBLE_TOP", "HEAD_AND_SHOULDERS"]:
            signal = -1  # Short
        
        # Si no hay señal clara o es la misma posición actual, no hacer nada
        if signal == 0 or signal == self.current_position:
            return
        
        # Calcular ratio riesgo/recompensa
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward = reward / risk if risk > 0 else 0
        
        # Verificar ratio mínimo de riesgo/recompensa
        if risk_reward < self.min_risk_reward:
            logger.info(f"Señal ignorada: ratio R/R {risk_reward:.2f} inferior al mínimo {self.min_risk_reward}")
            return
        
        # Calcular tamaño de posición basado en confianza
        position_size = min(0.8, confidence)
        
        # Enviar señal de trading
        self.nt_client.send_trading_signal(
            signal=signal,
            position_size=position_size,
            stop_loss=abs(entry_price - stop_loss) / data["ATR"].iloc[-1] if "ATR" in data else 20,
            take_profit=abs(take_profit - entry_price) / data["ATR"].iloc[-1] if "ATR" in data else 30,
            pattern_type=pattern_name,
            confidence=confidence,
            risk_reward=risk_reward
        )
        
        logger.info(f"Señal enviada: {signal} con tamaño {position_size:.2f}, "
                   f"SL={stop_loss:.2f}, TP={take_profit:.2f}, R/R={risk_reward:.2f}")
    
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