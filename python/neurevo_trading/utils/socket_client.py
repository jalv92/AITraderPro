import socket
import time
import threading
import queue
import logging
import json
import select
from typing import Dict, List, Optional, Tuple, Union, Callable

class NinjaTraderClient:
    """
    Cliente para comunicación bidireccional con NinjaTrader a través de TCP.
    Maneja dos conexiones separadas:
    - Data Connection: Recibe datos de mercado desde NinjaTrader
    - Order Connection: Envía señales de trading a NinjaTrader
    """
    
    def __init__(self, data_host: str = "127.0.0.1", data_port: int = 5000,
                 order_host: str = "127.0.0.1", order_port: int = 5001,
                 reconnect_attempts: int = 5, reconnect_delay: int = 3,
                 logger: Optional[logging.Logger] = None):
        """
        Inicializa el cliente NinjaTrader.
        
        Args:
            data_host: Host para la conexión de datos
            data_port: Puerto para la conexión de datos
            order_host: Host para la conexión de órdenes
            order_port: Puerto para la conexión de órdenes
            reconnect_attempts: Número de intentos de reconexión
            reconnect_delay: Retraso entre intentos de reconexión (segundos)
            logger: Logger opcional
        """
        # Configuración de conexiones
        self.data_host = data_host
        self.data_port = data_port
        self.order_host = order_host
        self.order_port = order_port
        
        # Parámetros de reconexión
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        # Inicialización de sockets
        self.data_socket = None
        self.order_socket = None
        
        # Estado de conexión
        self.connected_data = False
        self.connected_order = False
        
        # Colas para mensajes
        self.data_queue = queue.Queue()
        self.order_queue = queue.Queue()
        self.data_send_queue = queue.Queue()
        self.order_send_queue = queue.Queue()
        
        # Configuración de logging
        self.logger = logger or self._setup_default_logger()
        
        # Flags de control
        self._running = False
        self._data_thread = None
        self._order_thread = None
        self._process_thread = None
        
        # Callbacks
        self.on_data_received = None
        self.on_order_update = None
        self.on_connection_status_changed = None
        self.on_trade_executed = None
        
        # Información de mercado y trading
        self.instrument_info = {}
        self.current_position = 0  # 0 = sin posición, 1 = long, -1 = short
        self.last_signal_time = None
        self.trades_history = []
    
    def _setup_default_logger(self) -> logging.Logger:
        """Configura un logger por defecto si no se proporciona ninguno."""
        logger = logging.getLogger("NinjaTraderClient")
        logger.setLevel(logging.INFO)
        
        # Crear manejador de consola si no existe
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def connect(self) -> bool:
        """
        Establece conexiones con NinjaTrader.
        
        Returns:
            bool: True si ambas conexiones se establecieron correctamente.
        """
        if self._running:
            self.logger.warning("Client already running, disconnect first")
            return False
        
        # Iniciar hilos
        self._running = True
        
        # Iniciar hilos de conexión
        self._data_thread = threading.Thread(target=self._data_connection_thread, daemon=True)
        self._order_thread = threading.Thread(target=self._order_connection_thread, daemon=True)
        self._process_thread = threading.Thread(target=self._process_thread_func, daemon=True)
        
        self._data_thread.start()
        self._order_thread.start()
        self._process_thread.start()
        
        # Esperar a que ambas conexiones se establezcan (con timeout)
        timeout = time.time() + 30  # 30 segundos timeout
        while time.time() < timeout and self._running:
            if self.connected_data and self.connected_order:
                self.logger.info("Successfully connected to NinjaTrader")
                return True
            time.sleep(0.1)
        
        # Si llegamos aquí, no se establecieron ambas conexiones
        if not (self.connected_data and self.connected_order):
            self.logger.error("Failed to connect to NinjaTrader within timeout")
            self.disconnect()
            return False
        
        return True
    
    def disconnect(self) -> None:
        """Cierra las conexiones con NinjaTrader."""
        self._running = False
        
        # Cerrar sockets
        if self.data_socket:
            try:
                self.data_socket.close()
            except Exception as e:
                self.logger.error(f"Error closing data socket: {e}")
            finally:
                self.data_socket = None
        
        if self.order_socket:
            try:
                self.order_socket.close()
            except Exception as e:
                self.logger.error(f"Error closing order socket: {e}")
            finally:
                self.order_socket = None
        
        # Actualizar estados
        self.connected_data = False
        self.connected_order = False
        
        # Esperar a que terminen los hilos
        if self._data_thread and self._data_thread.is_alive():
            self._data_thread.join(2.0)
        
        if self._order_thread and self._order_thread.is_alive():
            self._order_thread.join(2.0)
        
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(2.0)
        
        self.logger.info("Disconnected from NinjaTrader")
        
        # Notificar cambio de estado si hay callback
        if self.on_connection_status_changed:
            self.on_connection_status_changed(False)
    
    def _connect_socket(self, host: str, port: int) -> Tuple[Optional[socket.socket], bool]:
        """
        Conecta a un socket TCP específico.
        
        Args:
            host: Dirección del host
            port: Número de puerto
            
        Returns:
            Tuple[socket, bool]: Socket conectado y estado de éxito
        """
        sock = None
        try:
            # Crear un nuevo socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)  # Timeout de 10 segundos para conexión
            
            # Intentar conectar
            self.logger.info(f"Connecting to {host}:{port}...")
            sock.connect((host, port))
            
            # Configurar socket para operación no bloqueante
            sock.setblocking(0)
            self.logger.info(f"Connected to {host}:{port}")
            return sock, True
        except Exception as e:
            self.logger.error(f"Error connecting to {host}:{port}: {e}")
            if sock:
                try:
                    sock.close()
                except:
                    pass
            return None, False
    
    def _data_connection_thread(self) -> None:
        """Hilo para gestionar la conexión de datos."""
        attempts = 0
        buffer = ""
        
        while self._running and attempts < self.reconnect_attempts:
            if not self.connected_data:
                self.logger.info(f"Connecting to data server (attempt {attempts+1}/{self.reconnect_attempts})...")
                self.data_socket, self.connected_data = self._connect_socket(self.data_host, self.data_port)
                
                if self.connected_data:
                    # Notificar cambio de estado si hay callback
                    if self.on_connection_status_changed:
                        self.on_connection_status_changed(
                            self.connected_data and self.connected_order
                        )
                    attempts = 0  # Resetear intentos tras conexión exitosa
                else:
                    attempts += 1
                    if attempts < self.reconnect_attempts:
                        time.sleep(self.reconnect_delay)
                    continue
            
            try:
                # Recibir datos
                ready_to_read = select.select([self.data_socket], [], [], 0.1)[0]
                if ready_to_read:
                    data = self.data_socket.recv(4096)
                    if data:
                        buffer += data.decode('utf-8')
                        
                        # Procesar mensajes completos
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            if line:  # Ignorar líneas vacías
                                self.data_queue.put(line)
                    else:
                        # Conexión cerrada por el servidor
                        self.logger.warning("Data connection closed by server")
                        self.connected_data = False
                        if self.data_socket:
                            self.data_socket.close()
                            self.data_socket = None
                        continue
                
                # Enviar datos pendientes
                if not self.data_send_queue.empty():
                    try:
                        ready_to_write = select.select([], [self.data_socket], [], 0.1)[1]
                        if ready_to_write:
                            message = self.data_send_queue.get_nowait()
                            if not message.endswith('\n'):
                                message += '\n'
                            self.data_socket.sendall(message.encode('utf-8'))
                    except queue.Empty:
                        pass
                    except Exception as e:
                        self.logger.error(f"Error sending data: {e}")
                        self.connected_data = False
                        if self.data_socket:
                            self.data_socket.close()
                            self.data_socket = None
                        continue
                
                # Enviar heartbeat periódico
                if time.time() % 15 < 0.1:  # Aproximadamente cada 15 segundos
                    try:
                        self.data_socket.sendall("PING\n".encode('utf-8'))
                    except Exception as e:
                        self.logger.error(f"Error sending heartbeat: {e}")
                        self.connected_data = False
                        if self.data_socket:
                            self.data_socket.close()
                            self.data_socket = None
            
            except Exception as e:
                self.logger.error(f"Error in data connection: {e}")
                self.connected_data = False
                if self.data_socket:
                    try:
                        self.data_socket.close()
                    except:
                        pass
                    self.data_socket = None
        
        self.logger.info("Data connection thread terminated")
    
    def _order_connection_thread(self) -> None:
        """Hilo para gestionar la conexión de órdenes."""
        attempts = 0
        buffer = ""
        
        while self._running and attempts < self.reconnect_attempts:
            if not self.connected_order:
                self.logger.info(f"Connecting to order server (attempt {attempts+1}/{self.reconnect_attempts})...")
                self.order_socket, self.connected_order = self._connect_socket(self.order_host, self.order_port)
                
                if self.connected_order:
                    # Notificar cambio de estado si hay callback
                    if self.on_connection_status_changed:
                        self.on_connection_status_changed(
                            self.connected_data and self.connected_order
                        )
                    attempts = 0  # Resetear intentos tras conexión exitosa
                else:
                    attempts += 1
                    if attempts < self.reconnect_attempts:
                        time.sleep(self.reconnect_delay)
                    continue
            
            try:
                # Recibir datos
                ready_to_read = select.select([self.order_socket], [], [], 0.1)[0]
                if ready_to_read:
                    data = self.order_socket.recv(4096)
                    if data:
                        buffer += data.decode('utf-8')
                        
                        # Procesar mensajes completos
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            if line:  # Ignorar líneas vacías
                                self.order_queue.put(line)
                    else:
                        # Conexión cerrada por el servidor
                        self.logger.warning("Order connection closed by server")
                        self.connected_order = False
                        if self.order_socket:
                            self.order_socket.close()
                            self.order_socket = None
                        continue
                
                # Enviar órdenes pendientes
                if not self.order_send_queue.empty():
                    try:
                        ready_to_write = select.select([], [self.order_socket], [], 0.1)[1]
                        if ready_to_write:
                            message = self.order_send_queue.get_nowait()
                            if not message.endswith('\n'):
                                message += '\n'
                            self.order_socket.sendall(message.encode('utf-8'))
                    except queue.Empty:
                        pass
                    except Exception as e:
                        self.logger.error(f"Error sending order: {e}")
                        self.connected_order = False
                        if self.order_socket:
                            self.order_socket.close()
                            self.order_socket = None
                        continue
                
                # Enviar heartbeat periódico
                if time.time() % 15 < 0.1:  # Aproximadamente cada 15 segundos
                    try:
                        self.order_socket.sendall("PING\n".encode('utf-8'))
                    except Exception as e:
                        self.logger.error(f"Error sending heartbeat: {e}")
                        self.connected_order = False
                        if self.order_socket:
                            self.order_socket.close()
                            self.order_socket = None
            
            except Exception as e:
                self.logger.error(f"Error in order connection: {e}")
                self.connected_order = False
                if self.order_socket:
                    try:
                        self.order_socket.close()
                    except:
                        pass
                    self.order_socket = None
        
        self.logger.info("Order connection thread terminated")
    
    def _process_thread_func(self) -> None:
        """Hilo para procesar mensajes recibidos."""
        while self._running:
            # Procesar mensajes de datos
            try:
                while not self.data_queue.empty():
                    message = self.data_queue.get_nowait()
                    self._process_data_message(message)
            except queue.Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error processing data message: {e}")
            
            # Procesar mensajes de órdenes
            try:
                while not self.order_queue.empty():
                    message = self.order_queue.get_nowait()
                    self._process_order_message(message)
            except queue.Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error processing order message: {e}")
            
            # Dormir para reducir uso de CPU
            time.sleep(0.01)
        
        self.logger.info("Process thread terminated")
    
    def _process_data_message(self, message: str) -> None:
        """
        Procesa mensajes recibidos en la conexión de datos.
        
        Args:
            message: Mensaje recibido
        """
        # Ignorar mensajes de heartbeat
        if message in ["PING", "PONG"]:
            return
        
        if message.startswith("SERVER_READY"):
            self.logger.info("Data server ready")
            # Enviar información del cliente
            self.send_data_message(f"CONNECT:PythonClient_v1.0")
        
        elif message.startswith("SERVER_INFO:"):
            # Procesar información del servidor
            parts = message.split(':', 1)[1].split(':')
            if len(parts) >= 2:
                server_type, instrument = parts[0], parts[1]
                self.instrument_info = {
                    "server_type": server_type,
                    "instrument": instrument
                }
                self.logger.info(f"Connected to {server_type} server, instrument: {instrument}")
        
        # Pasar el mensaje al callback si está definido
        if self.on_data_received:
            self.on_data_received(message)
    
    def _process_order_message(self, message: str) -> None:
        """
        Procesa mensajes recibidos en la conexión de órdenes.
        
        Args:
            message: Mensaje recibido
        """
        # Ignorar mensajes de heartbeat
        if message in ["PING", "PONG"]:
            return
        
        if message.startswith("ORDER_SERVER_READY"):
            self.logger.info("Order server ready")
        
        elif message.startswith("ORDER_CONFIRMED:"):
            # Procesar confirmación de orden
            try:
                params = message.split(':', 1)[1].split(',')
                if len(params) >= 4:
                    signal = int(params[0])
                    position_size = float(params[1])
                    pattern_type = params[2]
                    confidence = float(params[3])
                    
                    self.logger.info(f"Order confirmed: Signal={signal}, Size={position_size}, "
                                     f"Pattern={pattern_type}, Confidence={confidence}")
                    
                    # Actualizar estado de posición
                    self.current_position = signal
                    self.last_signal_time = time.time()
                    
                    # Notificar actualización de orden
                    if self.on_order_update:
                        self.on_order_update({
                            "type": "confirmation",
                            "signal": signal,
                            "size": position_size,
                            "pattern": pattern_type,
                            "confidence": confidence
                        })
            except Exception as e:
                self.logger.error(f"Error processing order confirmation: {e}")
        
        elif message.startswith("ERROR:"):
            # Procesar error
            error_message = message.split(':', 1)[1]
            self.logger.error(f"Order error: {error_message}")
            
            # Notificar error
            if self.on_order_update:
                self.on_order_update({
                    "type": "error",
                    "message": error_message
                })
        
        elif message.startswith("TRADE_EXECUTED:"):
            # Procesar ejecución de trade
            try:
                params = message.split(':', 1)[1].split(',')
                if len(params) >= 8:
                    action = params[0]
                    entry_price = float(params[1]) if params[1] else 0
                    exit_price = float(params[2]) if params[2] else 0
                    pnl = float(params[3]) if params[3] else 0
                    quantity = int(params[4]) if params[4] else 0
                    pattern = params[5]
                    confidence = float(params[6]) if params[6] else 0
                    risk_reward = float(params[7]) if params[7] else 0
                    
                    trade_info = {
                        "time": time.time(),
                        "action": action,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "quantity": quantity,
                        "pattern": pattern,
                        "confidence": confidence,
                        "risk_reward": risk_reward
                    }
                    
                    self.trades_history.append(trade_info)
                    self.logger.info(f"Trade executed: {action}, PnL: {pnl}, Pattern: {pattern}")
                    
                    # Notificar ejecución de trade
                    if self.on_trade_executed:
                        self.on_trade_executed(trade_info)
            except Exception as e:
                self.logger.error(f"Error processing trade execution: {e}")
    
    def send_data_message(self, message: str) -> bool:
        """
        Envía un mensaje por la conexión de datos.
        
        Args:
            message: Mensaje a enviar
            
        Returns:
            bool: True si el mensaje se encoló correctamente
        """
        if not self.connected_data:
            self.logger.warning("Data connection not established")
            return False
        
        try:
            self.data_send_queue.put(message)
            return True
        except Exception as e:
            self.logger.error(f"Error queuing data message: {e}")
            return False
    
    def send_trading_signal(self, signal: int, position_size: float, stop_loss: float, 
                           take_profit: float, pattern_type: str, confidence: float, 
                           risk_reward: float) -> bool:
        """
        Envía una señal de trading a NinjaTrader.
        
        Args:
            signal: Señal de trading (-1=short, 0=flat, 1=long)
            position_size: Tamaño de la posición (0-1)
            stop_loss: Nivel de stop loss en ticks
            take_profit: Nivel de take profit en ticks
            pattern_type: Tipo de patrón detectado
            confidence: Confianza en el patrón (0-1)
            risk_reward: Ratio riesgo/recompensa
            
        Returns:
            bool: True si la señal se encoló correctamente
        """
        if not self.connected_order:
            self.logger.warning("Order connection not established")
            return False
        
        # Validar parámetros
        if signal not in [-1, 0, 1]:
            self.logger.error(f"Invalid signal value: {signal}")
            return False
        
        if position_size < 0 or position_size > 1:
            self.logger.error(f"Invalid position size: {position_size}")
            return False
        
        # Formatear mensaje
        message = f"{signal},{position_size},{stop_loss},{take_profit},{pattern_type},{confidence},{risk_reward}"
        
        try:
            self.order_send_queue.put(message)
            self.logger.info(f"Enqueued trading signal: {message}")
            return True
        except Exception as e:
            self.logger.error(f"Error queuing order message: {e}")
            return False
    
    def is_connected(self) -> bool:
        """
        Comprueba si el cliente está conectado a NinjaTrader.
        
        Returns:
            bool: True si ambas conexiones están establecidas
        """
        return self.connected_data and self.connected_order
    
    def get_trades_history(self) -> List[Dict]:
        """
        Obtiene el historial de trades.
        
        Returns:
            List[Dict]: Lista de trades ejecutados
        """
        return self.trades_history.copy()