import numpy as np
import pandas as pd
import gym
from gym import spaces

class TradingEnvironment(gym.Env):
    """
    Entorno de trading para aprendizaje por refuerzo.
    Diseñado para detectar patrones de reversión y ejecutar operaciones.
    """
    
    def __init__(self, data, window_size=50, initial_balance=10000, commission=0.0, slippage=0.0):
        """
        Inicializa el entorno de trading.
        
        Args:
            data (pandas.DataFrame): Datos históricos con indicadores
            window_size (int): Tamaño de la ventana de observación
            initial_balance (float): Balance inicial
            commission (float): Comisión por operación (porcentaje)
            slippage (float): Deslizamiento por operación (porcentaje)
        """
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        
        # Información actual de estado
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 0: sin posición, 1: long, -1: short
        self.entry_price = 0
        self.current_pattern = None
        self.position_size = 0
        
        # Información de recompensa
        self.trade_history = []
        self.total_pnl = 0
        self.current_trade_pnl = 0
        
        # Calcular número de características (datos + posición)
        self.num_features = window_size * data.shape[1] + 4  # Ventana + [posición, balance, pattern_confidence, días_en_posición]
        
        # Definir espacios de acción y observación
        # Acción: [señal, tamaño_posición, stop_loss, take_profit]
        # señal: -1 (short), 0 (hold/close), 1 (long)
        self.action_space = spaces.Box(
            low=np.array([-1, 0, 0, 0]),
            high=np.array([1, 1, 100, 100]),
            dtype=np.float32
        )
        
        # Observación: Histórico de datos + estado actual
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features,),
            dtype=np.float32
        )
        
        # Resetear al inicializar
        self.reset()
        
    def reset(self):
        """
        Reinicia el entorno para un nuevo episodio.
        
        Returns:
            np.array: Observación inicial
        """
        # Reiniciar variables de estado
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_pattern = None
        self.position_size = 0
        self.days_in_position = 0
        
        # Reiniciar historial y PnL
        self.trade_history = []
        self.total_pnl = 0
        self.current_trade_pnl = 0
        
        return self._get_observation()
    
    def step(self, action):
        """
        Ejecuta un paso en el entorno con la acción dada.
        
        Args:
            action (np.array): [señal, tamaño_posición, stop_loss, take_profit]
        
        Returns:
            tuple: (observación, recompensa, terminado, info)
        """
        # Descomponer la acción
        signal = np.clip(int(np.round(action[0])), -1, 1)  # -1: short, 0: hold/close, 1: long
        position_size = np.clip(action[1], 0, 1)           # Porcentaje del balance a arriesgar
        stop_loss_ticks = np.clip(int(action[2]), 1, 100)  # Número de ticks para stop loss
        take_profit_ticks = np.clip(int(action[3]), 1, 100) # Número de ticks para take profit
        
        # Obtener precios actuales
        current_price = self.data.iloc[self.current_step]['Close']
        current_tick_size = self.data.iloc[self.current_step]['ATR'] / 10  # Estimación de tick size basada en ATR
        
        # Calcular niveles de stop loss y take profit
        stop_loss = stop_loss_ticks * current_tick_size
        take_profit = take_profit_ticks * current_tick_size
        
        # Procesar señal de trading
        reward = 0
        info = {}
        done = False
        
        # Verificar si SL o TP se activaron basados en la barra actual
        if self.position != 0:
            # Actualizar días en posición
            self.days_in_position += 1
            
            # Verificar stop loss para posición long
            if self.position == 1 and self.data.iloc[self.current_step]['Low'] <= (self.entry_price - stop_loss):
                # Stop loss hit para long
                exit_price = self.entry_price - stop_loss
                reward, pnl = self._close_position(exit_price, 'stop_loss')
                info['exit_reason'] = 'stop_loss'
            
            # Verificar take profit para posición long
            elif self.position == 1 and self.data.iloc[self.current_step]['High'] >= (self.entry_price + take_profit):
                # Take profit hit para long
                exit_price = self.entry_price + take_profit
                reward, pnl = self._close_position(exit_price, 'take_profit')
                info['exit_reason'] = 'take_profit'
            
            # Verificar stop loss para posición short
            elif self.position == -1 and self.data.iloc[self.current_step]['High'] >= (self.entry_price + stop_loss):
                # Stop loss hit para short
                exit_price = self.entry_price + stop_loss
                reward, pnl = self._close_position(exit_price, 'stop_loss')
                info['exit_reason'] = 'stop_loss'
            
            # Verificar take profit para posición short
            elif self.position == -1 and self.data.iloc[self.current_step]['Low'] <= (self.entry_price - take_profit):
                # Take profit hit para short
                exit_price = self.entry_price - take_profit
                reward, pnl = self._close_position(exit_price, 'take_profit')
                info['exit_reason'] = 'take_profit'
            
            # Actualizar PnL no realizado
            if self.position == 1:
                unrealized_pnl = (current_price - self.entry_price) * self.position_size
                self.current_trade_pnl = unrealized_pnl
            elif self.position == -1:
                unrealized_pnl = (self.entry_price - current_price) * self.position_size
                self.current_trade_pnl = unrealized_pnl
        
        # Procesar señal del agente
        if signal != 0 and self.position == 0:
            # Abrir nueva posición
            self.position = signal
            self.entry_price = current_price
            self.position_size = position_size * self.balance
            self.days_in_position = 0
            
            # Detectar patrón de reversión
            self.current_pattern = self._detect_pattern()
            
            # Registrar trade
            self.trade_history.append({
                'entry_time': self.data.index[self.current_step],
                'entry_price': self.entry_price,
                'position': self.position,
                'position_size': self.position_size,
                'pattern': self.current_pattern,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            })
            
            info['action'] = 'open_position'
            info['position'] = 'long' if self.position == 1 else 'short'
            info['pattern'] = self.current_pattern
            info['risk_reward'] = take_profit / stop_loss
            
            # Pequeña recompensa por abrir posición basada en patrón
            pattern_quality = self._get_pattern_quality(self.current_pattern)
            reward += pattern_quality * 0.1
        
        elif signal == 0 and self.position != 0:
            # Cerrar posición existente
            reward, pnl = self._close_position(current_price, 'signal')
            info['exit_reason'] = 'signal'
        
        elif signal != 0 and self.position != 0 and signal != self.position:
            # Cambiar dirección (cerrar y abrir nueva posición)
            reward, pnl = self._close_position(current_price, 'reversal')
            
            # Abrir nueva posición en dirección opuesta
            self.position = signal
            self.entry_price = current_price
            self.position_size = position_size * self.balance
            self.days_in_position = 0
            
            # Detectar patrón de reversión
            self.current_pattern = self._detect_pattern()
            
            # Registrar nuevo trade
            self.trade_history.append({
                'entry_time': self.data.index[self.current_step],
                'entry_price': self.entry_price,
                'position': self.position,
                'position_size': self.position_size,
                'pattern': self.current_pattern,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            })
            
            info['action'] = 'reverse_position'
            info['position'] = 'long' if self.position == 1 else 'short'
            info['pattern'] = self.current_pattern
            info['risk_reward'] = take_profit / stop_loss
            
            # Pequeña recompensa adicional por el cambio de dirección si es apropiado
            pattern_quality = self._get_pattern_quality(self.current_pattern)
            reward += pattern_quality * 0.05
        
        # Avanzar al siguiente paso
        self.current_step += 1
        
        # Verificar si el episodio ha terminado
        if self.current_step >= len(self.data) - 1:
            done = True
            
            # Cerrar cualquier posición abierta al final
            if self.position != 0:
                final_reward, pnl = self._close_position(current_price, 'end_of_episode')
                reward += final_reward
        
        # Agregar información adicional
        info['balance'] = self.balance
        info['total_pnl'] = self.total_pnl
        info['current_step'] = self.current_step
        info['current_price'] = current_price
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """
        Construye la observación actual para el agente.
        
        Returns:
            np.array: Observación con ventana de datos y estado actual
        """
        # Obtener ventana de datos históricos
        window_data = self.data.iloc[self.current_step - self.window_size:self.current_step]
        
        # Normalizar datos
        normalized_data = self._normalize_data(window_data)
        
        # Aplanar datos
        flattened_data = normalized_data.values.flatten()
        
        # Agregar información de estado actual
        position_info = np.array([
            self.position,                                # Posición actual
            self.balance / self.initial_balance,          # Balance normalizado
            1.0 if self.current_pattern else 0.0,         # Confianza en el patrón
            self.days_in_position / 10                    # Días en posición (normalizado)
        ])
        
        # Combinar todo en la observación final
        observation = np.concatenate([flattened_data, position_info])
        
        return observation
    
    def _normalize_data(self, data):
        """
        Normaliza los datos para la red neuronal.
        
        Args:
            data (pandas.DataFrame): Datos a normalizar
            
        Returns:
            pandas.DataFrame: Datos normalizados
        """
        # Copia para no modificar los originales
        normalized = data.copy()
        
        # Normalizar por columna
        for column in normalized.columns:
            if column in ['Timestamp', 'BarType', 'SwingHigh', 'SwingLow']:
                continue  # Saltar columnas categóricas
                
            mean = data[column].mean()
            std = data[column].std()
            if std != 0:
                normalized[column] = (data[column] - mean) / std
            else:
                normalized[column] = 0  # Evitar división por cero
        
        return normalized
    
    def _close_position(self, price, reason):
        """
        Cierra una posición abierta y calcula la recompensa.
        
        Args:
            price (float): Precio de cierre
            reason (str): Razón del cierre ('stop_loss', 'take_profit', 'signal', etc.)
            
        Returns:
            tuple: (recompensa, pnl)
        """
        # Calcular P&L
        if self.position == 1:  # Long
            pnl = (price - self.entry_price) * self.position_size
        else:  # Short
            pnl = (self.entry_price - price) * self.position_size
        
        # Aplicar comisiones y slippage
        commission_amount = self.position_size * self.commission
        slippage_amount = price * self.position_size * self.slippage
        net_pnl = pnl - commission_amount - slippage_amount
        
        # Actualizar balance
        self.balance += net_pnl
        self.total_pnl += net_pnl
        
        # Actualizar último trade
        if self.trade_history:
            last_trade = self.trade_history[-1]
            last_trade['exit_time'] = self.data.index[self.current_step]
            last_trade['exit_price'] = price
            last_trade['pnl'] = net_pnl
            last_trade['exit_reason'] = reason
        
        # Calcular recompensa basada en P&L y razón de cierre
        if reason == 'stop_loss':
            # Penalizar ligeramente los stop loss, pero no demasiado para no desincentivar la gestión de riesgos
            reward = -0.1 + (net_pnl / self.initial_balance * 5)
        elif reason == 'take_profit':
            # Premiar los take profit
            reward = 0.5 + (net_pnl / self.initial_balance * 10)
        else:
            # Recompensa neutral para otros tipos de cierre
            reward = net_pnl / self.initial_balance * 8
        
        # Resetear variables de posición
        self.position = 0
        self.entry_price = 0
        self.position_size = 0
        self.current_pattern = None
        self.days_in_position = 0
        self.current_trade_pnl = 0
        
        return reward, net_pnl
    
    def _detect_pattern(self):
        """
        Detecta patrones de reversión en los datos actuales.
        
        Returns:
            str: Tipo de patrón detectado o None
        """
        # Datos recientes para análisis de patrones
        recent_data = self.data.iloc[self.current_step - 20:self.current_step + 1]
        
        # Detectar Double Top
        if self._is_double_top(recent_data):
            return "DOUBLE_TOP"
        
        # Detectar Double Bottom
        if self._is_double_bottom(recent_data):
            return "DOUBLE_BOTTOM"
        
        # Detectar Head and Shoulders
        if self._is_head_and_shoulders(recent_data):
            return "HEAD_AND_SHOULDERS"
        
        # Detectar Inverse Head and Shoulders
        if self._is_inverse_head_and_shoulders(recent_data):
            return "INV_HEAD_AND_SHOULDERS"
        
        # No se detectó ningún patrón
        return None
    
    def _is_double_top(self, data):
        """Detecta patrón de doble techo"""
        # Implementación básica para detección de doble techo
        swing_highs = np.where(data['SwingHigh'] == 1)[0]
        
        if len(swing_highs) < 2:
            return False
        
        # Verificar si hay dos máximos locales cercanos en nivel de precio
        latest_swings = swing_highs[-2:]
        high1 = data.iloc[latest_swings[0]]['High']
        high2 = data.iloc[latest_swings[1]]['High']
        
        # Los dos máximos deben estar cerca en precio
        price_diff_percent = abs(high1 - high2) / high1
        
        # Verificar si hay una tendencia alcista antes del primer máximo
        pre_pattern_index = max(0, latest_swings[0] - 5)
        pre_pattern_price = data.iloc[pre_pattern_index]['Close']
        first_high_price = data.iloc[latest_swings[0]]['Close']
        uptrend = first_high_price > pre_pattern_price
        
        # Verificar neckline (línea de soporte) y ruptura
        between_highs = data.iloc[latest_swings[0]:latest_swings[1]]
        if len(between_highs) < 3:
            return False
            
        neckline = between_highs['Low'].min()
        latest_close = data.iloc[-1]['Close']
        
        # Confirmar ruptura de neckline
        neckline_break = latest_close < neckline
        
        return uptrend and price_diff_percent < 0.02 and neckline_break
    
    def _is_double_bottom(self, data):
        """Detecta patrón de doble suelo"""
        # Implementación básica para detección de doble suelo
        swing_lows = np.where(data['SwingLow'] == 1)[0]
        
        if len(swing_lows) < 2:
            return False
        
        # Verificar si hay dos mínimos locales cercanos en nivel de precio
        latest_swings = swing_lows[-2:]
        low1 = data.iloc[latest_swings[0]]['Low']
        low2 = data.iloc[latest_swings[1]]['Low']
        
        # Los dos mínimos deben estar cerca en precio
        price_diff_percent = abs(low1 - low2) / low1
        
        # Verificar si hay una tendencia bajista antes del primer mínimo
        pre_pattern_index = max(0, latest_swings[0] - 5)
        pre_pattern_price = data.iloc[pre_pattern_index]['Close']
        first_low_price = data.iloc[latest_swings[0]]['Close']
        downtrend = first_low_price < pre_pattern_price
        
        # Verificar neckline (línea de resistencia) y ruptura
        between_lows = data.iloc[latest_swings[0]:latest_swings[1]]
        if len(between_lows) < 3:
            return False
            
        neckline = between_lows['High'].max()
        latest_close = data.iloc[-1]['Close']
        
        # Confirmar ruptura de neckline
        neckline_break = latest_close > neckline
        
        return downtrend and price_diff_percent < 0.02 and neckline_break
    
    def _is_head_and_shoulders(self, data):
        """Detecta patrón de cabeza y hombros"""
        # Implementación simplificada para detección de cabeza y hombros
        swing_highs = np.where(data['SwingHigh'] == 1)[0]
        
        if len(swing_highs) < 3:
            return False
        
        # Necesitamos al menos 3 swing highs para el patrón
        latest_swings = swing_highs[-3:]
        
        # Asegurarse de que están en el orden correcto
        if not (latest_swings[0] < latest_swings[1] < latest_swings[2]):
            return False
        
        # Extraer precios de los hombros y la cabeza
        left_shoulder = data.iloc[latest_swings[0]]['High']
        head = data.iloc[latest_swings[1]]['High']
        right_shoulder = data.iloc[latest_swings[2]]['High']
        
        # Verificar que la cabeza es más alta que los hombros
        if not (head > left_shoulder and head > right_shoulder):
            return False
        
        # Verificar que los hombros son aproximadamente similares en altura
        shoulder_diff_percent = abs(left_shoulder - right_shoulder) / left_shoulder
        if shoulder_diff_percent > 0.05:  # Permitir hasta un 5% de diferencia
            return False
        
        # Encontrar la neckline (línea de soporte)
        between_left_and_head = data.iloc[latest_swings[0]:latest_swings[1]]
        between_head_and_right = data.iloc[latest_swings[1]:latest_swings[2]]
        
        if len(between_left_and_head) < 2 or len(between_head_and_right) < 2:
            return False
            
        left_trough = between_left_and_head['Low'].min()
        right_trough = between_head_and_right['Low'].min()
        
        # La neckline debe ser aproximadamente horizontal
        neckline_diff_percent = abs(left_trough - right_trough) / left_trough
        if neckline_diff_percent > 0.03:  # Permitir hasta un 3% de diferencia
            return False
        
        # Verificar ruptura de la neckline
        neckline = min(left_trough, right_trough)
        latest_close = data.iloc[-1]['Close']
        
        # Confirmar ruptura
        neckline_break = latest_close < neckline
        
        return neckline_break
    
    def _is_inverse_head_and_shoulders(self, data):
        """Detecta patrón de cabeza y hombros invertido"""
        # Implementación simplificada para detección de cabeza y hombros invertido
        swing_lows = np.where(data['SwingLow'] == 1)[0]
        
        if len(swing_lows) < 3:
            return False
        
        # Necesitamos al menos 3 swing lows para el patrón
        latest_swings = swing_lows[-3:]
        
        # Asegurarse de que están en el orden correcto
        if not (latest_swings[0] < latest_swings[1] < latest_swings[2]):
            return False
        
        # Extraer precios de los hombros y la cabeza
        left_shoulder = data.iloc[latest_swings[0]]['Low']
        head = data.iloc[latest_swings[1]]['Low']
        right_shoulder = data.iloc[latest_swings[2]]['Low']
        
        # Verificar que la cabeza es más baja que los hombros
        if not (head < left_shoulder and head < right_shoulder):
            return False
        
        # Verificar que los hombros son aproximadamente similares en altura
        shoulder_diff_percent = abs(left_shoulder - right_shoulder) / left_shoulder
        if shoulder_diff_percent > 0.05:  # Permitir hasta un 5% de diferencia
            return False
        
        # Encontrar la neckline (línea de resistencia)
        between_left_and_head = data.iloc[latest_swings[0]:latest_swings[1]]
        between_head_and_right = data.iloc[latest_swings[1]:latest_swings[2]]
        
        if len(between_left_and_head) < 2 or len(between_head_and_right) < 2:
            return False
            
        left_peak = between_left_and_head['High'].max()
        right_peak = between_head_and_right['High'].max()
        
        # La neckline debe ser aproximadamente horizontal
        neckline_diff_percent = abs(left_peak - right_peak) / left_peak
        if neckline_diff_percent > 0.03:  # Permitir hasta un 3% de diferencia
            return False
        
        # Verificar ruptura de la neckline
        neckline = max(left_peak, right_peak)
        latest_close = data.iloc[-1]['Close']
        
        # Confirmar ruptura
        neckline_break = latest_close > neckline
        
        return neckline_break
    
    def _get_pattern_quality(self, pattern):
        """
        Estima la calidad/confianza del patrón detectado.
        
        Args:
            pattern (str): Tipo de patrón
            
        Returns:
            float: Calidad del patrón (0-1)
        """
        if pattern is None:
            return 0.0
            
        # En una implementación completa, se evaluaría la calidad del patrón
        # basada en características específicas (simetría, volumen, etc.)
        return 0.8  # Valor predeterminado para esta implementación simplificada