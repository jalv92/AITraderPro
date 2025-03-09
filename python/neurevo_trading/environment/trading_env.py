import numpy as np
import pandas as pd
import gym
from gym import spaces

class TradingEnvironment(gym.Env):
    """
    Entorno de trading para aprendizaje por refuerzo.
    Diseñado para optimizar el crecimiento de la cuenta con una tendencia ascendente estable.
    """
    
    def __init__(self, data, window_size=50, initial_balance=10000, commission=0.0, slippage=0.0,
                 enable_sl_tp=False, default_sl_pct=0.02, default_tp_pct=0.03,
                 drawdown_thresholds=[0.05, 0.10, 0.15, 0.20]):
        """
        Inicializa el entorno de trading.
        
        Args:
            data: DataFrame con los datos del mercado
            window_size: Tamaño de la ventana de observación
            initial_balance: Balance inicial
            commission: Comisión por operación
            slippage: Deslizamiento en precios (slippage)
            enable_sl_tp: Si es True, habilita Stop Loss y Take Profit automáticos
            default_sl_pct: Porcentaje de Stop Loss por defecto (ej: 0.02 = 2%)
            default_tp_pct: Porcentaje de Take Profit por defecto (ej: 0.03 = 3%)
            drawdown_thresholds: Lista de umbrales de drawdown para penalizaciones progresivas
        """
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        
        # Configuración de Stop Loss y Take Profit
        self.enable_sl_tp = enable_sl_tp
        self.default_sl_pct = default_sl_pct
        self.default_tp_pct = default_tp_pct
        self.current_sl_price = None  # Precio actual de Stop Loss
        self.current_tp_price = None  # Precio actual de Take Profit
        
        # Umbrales de drawdown para penalización progresiva
        self.drawdown_thresholds = sorted(drawdown_thresholds)
        
        # Espacio de acción:
        # 0: No hacer nada
        # 1: Comprar (Long)
        # 2: Vender (Short)
        # Si enable_sl_tp está activado, las acciones incluyen:
        # 3-7: Comprar con SL/TP diferentes
        # 8-12: Vender con SL/TP diferentes
        if self.enable_sl_tp:
            # Con 5 combinaciones de SL/TP por cada dirección
            self.action_space = spaces.Discrete(13)
        else:
            # Sólo acciones básicas: nada, comprar, vender
            self.action_space = spaces.Discrete(3)
        
        # Calcular la dimensión de la observación (depende de los datos y ventana)
        observation_shape = len(data.columns) * window_size + 3  # +3 por posición, balance y días en posición
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                           shape=(observation_shape,), dtype=np.float32)
        
        # Información actual de estado
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 0: sin posición, 1: long, -1: short
        self.entry_price = 0
        self.position_size = 0
        self.days_in_position = 0
        
        # Seguimiento del PnL
        self.trade_history = []
        self.total_pnl = 0
        self.current_trade_pnl = 0
        self.equity_curve = [initial_balance]
        self.max_drawdown = 0
        self.max_balance = initial_balance
        
        # Resetear al inicializar
        self.reset()
        
    def reset(self):
        """
        Reinicia el entorno al estado inicial.
        
        Returns:
            Observación inicial
        """
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0  # 0: sin posición, 1: long, -1: short
        self.entry_price = 0
        self.position_size = 0
        self.days_in_position = 0
        self.current_trade_pnl = 0
        self.total_pnl = 0
        self.trade_history = []
        self.equity_curve = [self.initial_balance]
        self.max_balance = self.initial_balance
        self.max_drawdown = 0
        self.done = False
        
        # Reiniciar Stop Loss y Take Profit
        self.current_sl_price = None
        self.current_tp_price = None
        
        return self._get_observation()
    
    def step(self, action):
        """
        Ejecuta un paso en el entorno con la acción dada.
        
        Args:
            action: Acción a ejecutar
            
        Returns:
            tuple: (observación, recompensa, terminado, info)
        """
        # Verificar si el episodio ha terminado
        if self.done:
            return self._get_observation(), 0, True, {"balance": self.balance, "max_drawdown": self.max_drawdown}
        
        # Obtener precio actual
        current_price = self._get_current_price()
        
        # Guardar balance anterior
        prev_balance = self.balance
        
        # Procesar SL/TP antes de la acción nueva
        if self.position != 0:
            # Verificar si se ha alcanzado el SL o TP
            if self.enable_sl_tp and (self.current_sl_price is not None or self.current_tp_price is not None):
                # Para posiciones largas
                if self.position > 0:
                    # Verificar Stop Loss
                    if self.current_sl_price is not None and current_price <= self.current_sl_price:
                        self._close_position(current_price, "stop_loss")
                    # Verificar Take Profit
                    elif self.current_tp_price is not None and current_price >= self.current_tp_price:
                        self._close_position(current_price, "take_profit")
                
                # Para posiciones cortas
                elif self.position < 0:
                    # Verificar Stop Loss
                    if self.current_sl_price is not None and current_price >= self.current_sl_price:
                        self._close_position(current_price, "stop_loss")
                    # Verificar Take Profit
                    elif self.current_tp_price is not None and current_price <= self.current_tp_price:
                        self._close_position(current_price, "take_profit")
        
        # Decodificar acción con SL/TP si está habilitado
        if self.enable_sl_tp and action >= 3:
            # Obtener dirección (compra/venta) y configuración de SL/TP
            if action < 8:  # Acciones de compra (long) con SL/TP
                direction = 1  # Compra
                sl_tp_config = (action - 3)  # 0-4 para diferentes configuraciones
            else:  # Acciones de venta (short) con SL/TP
                direction = 2  # Venta
                sl_tp_config = (action - 8)  # 0-4 para diferentes configuraciones
            
            # Calcular SL/TP basados en la configuración
            sl_pct, tp_pct = self._get_sl_tp_values(sl_tp_config)
            
            # Configurar precio actual de SL/TP
            if direction == 1:  # Compra (long)
                self.current_sl_price = current_price * (1 - sl_pct)
                self.current_tp_price = current_price * (1 + tp_pct)
            else:  # Venta (short)
                self.current_sl_price = current_price * (1 + sl_pct)
                self.current_tp_price = current_price * (1 - tp_pct)
                
            # Actualizar acción al tipo básico (compra/venta) para el procesamiento estándar
            action = direction
        else:
            # Para acciones básicas (sin SL/TP específico) pero con SL/TP habilitado
            if self.enable_sl_tp and action < 3:
                if action == 1:  # Compra con SL/TP por defecto
                    self.current_sl_price = current_price * (1 - self.default_sl_pct)
                    self.current_tp_price = current_price * (1 + self.default_tp_pct)
                elif action == 2:  # Venta con SL/TP por defecto
                    self.current_sl_price = current_price * (1 + self.default_sl_pct)
                    self.current_tp_price = current_price * (1 - self.default_tp_pct)
        
        # Procesar acción básica (0=nada, 1=comprar, 2=vender)
        if action == 1:  # Comprar (Long)
            # Si ya estamos en posición corta, cerrar primero
            if self.position < 0:
                self._close_position(current_price, "reverse")
                
            # Abrir posición larga si no la tenemos ya
            if self.position == 0:
                self._open_position(current_price, 1)
        
        elif action == 2:  # Vender (Short)
            # Si ya estamos en posición larga, cerrar primero
            if self.position > 0:
                self._close_position(current_price, "reverse")
                
            # Abrir posición corta si no la tenemos ya
            if self.position == 0:
                self._open_position(current_price, -1)
        
        # Avanzar al siguiente paso
        self.current_step += 1
        
        # Verificar si hemos llegado al final de los datos
        if self.current_step >= len(self.data):
            self.done = True
            
            # Cerrar posiciones abiertas al final
            if self.position != 0:
                self._close_position(current_price, "end_of_data")
        
        # Actualizar máximo histórico de balance y drawdown
        if self.balance > self.max_balance:
            self.max_balance = self.balance
        
        # Calcular drawdown actual
        if self.max_balance > 0:
            current_drawdown = 1 - (self.balance / self.max_balance)
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
        
        # Actualizar la curva de equity
        self.equity_curve.append(self.balance)
        
        # Calcular recompensa
        reward = self._calculate_reward(self.balance)
        
        # Información adicional
        info = {
            "balance": self.balance,
            "position": self.position,
            "sl_price": self.current_sl_price,
            "tp_price": self.current_tp_price,
            "max_drawdown": self.max_drawdown
        }
        
        return self._get_observation(), reward, self.done, info
    
    def _calculate_reward(self, current_balance):
        """
        Calcula la recompensa basada en el crecimiento de la cuenta y el drawdown.
        Implementa penalizaciones progresivas por drawdowns excesivos.
        
        Args:
            current_balance: Balance actual de la cuenta
            
        Returns:
            float: Valor de recompensa
        """
        # Calcular rendimiento en este paso
        if len(self.equity_curve) < 2:
            return 0
        
        previous_balance = self.equity_curve[-2]
        
        # Evitar divisiones por cero o valores extremos
        if previous_balance == 0 or abs(previous_balance) < 1e-10:
            previous_balance = self.initial_balance
        
        # Calcular cambio porcentual en lugar de absoluto
        balance_change_pct = (current_balance - previous_balance) / max(abs(previous_balance), 1.0)
        
        # Limitar el cambio a un rango razonable para evitar valores extremos
        balance_change_pct = np.clip(balance_change_pct, -1.0, 1.0)
        
        # Recompensa base por el cambio en el balance (escalada)
        reward = balance_change_pct * 10  # Escalar para que sea significativo pero no extremo
        
        # Sistema de penalización progresiva por drawdown
        drawdown_penalty = 0
        
        # Si el drawdown excede los umbrales, aplicar penalizaciones progresivas
        if self.max_drawdown > 0:
            # Aplicar penalizaciones base por el drawdown actual
            drawdown_penalty = self.max_drawdown * 10
            
            # Añadir penalizaciones adicionales por cada umbral superado
            for i, threshold in enumerate(self.drawdown_thresholds):
                if self.max_drawdown > threshold:
                    # Penalización progresiva más severa a medida que se superan umbrales
                    # Factor exponencial: cada umbral superado multiplica por 1.5
                    severity_factor = 1.5 ** (i + 1)
                    threshold_penalty = (self.max_drawdown - threshold) * 15 * severity_factor
                    drawdown_penalty += threshold_penalty
            
            # Limitar la penalización total para evitar valores extremos
            drawdown_penalty = min(drawdown_penalty, 50.0)
        
        reward -= drawdown_penalty
        
        # Penalización por no operar (permanecer en efectivo demasiado tiempo)
        if self.position == 0 and len(self.trade_history) > 0:
            last_trade = self.trade_history[-1]
            idle_time = self.current_step - last_trade.get("exit_date", last_trade["entry_date"])
            if idle_time > 20:  # Penalizar después de 20 periodos sin operar
                idle_penalty = min(0.01 * (idle_time - 20), 1.0)  # Limitar a un máximo de 1
                reward -= idle_penalty
        
        # Factor de estabilidad (recompensar crecimiento consistente)
        if len(self.equity_curve) > 30:
            recent_equity = self.equity_curve[-30:]
            
            # Calcular tendencia de la curva de equity usando regresión lineal
            x = np.arange(len(recent_equity))
            y = np.array(recent_equity)
            
            try:
                # Calcular pendiente usando regresión lineal básica
                slope = np.polyfit(x, y, 1)[0]
                
                # Normalizar pendiente (queremos una pendiente de aproximadamente 45 grados)
                # Una pendiente ideal sería:
                ideal_slope = (self.initial_balance * 0.01) / 30  # 1% de crecimiento en 30 pasos
                
                # Bonificación por tener una pendiente positiva y estable
                slope_factor = min(slope / ideal_slope, 1.0) if ideal_slope > 0 else 0
                stability_bonus = slope_factor * 0.5
                
                reward += stability_bonus
            except:
                pass  # Ignorar errores en el cálculo de la pendiente
        
        # Limitar la recompensa final a un rango razonable
        reward = np.clip(reward, -100.0, 100.0)
        
        return float(reward)
    
    def _get_sl_tp_values(self, config_index):
        """
        Devuelve los valores de Stop Loss y Take Profit basados en el índice de configuración.
        
        Args:
            config_index: Índice entre 0 y 4 para diferentes configuraciones
            
        Returns:
            tuple: (stop_loss_percentage, take_profit_percentage)
        """
        # Configuraciones predefinidas de SL/TP
        # El índice determina la relación riesgo/recompensa y la distancia
        config = {
            0: (0.01, 0.02),  # SL 1%, TP 2% - Conservador, relación 1:2
            1: (0.02, 0.03),  # SL 2%, TP 3% - Equilibrado, relación 1:1.5
            2: (0.03, 0.06),  # SL 3%, TP 6% - Agresivo, relación 1:2
            3: (0.01, 0.01),  # SL 1%, TP 1% - Scalping, relación 1:1
            4: (0.05, 0.10),  # SL 5%, TP 10% - Muy agresivo, relación 1:2
        }
        
        return config.get(config_index, (self.default_sl_pct, self.default_tp_pct))
    
    def _get_observation(self):
        """
        Construye el vector de observación.
        
        Returns:
            np.array: Vector de observación
        """
        # Obtener ventana de datos
        data_window = self.data.iloc[self.current_step - self.window_size:self.current_step]
        
        # Normalizar datos (con mensajes de error desactivados)
        normalized_data = self._normalize_data(data_window, verbose=False)
        
        # Aplanar datos
        flattened_data = normalized_data.values.flatten()
        
        # Información adicional
        position_info = np.array([
            self.position,
            self.balance / self.initial_balance,  # Normalizar balance
            self.days_in_position / 30  # Normalizar días en posición
        ])
        
        # Concatenar todo
        observation = np.concatenate([flattened_data, position_info])
        
        return observation
    
    def _normalize_data(self, data: pd.DataFrame, verbose=False) -> pd.DataFrame:
        """
        Normaliza los datos para el entrenamiento usando diferentes estrategias según el tipo de columna.
        Implementa múltiples protecciones contra valores inválidos o extremos.
        
        Args:
            data: DataFrame con los datos a normalizar
            verbose: Si es True, muestra mensajes detallados
            
        Returns:
            DataFrame con los datos normalizados
        """
        # Si los datos están vacíos o son None, devolver un DataFrame vacío
        if data is None or len(data) == 0:
            if verbose:
                print("ADVERTENCIA: Datos vacíos para normalizar")
            return pd.DataFrame()
        
        # Crear una copia para no modificar los datos originales
        try:
            result = data.copy()
        except Exception as e:
            if verbose:
                print(f"ERROR al copiar datos para normalización: {e}")
            # Intentar una copia alternativa si falla la primera
            try:
                result = pd.DataFrame(data.values, columns=data.columns)
            except:
                if verbose:
                    print("ERROR crítico: No se pudieron copiar los datos para normalización")
                # En caso de error crítico, devolver datos originales
                return data
        
        # Verificar número mínimo de filas para normalización
        if len(result) < 2:
            if verbose:
                print("ADVERTENCIA: Insuficientes datos para normalización estadística")
            # No hay suficientes datos para calcular estadísticas
            return result
        
        # Pre-procesamiento: remover NaN e infinitos antes de normalizar
        result = result.replace([np.inf, -np.inf], np.nan)
        
        # Detectar y guardar columnas con todos valores NaN o constantes
        problematic_columns = []
        for col in result.columns:
            # Verificar si todos son NaN
            if result[col].isna().all():
                result[col] = 0  # Reemplazar con ceros
                problematic_columns.append(col)
                continue
                
            # Verificar si todos los valores son iguales
            if result[col].nunique() <= 1:
                problematic_columns.append(col)
        
        # Aplicar normalización por columna con manejo de casos especiales
        for column in result.columns:
            # Saltear columnas ya marcadas como problemáticas
            if column in problematic_columns:
                result[column] = 0
                continue
                
            try:
                # Convertir nombres de columnas a minúsculas para comparación
                column_lower = column.lower()
                
                # NORMALIZACIÓN DE PRECIOS
                price_terms = ["open", "high", "low", "close", "price"]
                is_price_column = False
                for term in price_terms:
                    if term in column_lower:
                        is_price_column = True
                        break
                
                if is_price_column:
                    # Buscar primer valor válido (no NaN)
                    first_valid_idx = result[column].first_valid_index()
                    
                    if first_valid_idx is not None:
                        first_valid = float(result[column].loc[first_valid_idx])
                        
                        # Verificar que sea un valor válido y no cero
                        if np.isfinite(first_valid) and first_valid != 0:
                            # Normalizar como porcentaje de cambio desde el primer valor
                            result[column] = result[column] / first_valid - 1.0
                        else:
                            # Usar Z-score con protección contra desviación estándar cero
                            mean = float(result[column].mean())
                            std = float(result[column].std())
                            if np.isfinite(mean) and np.isfinite(std) and std > 1e-8:
                                result[column] = (result[column] - mean) / std
                            else:
                                # Si no es posible normalizar, restar la media si es válida
                                if np.isfinite(mean):
                                    result[column] = result[column] - mean
                    else:
                        # Si no hay valores válidos, establecer a cero
                        result[column] = 0
                
                # NORMALIZACIÓN DE VOLUMEN
                elif "volume" in column_lower:
                    # Para volumen, usar normalización min-max con protección
                    min_val = float(result[column].min())
                    max_val = float(result[column].max())
                    
                    # Verificar que sea una normalización válida
                    if np.isfinite(min_val) and np.isfinite(max_val) and max_val > min_val:
                        result[column] = (result[column] - min_val) / (max_val - min_val)
                    else:
                        # Normalizar por la media con protección contra división por cero
                        mean = float(result[column].mean())
                        if np.isfinite(mean) and mean > 1e-8:
                            result[column] = result[column] / mean
                        else:
                            # Si no hay valores válidos, establecer a cero
                            result[column] = 0
                
                # NORMALIZACIÓN DE INDICADORES TÉCNICOS Y OTRAS COLUMNAS
                else:
                    # Intentar Z-score primero (método preferido para la mayoría de indicadores)
                    try:
                        mean = float(result[column].mean())
                        std = float(result[column].std())
                        
                        # Verificar que la normalización sea válida
                        if np.isfinite(mean) and np.isfinite(std) and std > 1e-8:
                            result[column] = (result[column] - mean) / std
                        else:
                            # Si la desviación estándar es muy pequeña, intentar centrar
                            if np.isfinite(mean):
                                result[column] = result[column] - mean
                            else:
                                # Intentar normalización min-max como último recurso
                                min_val = float(result[column].min())
                                max_val = float(result[column].max())
                                
                                if np.isfinite(min_val) and np.isfinite(max_val) and max_val > min_val:
                                    result[column] = (result[column] - min_val) / (max_val - min_val)
                                else:
                                    # Si todo falla, usar división por el máximo absoluto
                                    abs_max = float(result[column].abs().max())
                                    if np.isfinite(abs_max) and abs_max > 1e-8:
                                        result[column] = result[column] / abs_max
                                    else:
                                        # Como último recurso, establecer a cero
                                        result[column] = 0
                    except Exception as inner_e:
                        # Si hay un error en los cálculos, intentar métodos alternativos
                        if verbose:
                            print(f"Error en normalización Z-score para {column}: {inner_e}")
                        
                        try:
                            # Intentar simplemente dividir por el máximo absoluto
                            abs_max = float(result[column].abs().max())
                            if np.isfinite(abs_max) and abs_max > 1e-8:
                                result[column] = result[column] / abs_max
                            else:
                                result[column] = 0
                        except:
                            # Si todo falla, establecer a cero
                            result[column] = 0
            
            except Exception as e:
                # Si falla la normalización para cualquier columna, manejar el error
                if verbose:
                    print(f"ERROR en normalización para columna {column}: {e}")
                # Asegurar que no hay valores NaN estableciendo a cero
                result[column] = 0
        
        # Post-procesamiento: verificar y reemplazar cualquier valor problemático restante
        try:
            # Reemplazar infinitos o NaN con ceros
            result = result.replace([np.inf, -np.inf], 0).fillna(0)
            
            # Verificar valores extremos (clipping)
            for col in result.columns:
                # Limitar valores extremos a ±10 para evitar problemas numéricos
                result[col] = result[col].clip(-10.0, 10.0)
        except Exception as e:
            if verbose:
                print(f"ERROR en post-procesamiento de normalización: {e}")
            # Si el post-procesamiento falla, intentar limpiar cada columna individualmente
            for col in result.columns:
                try:
                    result[col] = result[col].replace([np.inf, -np.inf], 0).fillna(0).clip(-10.0, 10.0)
                except:
                    result[col] = 0
        
        return result
    
    def _close_position(self, price, reason):
        """
        Cierra la posición actual.
        
        Args:
            price: Precio de cierre
            reason: Razón del cierre (signal, stop_loss, take_profit, etc.)
        """
        if self.position == 0:
            return
        
        # Calcular PnL
        price_diff = price - self.entry_price
        if self.position < 0:  # Short
            price_diff = -price_diff
        
        # Limitar la diferencia de precio a valores razonables para evitar explosiones numéricas
        max_price_change = self.entry_price * 0.5  # Limitar a 50% del precio de entrada
        price_diff = np.clip(price_diff, -max_price_change, max_price_change)
        
        # Calcular PnL con diferencia limitada
        pnl = price_diff * abs(self.position_size)
        
        # Aplicar comisiones
        commission_amount = self.position_size * price * self.commission
        pnl -= commission_amount
        
        # Actualizar balance (con verificación de valores extremos)
        self.balance += pnl
        
        # Evitar valores extremos en el balance
        if not np.isfinite(self.balance) or abs(self.balance) > 1e12:
            self.balance = self.initial_balance * 0.5  # Reiniciar a la mitad del balance inicial si hay problemas
        
        self.total_pnl += pnl
        
        # Actualizar historial con información de SL/TP
        last_trade = self.trade_history[-1]
        last_trade.update({
            "exit_price": price,
            "exit_date": self.current_step,
            "pnl": pnl,
            "exit_reason": reason
        })
        
        # Métricas específicas para stop loss y take profit
        if reason == "stop_loss":
            last_trade["sl_triggered"] = True
            last_trade["sl_price"] = self.current_sl_price
        elif reason == "take_profit":
            last_trade["tp_triggered"] = True
            last_trade["tp_price"] = self.current_tp_price
        
        # Resetear posición
        self.position = 0
        self.entry_price = 0
        self.position_size = 0
        self.current_trade_pnl = 0
        self.days_in_position = 0
        
        # Limpiar SL/TP
        self.current_sl_price = None
        self.current_tp_price = None
    
    def _get_current_price(self):
        """
        Obtiene el precio actual.
        
        Returns:
            float: Precio actual
        """
        # Intentar obtener el precio con 'close' (minúsculas)
        if 'close' in self.data.columns:
            return self.data.iloc[self.current_step]["close"]
        # Intentar obtener el precio con 'Close' (mayúsculas)
        elif 'Close' in self.data.columns:
            return self.data.iloc[self.current_step]["Close"]
        # Si no se encuentra ninguna de las dos, intentar buscar una columna que contenga 'close' o 'Close'
        else:
            close_columns = [col for col in self.data.columns if 'close' in col.lower()]
            if close_columns:
                return self.data.iloc[self.current_step][close_columns[0]]
            else:
                raise ValueError("No se encuentra una columna de precio de cierre en los datos. Las columnas disponibles son: " + str(list(self.data.columns)))
    
    def _open_position(self, price, direction):
        """
        Abre una nueva posición.
        
        Args:
            price: Precio de entrada
            direction: Dirección de la posición (1 para long, -1 para short)
        """
        self.position = direction
        self.entry_price = price
        self.position_size = 1  # Tamaño fijo de 1 contrato/unidad
        self.days_in_position = 0
        
        # Aplicar comisión
        commission_cost = price * self.position_size * self.commission
        self.balance -= commission_cost
        
        # Registrar en historial
        self.trade_history.append({
            "type": "long" if direction == 1 else "short",
            "entry_price": price,
            "entry_date": self.current_step,
            "sl_price": self.current_sl_price,
            "tp_price": self.current_tp_price
        })