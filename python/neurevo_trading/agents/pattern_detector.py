import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional

class PatternDetectorCNN(nn.Module):
    """
    Detector de patrones de reversión basado en redes neuronales convolucionales.
    Diseñado para identificar patrones de precio como doble techo/suelo, 
    cabeza y hombros, etc.
    """
    
    def __init__(self, input_channels, window_size, num_patterns=5):
        """
        Inicializa el detector de patrones.
        
        Args:
            input_channels (int): Número de características de entrada (OHLC + indicadores)
            window_size (int): Tamaño de la ventana de tiempo
            num_patterns (int): Número de patrones a detectar
        """
        super(PatternDetectorCNN, self).__init__()
        
        self.input_channels = input_channels
        self.window_size = window_size
        self.num_patterns = num_patterns
        
        # Capas convolucionales para extraer características
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # Pooling para reducir dimensionalidad
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calcular tamaño después de convoluciones y pooling
        self.fc_input_size = 128 * (window_size // 8)
        
        # Capas fully-connected para clasificación
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # Capa de salida para clasificación de patrones
        self.pattern_head = nn.Linear(128, num_patterns)
        
        # Capa de salida para regresión (calidad/confianza del patrón)
        self.confidence_head = nn.Linear(128, 1)
        
        # Dropout para regularización
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        Pasa los datos a través de la red.
        
        Args:
            x (torch.Tensor): Tensor de forma [batch_size, input_channels, window_size]
        
        Returns:
            tuple: (pattern_logits, confidence)
        """
        # Capas convolucionales
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Aplanar para capas fully-connected
        x = x.view(-1, self.fc_input_size)
        
        # Fully-connected con dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Cabezas de salida
        pattern_logits = self.pattern_head(x)
        confidence = torch.sigmoid(self.confidence_head(x))
        
        return pattern_logits, confidence


class PatternDetector:
    """
    Clase para detectar patrones de reversión en datos de precios.
    Combina el modelo de CNN con análisis técnico clásico.
    """
    
    def __init__(self, input_channels=5, window_size=50, device=None):
        """
        Inicializa el detector de patrones.
        
        Args:
            input_channels (int): Número de características por punto de tiempo
            window_size (int): Tamaño de la ventana de tiempo
            device (str): Dispositivo para cálculos ('cuda' o 'cpu')
        """
        self.input_channels = input_channels
        self.window_size = window_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Crear modelo CNN
        self.model = PatternDetectorCNN(input_channels, window_size).to(self.device)
        
        # Nombres de patrones
        self.pattern_names = [
            'NO_PATTERN',
            'DOUBLE_TOP',
            'DOUBLE_BOTTOM',
            'HEAD_AND_SHOULDERS',
            'INV_HEAD_AND_SHOULDERS'
        ]
        
        # Inicializar history para seguimiento
        self.detection_history = []
    
    def detect(self, data: pd.DataFrame) -> Dict:
        """
        Detecta patrones en los datos proporcionados.
        
        Args:
            data (pd.DataFrame): DataFrame con datos OHLCV e indicadores
            
        Returns:
            dict: Información del patrón detectado 
                 {pattern_name, confidence, entry_price, stop_loss, take_profit}
        """
        # Verificar longitud suficiente de datos
        if len(data) < self.window_size:
            return {'pattern_name': 'NO_PATTERN', 'confidence': 0.0}
        
        # Obtener últimos datos
        recent_data = data.iloc[-self.window_size:].copy()
        
        # Procesar datos para el modelo
        processed_data = self._preprocess_data(recent_data)
        
        # Ejecutar modelo para obtener predicción
        with torch.no_grad():
            processed_data = torch.tensor(processed_data, dtype=torch.float32).to(self.device)
            pattern_logits, confidence = self.model(processed_data)
            
            # Obtener patrón con mayor probabilidad
            pattern_idx = torch.argmax(pattern_logits, dim=1).item()
            pattern_name = self.pattern_names[pattern_idx]
            confidence_value = confidence.item()
        
        # Verificar también con técnicas clásicas
        technical_pattern = self._detect_pattern_technical(recent_data)
        
        # Si el modelo y las técnicas clásicas coinciden, aumentar la confianza
        if technical_pattern == pattern_name:
            confidence_value = min(1.0, confidence_value * 1.2)
        elif technical_pattern != 'NO_PATTERN' and pattern_name != 'NO_PATTERN':
            # Si ambos detectan patrones pero diferentes, usar el del modelo pero reducir confianza
            confidence_value *= 0.8
        elif technical_pattern != 'NO_PATTERN' and pattern_name == 'NO_PATTERN':
            # Si solo las técnicas clásicas detectan un patrón, usarlo con confianza media
            pattern_name = technical_pattern
            confidence_value = 0.6
        
        # Determinar niveles de entrada, stop loss y take profit
        entry_price, stop_loss, take_profit = self._calculate_trade_levels(recent_data, pattern_name)
        
        # Guardar en historial
        detection = {
            'timestamp': data.index[-1],
            'pattern_name': pattern_name,
            'confidence': confidence_value,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        self.detection_history.append(detection)
        
        return detection
    
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocesa los datos para el modelo CNN.
        
        Args:
            data (pd.DataFrame): Datos a preprocesar
            
        Returns:
            np.ndarray: Datos procesados listos para el modelo
        """
        # Seleccionar características principales (ajustar según las columnas disponibles)
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'FastEMA', 'SlowEMA', 'RSI', 'MACD', 'ATR'
        ]
        
        # Usar las que estén disponibles
        available_features = [f for f in features if f in data.columns]
        if len(available_features) < 5:
            # Si no hay suficientes, usar al menos OHLC y agregar columnas de ceros
            minimal_features = ['Open', 'High', 'Low', 'Close']
            for feature in minimal_features:
                if feature not in data.columns:
                    data[feature] = 0  # Valor por defecto
            available_features = minimal_features
            
            # Agregar columnas adicionales hasta tener al menos 5
            while len(available_features) < 5:
                new_feature = f'Dummy{len(available_features)+1}'
                data[new_feature] = 0
                available_features.append(new_feature)
        
        # Seleccionar solo las primeras input_channels características
        selected_features = available_features[:self.input_channels]
        
        # Normalizar las características
        normalized_data = data[selected_features].copy()
        for feature in selected_features:
            mean = normalized_data[feature].mean()
            std = normalized_data[feature].std()
            if std > 0:
                normalized_data[feature] = (normalized_data[feature] - mean) / std
            else:
                normalized_data[feature] = 0
        
        # Convertir a formato para CNN [batch_size, channels, sequence_length]
        processed_data = normalized_data.values.T  # Transponer para que cada fila sea una característica
        processed_data = np.expand_dims(processed_data, axis=0)  # Añadir dimensión de batch
        
        return processed_data
    
    def _detect_pattern_technical(self, data: pd.DataFrame) -> str:
        """
        Detecta patrones usando técnicas de análisis técnico clásico.
        
        Args:
            data (pd.DataFrame): Datos para análisis
            
        Returns:
            str: Nombre del patrón detectado
        """
        # Implementar detección de patrones usando indicadores técnicos
        # Esta es una implementación simplificada
        
        # Verificar columnas necesarias
        required_columns = ['High', 'Low', 'Close', 'SwingHigh', 'SwingLow']
        for col in required_columns:
            if col not in data.columns:
                return 'NO_PATTERN'
        
        # Double Top
        if self._is_double_top(data):
            return 'DOUBLE_TOP'
        
        # Double Bottom
        if self._is_double_bottom(data):
            return 'DOUBLE_BOTTOM'
        
        # Head and Shoulders
        if self._is_head_and_shoulders(data):
            return 'HEAD_AND_SHOULDERS'
        
        # Inverse Head and Shoulders
        if self._is_inverse_head_and_shoulders(data):
            return 'INV_HEAD_AND_SHOULDERS'
        
        return 'NO_PATTERN'
    
    def _is_double_top(self, data: pd.DataFrame) -> bool:
        """Detecta patrón de doble techo"""
        # Detectar dos máximos recientes
        if 'SwingHigh' not in data.columns:
            return False
            
        swing_highs = data[data['SwingHigh'] == 1].index
        if len(swing_highs) < 2:
            return False
        
        # Obtener los dos últimos swing highs
        last_two_highs = swing_highs[-2:]
        
        # Verificar que están cerca en precio
        high1 = data.loc[last_two_highs[0], 'High']
        high2 = data.loc[last_two_highs[1], 'High']
        price_diff_percent = abs(high1 - high2) / high1
        
        if price_diff_percent > 0.03:  # Diferencia de más del 3%
            return False
        
        # Verificar que hay un valle entre los dos picos
        between_idx = data.index[data.index > last_two_highs[0]]
        between_idx = between_idx[between_idx < last_two_highs[1]]
        
        if len(between_idx) < 3:
            return False
            
        between_low = data.loc[between_idx, 'Low'].min()
        
        # Verificar ruptura de neckline (nivel de soporte)
        last_close = data['Close'].iloc[-1]
        neckline_broken = last_close < between_low
        
        # Verificar tendencia previa alcista
        first_idx = data.index.get_loc(last_two_highs[0])
        if first_idx < 5:
            return False
            
        prev_prices = data['Close'].iloc[first_idx-5:first_idx]
        uptrend = prev_prices.iloc[0] < prev_prices.iloc[-1]
        
        return neckline_broken and uptrend
    
    def _is_double_bottom(self, data: pd.DataFrame) -> bool:
        """Detecta patrón de doble suelo"""
        # Detectar dos mínimos recientes
        if 'SwingLow' not in data.columns:
            return False
            
        swing_lows = data[data['SwingLow'] == 1].index
        if len(swing_lows) < 2:
            return False
        
        # Obtener los dos últimos swing lows
        last_two_lows = swing_lows[-2:]
        
        # Verificar que están cerca en precio
        low1 = data.loc[last_two_lows[0], 'Low']
        low2 = data.loc[last_two_lows[1], 'Low']
        price_diff_percent = abs(low1 - low2) / low1
        
        if price_diff_percent > 0.03:  # Diferencia de más del 3%
            return False
        
        # Verificar que hay un pico entre los dos valles
        between_idx = data.index[data.index > last_two_lows[0]]
        between_idx = between_idx[between_idx < last_two_lows[1]]
        
        if len(between_idx) < 3:
            return False
            
        between_high = data.loc[between_idx, 'High'].max()
        
        # Verificar ruptura de neckline (nivel de resistencia)
        last_close = data['Close'].iloc[-1]
        neckline_broken = last_close > between_high
        
        # Verificar tendencia previa bajista
        first_idx = data.index.get_loc(last_two_lows[0])
        if first_idx < 5:
            return False
            
        prev_prices = data['Close'].iloc[first_idx-5:first_idx]
        downtrend = prev_prices.iloc[0] > prev_prices.iloc[-1]
        
        return neckline_broken and downtrend
    
    def _is_head_and_shoulders(self, data: pd.DataFrame) -> bool:
        """Detecta patrón de cabeza y hombros"""
        # Implementación similar a la del entorno de trading
        if 'SwingHigh' not in data.columns:
            return False
            
        # Buscar tres máximos locales recientes
        swing_highs = data[data['SwingHigh'] == 1].index
        if len(swing_highs) < 3:
            return False
        
        # Obtener los tres últimos swing highs
        last_three_highs = swing_highs[-3:]
        
        # Verificar que están en secuencia
        if not (data.index.get_loc(last_three_highs[0]) < 
                data.index.get_loc(last_three_highs[1]) < 
                data.index.get_loc(last_three_highs[2])):
            return False
        
        # Obtener alturas de los picos
        left_shoulder = data.loc[last_three_highs[0], 'High']
        head = data.loc[last_three_highs[1], 'High']
        right_shoulder = data.loc[last_three_highs[2], 'High']
        
        # Verificar que la cabeza es más alta que los hombros
        if not (head > left_shoulder and head > right_shoulder):
            return False
        
        # Verificar que los hombros son similares en altura
        shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
        if shoulder_diff > 0.1:  # Diferencia de más del 10%
            return False
        
        # Encontrar los valles entre los picos
        between_left_head = data.index[data.index > last_three_highs[0]]
        between_left_head = between_left_head[between_left_head < last_three_highs[1]]
        
        between_head_right = data.index[data.index > last_three_highs[1]]
        between_head_right = between_head_right[between_head_right < last_three_highs[2]]
        
        if len(between_left_head) < 3 or len(between_head_right) < 3:
            return False
        
        # Encontrar la neckline
        left_valley = data.loc[between_left_head, 'Low'].min()
        right_valley = data.loc[between_head_right, 'Low'].min()
        
        # La neckline debe ser aproximadamente horizontal
        valley_diff = abs(left_valley - right_valley) / left_valley
        if valley_diff > 0.05:  # Diferencia de más del 5%
            return False
        
        # Verificar ruptura de la neckline
        neckline = (left_valley + right_valley) / 2
        last_close = data['Close'].iloc[-1]
        
        return last_close < neckline
    
    def _is_inverse_head_and_shoulders(self, data: pd.DataFrame) -> bool:
        """Detecta patrón de cabeza y hombros invertido"""
        # Implementación similar a la del entorno de trading
        if 'SwingLow' not in data.columns:
            return False
            
        # Buscar tres mínimos locales recientes
        swing_lows = data[data['SwingLow'] == 1].index
        if len(swing_lows) < 3:
            return False
        
        # Obtener los tres últimos swing lows
        last_three_lows = swing_lows[-3:]
        
        # Verificar que están en secuencia
        if not (data.index.get_loc(last_three_lows[0]) < 
                data.index.get_loc(last_three_lows[1]) < 
                data.index.get_loc(last_three_lows[2])):
            return False
        
        # Obtener alturas de los valles
        left_shoulder = data.loc[last_three_lows[0], 'Low']
        head = data.loc[last_three_lows[1], 'Low']
        right_shoulder = data.loc[last_three_lows[2], 'Low']
        
        # Verificar que la cabeza es más baja que los hombros
        if not (head < left_shoulder and head < right_shoulder):
            return False
        
        # Verificar que los hombros son similares en altura
        shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
        if shoulder_diff > 0.1:  # Diferencia de más del 10%
            return False
        
        # Encontrar los picos entre los valles
        between_left_head = data.index[data.index > last_three_lows[0]]
        between_left_head = between_left_head[between_left_head < last_three_lows[1]]
        
        between_head_right = data.index[data.index > last_three_lows[1]]
        between_head_right = between_head_right[between_head_right < last_three_lows[2]]
        
        if len(between_left_head) < 3 or len(between_head_right) < 3:
            return False
        
        # Encontrar la neckline
        left_peak = data.loc[between_left_head, 'High'].max()
        right_peak = data.loc[between_head_right, 'High'].max()
        
        # La neckline debe ser aproximadamente horizontal
        peak_diff = abs(left_peak - right_peak) / left_peak
        if peak_diff > 0.05:  # Diferencia de más del 5%
            return False
        
        # Verificar ruptura de la neckline
        neckline = (left_peak + right_peak) / 2
        last_close = data['Close'].iloc[-1]
        
        return last_close > neckline
    
    def _calculate_trade_levels(self, data: pd.DataFrame, pattern_name: str) -> Tuple[float, float, float]:
        """
        Calcula niveles de entrada, stop loss y take profit para el patrón.
        
        Args:
            data (pd.DataFrame): Datos para análisis
            pattern_name (str): Nombre del patrón detectado
            
        Returns:
            tuple: (entry_price, stop_loss, take_profit)
        """
        # Valores por defecto
        entry_price = data['Close'].iloc[-1]
        atr = data['ATR'].iloc[-1] if 'ATR' in data.columns else (data['High'] - data['Low']).mean()
        
        # Configurar niveles según el patrón
        if pattern_name == 'DOUBLE_TOP' or pattern_name == 'HEAD_AND_SHOULDERS':
            # Patrones de reversión bajista
            stop_loss = entry_price + 2 * atr
            take_profit = entry_price - 3 * atr
            
        elif pattern_name == 'DOUBLE_BOTTOM' or pattern_name == 'INV_HEAD_AND_SHOULDERS':
            # Patrones de reversión alcista
            stop_loss = entry_price - 2 * atr
            take_profit = entry_price + 3 * atr
            
        else:
            # Sin patrón o no reconocido
            stop_loss = entry_price - 2 * atr if entry_price > data['Close'].mean() else entry_price + 2 * atr
            take_profit = entry_price + 3 * atr if entry_price > data['Close'].mean() else entry_price - 3 * atr
        
        return entry_price, stop_loss, take_profit
    
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo entrenado.
        
        Args:
            filepath (str): Ruta para guardar el modelo
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_channels': self.input_channels,
            'window_size': self.window_size,
            'pattern_names': self.pattern_names
        }, filepath)
        print(f"Modelo guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Carga un modelo previamente entrenado.
        
        Args:
            filepath (str): Ruta del modelo guardado
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Verificar compatibilidad
        if checkpoint['input_channels'] != self.input_channels or checkpoint['window_size'] != self.window_size:
            print(f"Advertencia: Configuración del modelo guardado ({checkpoint['input_channels']} canales, "
                  f"ventana de {checkpoint['window_size']}) difiere de la actual "
                  f"({self.input_channels} canales, ventana de {self.window_size})")
            
            # Recrear modelo con parámetros correctos
            self.input_channels = checkpoint['input_channels']
            self.window_size = checkpoint['window_size']
            self.model = PatternDetectorCNN(self.input_channels, self.window_size).to(self.device)
        
        # Cargar parámetros y metadatos
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.pattern_names = checkpoint['pattern_names']
        
        print(f"Modelo cargado desde {filepath}")