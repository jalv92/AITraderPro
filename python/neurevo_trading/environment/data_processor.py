import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union
import logging
import re

from neurevo_trading.utils.feature_engineering import create_features

# Configurar logging
logger = logging.getLogger("DataProcessor")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class DataProcessor:
    """
    Procesador de datos para AITraderPro.
    Maneja la carga, limpieza y preparación de datos para el análisis y trading.
    """
    
    def __init__(self):
        """
        Inicializa el procesador de datos.
        """
        self.required_columns = ['open', 'high', 'low', 'close']
        self.datetime_format = '%Y-%m-%d %H:%M:%S'
        self.data_cache = {}
    
    def load_csv(self, filepath: str, cache: bool = True, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Carga datos desde un archivo CSV.
        
        Args:
            filepath: Ruta al archivo CSV
            cache: Si es True, almacena el DataFrame en caché
            nrows: Número máximo de filas a cargar (opcional)
            
        Returns:
            DataFrame con los datos cargados
        """
        try:
            # Si se solicitan filas limitadas, no usar caché
            if nrows is not None:
                cache = False
            
            if filepath in self.data_cache and cache:
                logger.info(f"Usando datos en caché para {filepath}")
                return self.data_cache[filepath].copy()
            
            logger.info(f"Cargando datos desde {filepath}")
            
            # Cargar CSV (con opción de limitar filas)
            if nrows is not None:
                logger.info(f"Cargando solo {nrows} filas")
                data = pd.read_csv(filepath, nrows=nrows)
            else:
                data = pd.read_csv(filepath)
            
            # Convertir nombres de columnas a minúsculas para consistencia
            data.columns = [col.lower() for col in data.columns]
            logger.info(f"Columnas después de convertir a minúsculas: {list(data.columns)}")
            
            # Verificar si hay columna de timestamp y convertirla a datetime
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'], format=self.datetime_format, errors='coerce')
                # Establecer timestamp como índice si es una columna válida
                if not data['timestamp'].isnull().any():
                    data.set_index('timestamp', inplace=True)
            
            # Convertir columnas numéricas
            for col in data.columns:
                if col not in ['timestamp', 'date', 'time', 'datetime']:
                    try:
                        data[col] = pd.to_numeric(data[col])
                    except:
                        pass
            
            # Almacenar en caché si se solicita
            if cache:
                self.data_cache[filepath] = data.copy()
            
            logger.info(f"Datos cargados: {len(data)} filas, {len(data.columns)} columnas")
            return data
        
        except Exception as e:
            logger.error(f"Error al cargar {filepath}: {e}")
            raise
    
    def prepare_data(self, data: pd.DataFrame, add_features: bool = True) -> pd.DataFrame:
        """
        Prepara los datos para su uso en análisis y trading.
        
        Args:
            data: DataFrame con datos de precios
            add_features: Si es True, agrega características adicionales
            
        Returns:
            DataFrame preparado para análisis
        """
        try:
            # Hacer una copia para no modificar el original
            prepared_data = data.copy()
            
            # Asegurarse de que nombres de columnas estén en minúsculas
            prepared_data.columns = [col.lower() for col in prepared_data.columns]
            
            # Verificar columnas requeridas
            missing_columns = [col for col in self.required_columns if col not in prepared_data.columns]
            
            # Si hay columnas faltantes, intentar buscarlas con nombres que contengan las palabras clave
            if missing_columns:
                logger.warning(f"Faltan columnas requeridas: {missing_columns}")
                for missing_col in missing_columns.copy():
                    # Buscar columnas que contienen el nombre requerido
                    matching_cols = [col for col in prepared_data.columns if missing_col in col.lower()]
                    if matching_cols:
                        logger.info(f"Renombrando columna {matching_cols[0]} a {missing_col}")
                        prepared_data[missing_col] = prepared_data[matching_cols[0]]
                        missing_columns.remove(missing_col)
            
            # Verificar nuevamente si faltan columnas
            if missing_columns:
                # Intentar usar mayúsculas
                uppercase_columns = [col.upper() for col in self.required_columns]
                print(uppercase_columns)
                missing_uppercase = [col for col in uppercase_columns if col not in data.columns]
                if not missing_uppercase:
                    # Las columnas están en mayúsculas, copiarlas
                    for i, col in enumerate(uppercase_columns):
                        if col in data.columns:
                            prepared_data[self.required_columns[i]] = data[col]
                    logger.info("Columnas en mayúsculas copiadas a minúsculas")
                else:
                    raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
            
            # Agregar características si se solicita
            if add_features:
                prepared_data = self.add_technical_indicators(prepared_data)
            
            # Eliminar filas con valores NaN
            prepared_data.dropna(inplace=True)
            
            logger.info(f"Datos preparados: {len(prepared_data)} filas con {len(prepared_data.columns)} columnas")
            return prepared_data
        
        except Exception as e:
            logger.error(f"Error al preparar datos: {e}")
            raise
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega indicadores técnicos a los datos.
        
        Args:
            data: DataFrame con datos OHLC
            
        Returns:
            DataFrame con indicadores técnicos añadidos
        """
        # Asegurarse de que los nombres de columnas estén en minúsculas
        ohlc_data = data.copy()
        
        # Si no existen las columnas requeridas pero existen en mayúsculas, usar esas
        for req_col in self.required_columns:
            if req_col not in ohlc_data.columns and req_col.upper() in ohlc_data.columns:
                ohlc_data[req_col] = ohlc_data[req_col.upper()]
        
        # Agregar indicadores solo si existen las columnas necesarias
        if all(col in ohlc_data.columns for col in self.required_columns):
            # Medias móviles
            ohlc_data['sma_10'] = ohlc_data['close'].rolling(window=10).mean()
            ohlc_data['sma_20'] = ohlc_data['close'].rolling(window=20).mean()
            ohlc_data['sma_50'] = ohlc_data['close'].rolling(window=50).mean()
            
            # RSI (Relative Strength Index)
            delta = ohlc_data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            ohlc_data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD (Moving Average Convergence Divergence)
            ema_12 = ohlc_data['close'].ewm(span=12, adjust=False).mean()
            ema_26 = ohlc_data['close'].ewm(span=26, adjust=False).mean()
            ohlc_data['macd'] = ema_12 - ema_26
            ohlc_data['macd_signal'] = ohlc_data['macd'].ewm(span=9, adjust=False).mean()
            ohlc_data['macd_hist'] = ohlc_data['macd'] - ohlc_data['macd_signal']
            
            # Bandas de Bollinger
            ohlc_data['bb_middle'] = ohlc_data['close'].rolling(window=20).mean()
            std_dev = ohlc_data['close'].rolling(window=20).std()
            ohlc_data['bb_upper'] = ohlc_data['bb_middle'] + 2 * std_dev
            ohlc_data['bb_lower'] = ohlc_data['bb_middle'] - 2 * std_dev
            
            # ATR (Average True Range)
            high_low = ohlc_data['high'] - ohlc_data['low']
            high_close = (ohlc_data['high'] - ohlc_data['close'].shift()).abs()
            low_close = (ohlc_data['low'] - ohlc_data['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            ohlc_data['atr'] = true_range.rolling(14).mean()
            
            # Stochastic Oscillator
            low_14 = ohlc_data['low'].rolling(window=14).min()
            high_14 = ohlc_data['high'].rolling(window=14).max()
            ohlc_data['stoch_k'] = 100 * ((ohlc_data['close'] - low_14) / (high_14 - low_14))
            ohlc_data['stoch_d'] = ohlc_data['stoch_k'].rolling(window=3).mean()
            
            # Características de volatilidad
            ohlc_data['volatility'] = ohlc_data['close'].pct_change().rolling(window=10).std() * np.sqrt(10)
            
            # Características de tendencia
            ohlc_data['trend_5_20'] = ohlc_data['sma_5'] - ohlc_data['sma_20'] if 'sma_5' in ohlc_data.columns else ohlc_data['sma_10'] - ohlc_data['sma_20']
            ohlc_data['trend_direction'] = np.sign(ohlc_data['trend_5_20'])
            
            logger.info(f"Añadidos {len(ohlc_data.columns) - len(data.columns)} indicadores técnicos")
        else:
            logger.warning("No se pudieron agregar indicadores técnicos: faltan columnas OHLC")
        
        return ohlc_data
    
    def split_data(self, data: pd.DataFrame, train_size: float = 0.7, 
                  val_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide los datos en conjuntos de entrenamiento, validación y prueba.
        
        Args:
            data: DataFrame con datos de precios
            train_size: Proporción para entrenamiento
            val_size: Proporción para validación
            
        Returns:
            Tuple con DataFrames (train, val, test)
        """
        try:
            # Validar tamaños
            if train_size + val_size >= 1.0:
                logger.warning("La suma de train_size y val_size debe ser menor que 1.0. Ajustando...")
                test_size = 0.1
                total = train_size + val_size
                train_size = train_size / total * 0.9
                val_size = val_size / total * 0.9
            else:
                test_size = 1.0 - train_size - val_size
            
            # Número de filas para cada conjunto
            n = len(data)
            train_end = int(n * train_size)
            val_end = train_end + int(n * val_size)
            
            # Dividir datos
            train_data = data.iloc[:train_end].copy()
            val_data = data.iloc[train_end:val_end].copy()
            test_data = data.iloc[val_end:].copy()
            
            logger.info(f"Datos divididos: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
            return train_data, val_data, test_data
        
        except Exception as e:
            logger.error(f"Error al dividir datos: {e}")
            raise
    
    def normalize_data(self, data: pd.DataFrame, method: str = 'zscore',
                      columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Normaliza los datos para su uso en modelos de aprendizaje automático.
        
        Args:
            data: DataFrame con datos
            method: Método de normalización ('zscore', 'minmax')
            columns: Lista de columnas a normalizar (None para todas)
            
        Returns:
            Tuple con (datos_normalizados, parámetros_normalización)
        """
        try:
            # Crear copia para no modificar el original
            df = data.copy()
            
            # Determinar columnas a normalizar
            if columns is None:
                columns = [col for col in df.columns if col not in ['timestamp', 'date', 'time', 'datetime', 'bartype', 'swinghigh', 'swinglow']]
            
            # Parámetros de normalización
            norm_params = {}
            
            if method == 'zscore':
                # Normalización Z-score (media=0, std=1)
                for col in columns:
                    if col in df.columns:
                        mean = df[col].mean()
                        std = df[col].std()
                        
                        # Evitar divisiones por cero
                        if std == 0:
                            std = 1
                        
                        df[col] = (df[col] - mean) / std
                        norm_params[col] = {'mean': mean, 'std': std}
            
            elif method == 'minmax':
                # Normalización Min-Max (rango 0-1)
                for col in columns:
                    if col in df.columns:
                        min_val = df[col].min()
                        max_val = df[col].max()
                        
                        # Evitar divisiones por cero
                        if max_val == min_val:
                            df[col] = 0
                        else:
                            df[col] = (df[col] - min_val) / (max_val - min_val)
                        
                        norm_params[col] = {'min': min_val, 'max': max_val}
            
            else:
                logger.warning(f"Método de normalización '{method}' no reconocido. Usando datos sin normalizar.")
            
            return df, norm_params
        
        except Exception as e:
            logger.error(f"Error al normalizar datos: {e}")
            raise
    
    def denormalize_data(self, data: pd.DataFrame, norm_params: Dict, 
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Desnormaliza los datos a sus valores originales.
        
        Args:
            data: DataFrame con datos normalizados
            norm_params: Parámetros de normalización
            columns: Lista de columnas a desnormalizar (None para todas)
            
        Returns:
            DataFrame con datos desnormalizados
        """
        try:
            # Crear copia para no modificar el original
            df = data.copy()
            
            # Determinar columnas a desnormalizar
            if columns is None:
                columns = list(norm_params.keys())
            
            # Desnormalizar columnas
            for col in columns:
                if col in df.columns and col in norm_params:
                    params = norm_params[col]
                    
                    if 'mean' in params and 'std' in params:
                        # Desnormalizar Z-score
                        df[col] = df[col] * params['std'] + params['mean']
                    
                    elif 'min' in params and 'max' in params:
                        # Desnormalizar Min-Max
                        df[col] = df[col] * (params['max'] - params['min']) + params['min']
            
            return df
        
        except Exception as e:
            logger.error(f"Error al desnormalizar datos: {e}")
            raise
    
    def create_window_samples(self, data: pd.DataFrame, window_size: int, 
                             target_column: Optional[str] = None, 
                             step: int = 1) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Crea muestras de ventana deslizante para entrenamiento de modelos.
        
        Args:
            data: DataFrame con datos
            window_size: Tamaño de la ventana
            target_column: Columna objetivo (None si no hay objetivo)
            step: Paso entre ventanas
            
        Returns:
            Tuple con (X_samples, y_samples) donde y_samples puede ser None
        """
        try:
            n_samples = (len(data) - window_size) // step + 1
            
            if n_samples <= 0:
                logger.error(f"Tamaño de ventana {window_size} demasiado grande para {len(data)} muestras")
                raise ValueError(f"Tamaño de ventana {window_size} demasiado grande para {len(data)} muestras")
            
            # Crear muestras X
            X_samples = np.zeros((n_samples, window_size, len(data.columns)))
            
            for i in range(n_samples):
                X_samples[i] = data.iloc[i * step:i * step + window_size].values
            
            # Crear muestras y si se especifica una columna objetivo
            if target_column is not None and target_column in data.columns:
                target_idx = data.columns.get_loc(target_column)
                y_samples = np.array([X_samples[i, -1, target_idx] for i in range(n_samples)])
                return X_samples, y_samples
            
            return X_samples, None
        
        except Exception as e:
            logger.error(f"Error al crear muestras de ventana: {e}")
            raise