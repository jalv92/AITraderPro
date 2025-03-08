import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union
import logging

from neurevo_trading.utils.feature_engineering import create_features

class DataProcessor:
    """
    Procesador de datos para AITraderPro.
    Maneja la carga, limpieza y preparación de datos para el análisis y trading.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Inicializa el procesador de datos.
        
        Args:
            logger: Logger opcional
        """
        self.logger = logger or self._setup_default_logger()
        self.data_cache = {}
    
    def _setup_default_logger(self) -> logging.Logger:
        """Configura un logger por defecto si no se proporciona ninguno."""
        logger = logging.getLogger("DataProcessor")
        logger.setLevel(logging.INFO)
        
        # Crear manejador de consola si no existe
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_csv(self, filepath: str, cache: bool = True, 
                date_column: str = 'Timestamp', parse_dates: bool = True) -> pd.DataFrame:
        """
        Carga datos desde un archivo CSV.
        
        Args:
            filepath: Ruta al archivo CSV
            cache: Si es True, almacena el DataFrame en caché
            date_column: Nombre de la columna de fecha/hora
            parse_dates: Si es True, convierte la columna de fecha a datetime
            
        Returns:
            DataFrame con los datos cargados
        """
        try:
            if filepath in self.data_cache and cache:
                self.logger.info(f"Usando datos en caché para {filepath}")
                return self.data_cache[filepath].copy()
            
            self.logger.info(f"Cargando datos desde {filepath}")
            
            # Cargar CSV
            df = pd.read_csv(filepath)
            
            # Configurar índice de tiempo si existe
            if date_column in df.columns and parse_dates:
                df[date_column] = pd.to_datetime(df[date_column])
                df.set_index(date_column, inplace=True)
            
            # Convertir columnas numéricas
            for col in df.columns:
                if col not in ['Timestamp', 'Date', 'Time', 'DateTime']:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except:
                        pass
            
            # Almacenar en caché si se solicita
            if cache:
                self.data_cache[filepath] = df.copy()
            
            self.logger.info(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
            return df
        
        except Exception as e:
            self.logger.error(f"Error al cargar {filepath}: {e}")
            raise
    
    def prepare_data(self, data: pd.DataFrame, add_features: bool = True, 
                    window_size: int = 50) -> pd.DataFrame:
        """
        Prepara los datos para su uso en análisis y trading.
        
        Args:
            data: DataFrame con datos de precios
            add_features: Si es True, agrega características adicionales
            window_size: Tamaño de la ventana para cálculos
            
        Returns:
            DataFrame preparado para análisis
        """
        try:
            # Crear copia para no modificar el original
            df = data.copy()
            
            # Verificar columnas mínimas requeridas
            required_cols = ['Open', 'High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.logger.error(f"Faltan columnas requeridas: {missing_cols}")
                raise ValueError(f"Faltan columnas requeridas: {missing_cols}")
            
            # Eliminar filas con valores faltantes en columnas críticas
            df = df.dropna(subset=required_cols)
            
            # Asegurarse de que exista una columna de volumen
            if 'Volume' not in df.columns:
                df['Volume'] = 0
                self.logger.warning("Columna 'Volume' no encontrada, usando valores por defecto")
            
            # Agregar características si se solicita
            if add_features:
                df = create_features(df)
            
            # Eliminar filas con valores NaN que puedan haberse creado
            df = df.dropna()
            
            self.logger.info(f"Datos preparados: {len(df)} filas, {len(df.columns)} columnas")
            return df
        
        except Exception as e:
            self.logger.error(f"Error al preparar datos: {e}")
            raise
    
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
                self.logger.warning("La suma de train_size y val_size debe ser menor que 1.0. Ajustando...")
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
            
            self.logger.info(f"Datos divididos: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
            return train_data, val_data, test_data
        
        except Exception as e:
            self.logger.error(f"Error al dividir datos: {e}")
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
                columns = [col for col in df.columns if col not in ['Timestamp', 'Date', 'Time', 'DateTime', 'BarType', 'SwingHigh', 'SwingLow']]
            
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
                self.logger.warning(f"Método de normalización '{method}' no reconocido. Usando datos sin normalizar.")
            
            return df, norm_params
        
        except Exception as e:
            self.logger.error(f"Error al normalizar datos: {e}")
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
            self.logger.error(f"Error al desnormalizar datos: {e}")
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
                self.logger.error(f"Tamaño de ventana {window_size} demasiado grande para {len(data)} muestras")
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
            self.logger.error(f"Error al crear muestras de ventana: {e}")
            raise