# python/neurevo_trading/utils/feature_engineering.py
import pandas as pd
import numpy as np

def create_features(data):
    """
    Crea características adicionales para análisis y modelado.
    
    Args:
        data (pandas.DataFrame): DataFrame con datos OHLCV e indicadores técnicos
        
    Returns:
        pandas.DataFrame: DataFrame con características adicionales
    """
    # Crear una copia para no modificar el original
    df = data.copy()
    
    # Asegurarse de que tenemos las columnas básicas
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas: {missing_cols}")
    
    # Características basadas en patrones de velas
    df['Body'] = abs(df['Close'] - df['Open'])
    df['UpperShadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['LowerShadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['BodyToRange'] = df['Body'] / (df['High'] - df['Low'])
    
    # Posición relativa dentro de la vela
    df['RelativeClose'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    # Detectar patrones de swing
    if 'SwingHigh' not in df.columns:
        df['SwingHigh'] = detect_swing_highs(df, window=5)
    
    if 'SwingLow' not in df.columns:
        df['SwingLow'] = detect_swing_lows(df, window=5)
    
    # Indicadores básicos si no existen
    # EMAs
    if 'FastEMA' not in df.columns and 'Close' in df.columns:
        df['FastEMA'] = df['Close'].ewm(span=9, adjust=False).mean()
    
    if 'SlowEMA' not in df.columns and 'Close' in df.columns:
        df['SlowEMA'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # RSI
    if 'RSI' not in df.columns and 'Close' in df.columns:
        df['RSI'] = calculate_rsi(df['Close'], window=14)
    
    # ATR
    if 'ATR' not in df.columns:
        df['ATR'] = calculate_atr(df, window=14)
    
    # MACD
    if 'MACD' not in df.columns and 'Close' in df.columns:
        macd, signal, hist = calculate_macd(df['Close'])
        df['MACD'] = macd
        if 'MACD_Signal' not in df.columns:
            df['MACD_Signal'] = signal
        if 'MACD_Hist' not in df.columns:
            df['MACD_Hist'] = hist
    
    # Características de tendencia
    df['EMACrossover'] = np.where(
        (df['FastEMA'] > df['SlowEMA']) & (df['FastEMA'].shift(1) <= df['SlowEMA'].shift(1)), 
        1, 
        np.where(
            (df['FastEMA'] < df['SlowEMA']) & (df['FastEMA'].shift(1) >= df['SlowEMA'].shift(1)),
            -1, 
            0
        )
    )
    
    df['PriceToSlowEMA'] = (df['Close'] - df['SlowEMA']) / df['SlowEMA'] * 100
    
    # Características de volatilidad
    df['RangePercent'] = (df['High'] - df['Low']) / df['Low'] * 100
    
    # Eliminar filas con valores NaN que se hayan creado
    df = df.dropna()
    
    return df

def detect_patterns(data):
    """
    Detecta patrones de reversión en los datos proporcionados.
    
    Args:
        data (pandas.DataFrame): DataFrame con datos OHLCV e indicadores
        
    Returns:
        list: Lista de patrones detectados con información de cada uno
    """
    # Crear una copia para no modificar el original
    df = data.copy()
    
    # Asegurarse de que tenemos las columnas necesarias
    if 'SwingHigh' not in df.columns or 'SwingLow' not in df.columns:
        df = create_features(df)
    
    patterns = []
    
    # Detectar Double Top
    double_tops = detect_double_top(df)
    for pattern in double_tops:
        patterns.append(pattern)
    
    # Detectar Double Bottom
    double_bottoms = detect_double_bottom(df)
    for pattern in double_bottoms:
        patterns.append(pattern)
    
    # Detectar Head and Shoulders
    head_shoulders = detect_head_and_shoulders(df)
    for pattern in head_shoulders:
        patterns.append(pattern)
    
    # Detectar Inverse Head and Shoulders
    inv_head_shoulders = detect_inverse_head_and_shoulders(df)
    for pattern in inv_head_shoulders:
        patterns.append(pattern)
    
    # Ordenar por confianza
    patterns.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    return patterns

def detect_swing_highs(df, window=5):
    """Detecta swing highs (máximos locales)"""
    highs = df['High'].values
    swing_highs = np.zeros(len(highs))
    
    for i in range(window, len(highs) - window):
        left_window = highs[i-window:i]
        right_window = highs[i+1:i+window+1]
        
        if highs[i] > np.max(left_window) and highs[i] > np.max(right_window):
            swing_highs[i] = 1
    
    return swing_highs

def detect_swing_lows(df, window=5):
    """Detecta swing lows (mínimos locales)"""
    lows = df['Low'].values
    swing_lows = np.zeros(len(lows))
    
    for i in range(window, len(lows) - window):
        left_window = lows[i-window:i]
        right_window = lows[i+1:i+window+1]
        
        if lows[i] < np.min(left_window) and lows[i] < np.min(right_window):
            swing_lows[i] = 1
    
    return swing_lows

def calculate_rsi(series, window=14):
    """Calcula el RSI (Relative Strength Index)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_atr(df, window=14):
    """Calcula el ATR (Average True Range)"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    return atr

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calcula el MACD (Moving Average Convergence Divergence)"""
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    
    macd = fast_ema - slow_ema
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    
    return macd, macd_signal, macd_hist

def detect_double_top(df):
    """Detecta patrones de doble techo"""
    patterns = []
    
    # Obtener índices de swing highs
    swing_high_indices = np.where(df['SwingHigh'] == 1)[0]
    
    # Necesitamos al menos dos swing highs para formar un doble techo
    if len(swing_high_indices) < 2:
        return patterns
    
    # Analizar pares de swing highs
    for i in range(1, len(swing_high_indices)):
        idx1 = swing_high_indices[i-1]
        idx2 = swing_high_indices[i]
        
        # Verificar que no estén demasiado cerca
        if idx2 - idx1 < 5:
            continue
        
        # Verificar que los dos picos estén a niveles similares
        price_diff_percent = abs(df['High'].iloc[idx1] - df['High'].iloc[idx2]) / df['High'].iloc[idx1]
        if price_diff_percent > 0.03:  # 3% de diferencia máxima
            continue
        
        # Verificar que hay una tendencia alcista previa al primer pico
        pre_pattern_idx = max(0, idx1 - 5)
        uptrend = df['Close'].iloc[idx1] > df['Close'].iloc[pre_pattern_idx]
        if not uptrend:
            continue
        
        # Verificar que hay un valle entre los dos picos
        between_indices = list(range(idx1 + 1, idx2))
        if not between_indices:
            continue
        
        valley_idx = between_indices[df['Low'].iloc[between_indices].argmin()]
        neckline = df['Low'].iloc[valley_idx]
        
        # Verificar si hay una ruptura de la neckline después del segundo pico
        post_indices = list(range(idx2 + 1, min(idx2 + 10, len(df))))
        if not post_indices:
            continue
        
        # Buscar ruptura de la neckline
        for post_idx in post_indices:
            if df['Close'].iloc[post_idx] < neckline:
                # Patrón confirmado
                confidence = 0.8 - price_diff_percent * 10  # Ajustar confianza basada en similitud de picos
                
                # Calcular niveles de entrada y Stop Loss/Take Profit
                entry_price = df['Close'].iloc[post_idx]
                stop_loss = max(df['High'].iloc[idx1], df['High'].iloc[idx2]) + df['ATR'].iloc[post_idx]
                take_profit = entry_price - (2 * (stop_loss - entry_price))  # Ratio riesgo/recompensa 1:2
                
                patterns.append({
                    'pattern': 'DOUBLE_TOP',
                    'confidence': confidence,
                    'high1_idx': idx1,
                    'high2_idx': idx2,
                    'neckline': neckline,
                    'confirmation_idx': post_idx,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_reward': abs((take_profit - entry_price) / (stop_loss - entry_price)) if stop_loss != entry_price else 0
                })
                break
    
    return patterns

def detect_double_bottom(df):
    """Detecta patrones de doble suelo"""
    patterns = []
    
    # Obtener índices de swing lows
    swing_low_indices = np.where(df['SwingLow'] == 1)[0]
    
    # Necesitamos al menos dos swing lows para formar un doble suelo
    if len(swing_low_indices) < 2:
        return patterns
    
    # Analizar pares de swing lows
    for i in range(1, len(swing_low_indices)):
        idx1 = swing_low_indices[i-1]
        idx2 = swing_low_indices[i]
        
        # Verificar que no estén demasiado cerca
        if idx2 - idx1 < 5:
            continue
        
        # Verificar que los dos valles estén a niveles similares
        price_diff_percent = abs(df['Low'].iloc[idx1] - df['Low'].iloc[idx2]) / df['Low'].iloc[idx1]
        if price_diff_percent > 0.03:  # 3% de diferencia máxima
            continue
        
        # Verificar que hay una tendencia bajista previa al primer valle
        pre_pattern_idx = max(0, idx1 - 5)
        downtrend = df['Close'].iloc[idx1] < df['Close'].iloc[pre_pattern_idx]
        if not downtrend:
            continue
        
        # Verificar que hay un pico entre los dos valles
        between_indices = list(range(idx1 + 1, idx2))
        if not between_indices:
            continue
        
        peak_idx = between_indices[df['High'].iloc[between_indices].argmax()]
        neckline = df['High'].iloc[peak_idx]
        
        # Verificar si hay una ruptura de la neckline después del segundo valle
        post_indices = list(range(idx2 + 1, min(idx2 + 10, len(df))))
        if not post_indices:
            continue
        
        # Buscar ruptura de la neckline
        for post_idx in post_indices:
            if df['Close'].iloc[post_idx] > neckline:
                # Patrón confirmado
                confidence = 0.8 - price_diff_percent * 10  # Ajustar confianza basada en similitud de valles
                
                # Calcular niveles de entrada y Stop Loss/Take Profit
                entry_price = df['Close'].iloc[post_idx]
                stop_loss = min(df['Low'].iloc[idx1], df['Low'].iloc[idx2]) - df['ATR'].iloc[post_idx]
                take_profit = entry_price + (2 * (entry_price - stop_loss))  # Ratio riesgo/recompensa 1:2
                
                patterns.append({
                    'pattern': 'DOUBLE_BOTTOM',
                    'confidence': confidence,
                    'low1_idx': idx1,
                    'low2_idx': idx2,
                    'neckline': neckline,
                    'confirmation_idx': post_idx,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_reward': abs((take_profit - entry_price) / (entry_price - stop_loss)) if entry_price != stop_loss else 0
                })
                break
    
    return patterns

def detect_head_and_shoulders(df):
    """Detecta patrones de cabeza y hombros"""
    patterns = []
    
    # Obtener índices de swing highs
    swing_high_indices = np.where(df['SwingHigh'] == 1)[0]
    
    # Necesitamos al menos tres swing highs para formar cabeza y hombros
    if len(swing_high_indices) < 3:
        return patterns
    
    # Analizar tríos de swing highs
    for i in range(2, len(swing_high_indices)):
        left_idx = swing_high_indices[i-2]
        head_idx = swing_high_indices[i-1]
        right_idx = swing_high_indices[i]
        
        # Verificar espaciado adecuado
        if head_idx - left_idx < 3 or right_idx - head_idx < 3:
            continue
        
        # Verificar que la cabeza es más alta que los hombros
        head_price = df['High'].iloc[head_idx]
        left_price = df['High'].iloc[left_idx]
        right_price = df['High'].iloc[right_idx]
        
        if not (head_price > left_price and head_price > right_price):
            continue
        
        # Verificar que los hombros están a niveles similares
        shoulder_diff = abs(left_price - right_price) / left_price
        if shoulder_diff > 0.05:  # 5% de diferencia máxima
            continue
        
        # Encontrar los valles entre los picos para la neckline
        left_valley_indices = list(range(left_idx + 1, head_idx))
        right_valley_indices = list(range(head_idx + 1, right_idx))
        
        if not left_valley_indices or not right_valley_indices:
            continue
        
        left_valley_idx = left_valley_indices[df['Low'].iloc[left_valley_indices].argmin()]
        right_valley_idx = right_valley_indices[df['Low'].iloc[right_valley_indices].argmin()]
        
        left_valley = df['Low'].iloc[left_valley_idx]
        right_valley = df['Low'].iloc[right_valley_idx]
        
        # La neckline debe ser aproximadamente horizontal
        neckline_diff = abs(left_valley - right_valley) / left_valley
        if neckline_diff > 0.03:  # 3% de diferencia máxima
            continue
        
        # Usar el promedio para la neckline
        neckline = (left_valley + right_valley) / 2
        
        # Verificar si hay una ruptura de la neckline después del hombro derecho
        post_indices = list(range(right_idx + 1, min(right_idx + 10, len(df))))
        if not post_indices:
            continue
        
        # Buscar ruptura de la neckline
        for post_idx in post_indices:
            if df['Close'].iloc[post_idx] < neckline:
                # Patrón confirmado
                # Calcular confianza basada en la calidad del patrón
                head_height = head_price - neckline
                pattern_quality = (head_height / head_price) * (1 - shoulder_diff) * (1 - neckline_diff)
                confidence = min(0.9, pattern_quality * 5)
                
                # Calcular niveles de entrada y Stop Loss/Take Profit
                entry_price = df['Close'].iloc[post_idx]
                stop_loss = head_price + df['ATR'].iloc[post_idx]
                price_target = entry_price - head_height  # Objetivo basado en la altura del patrón
                take_profit = price_target
                
                patterns.append({
                    'pattern': 'HEAD_AND_SHOULDERS',
                    'confidence': confidence,
                    'left_shoulder_idx': left_idx,
                    'head_idx': head_idx,
                    'right_shoulder_idx': right_idx,
                    'neckline': neckline,
                    'confirmation_idx': post_idx,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_reward': abs((take_profit - entry_price) / (stop_loss - entry_price)) if stop_loss != entry_price else 0
                })
                break
    
    return patterns

def detect_inverse_head_and_shoulders(df):
    """Detecta patrones de cabeza y hombros invertidos"""
    patterns = []
    
    # Obtener índices de swing lows
    swing_low_indices = np.where(df['SwingLow'] == 1)[0]
    
    # Necesitamos al menos tres swing lows para formar cabeza y hombros invertidos
    if len(swing_low_indices) < 3:
        return patterns
    
    # Analizar tríos de swing lows
    for i in range(2, len(swing_low_indices)):
        left_idx = swing_low_indices[i-2]
        head_idx = swing_low_indices[i-1]
        right_idx = swing_low_indices[i]
        
        # Verificar espaciado adecuado
        if head_idx - left_idx < 3 or right_idx - head_idx < 3:
            continue
        
        # Verificar que la cabeza es más baja que los hombros
        head_price = df['Low'].iloc[head_idx]
        left_price = df['Low'].iloc[left_idx]
        right_price = df['Low'].iloc[right_idx]
        
        if not (head_price < left_price and head_price < right_price):
            continue
        
        # Verificar que los hombros están a niveles similares
        shoulder_diff = abs(left_price - right_price) / left_price
        if shoulder_diff > 0.05:  # 5% de diferencia máxima
            continue
        
        # Encontrar los picos entre los valles para la neckline
        left_peak_indices = list(range(left_idx + 1, head_idx))
        right_peak_indices = list(range(head_idx + 1, right_idx))
        
        if not left_peak_indices or not right_peak_indices:
            continue
        
        left_peak_idx = left_peak_indices[df['High'].iloc[left_peak_indices].argmax()]
        right_peak_idx = right_peak_indices[df['High'].iloc[right_peak_indices].argmax()]
        
        left_peak = df['High'].iloc[left_peak_idx]
        right_peak = df['High'].iloc[right_peak_idx]
        
        # La neckline debe ser aproximadamente horizontal
        neckline_diff = abs(left_peak - right_peak) / left_peak
        if neckline_diff > 0.03:  # 3% de diferencia máxima
            continue
        
        # Usar el promedio para la neckline
        neckline = (left_peak + right_peak) / 2
        
        # Verificar si hay una ruptura de la neckline después del hombro derecho
        post_indices = list(range(right_idx + 1, min(right_idx + 10, len(df))))
        if not post_indices:
            continue
        
        # Buscar ruptura de la neckline
        for post_idx in post_indices:
            if df['Close'].iloc[post_idx] > neckline:
                # Patrón confirmado
                # Calcular confianza basada en la calidad del patrón
                head_depth = neckline - head_price
                pattern_quality = (head_depth / neckline) * (1 - shoulder_diff) * (1 - neckline_diff)
                confidence = min(0.9, pattern_quality * 5)
                
                # Calcular niveles de entrada y Stop Loss/Take Profit
                entry_price = df['Close'].iloc[post_idx]
                stop_loss = head_price - df['ATR'].iloc[post_idx]
                price_target = entry_price + head_depth  # Objetivo basado en la profundidad del patrón
                take_profit = price_target
                
                patterns.append({
                    'pattern': 'INV_HEAD_AND_SHOULDERS',
                    'confidence': confidence,
                    'left_shoulder_idx': left_idx,
                    'head_idx': head_idx,
                    'right_shoulder_idx': right_idx,
                    'neckline': neckline,
                    'confirmation_idx': post_idx,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_reward': abs((take_profit - entry_price) / (entry_price - stop_loss)) if entry_price != stop_loss else 0
                })
                break
    
    return patterns