import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import os

def set_style():
    """Configura el estilo de visualización."""
    plt.style.use('ggplot')
    sns.set_style('darkgrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['lines.linewidth'] = 1.5

def plot_price_data(data: pd.DataFrame, title: str = 'Price Chart', 
                   ma_periods: List[int] = None, 
                   save_path: Optional[str] = None):
    """
    Visualiza datos de precios con medias móviles opcionales.
    
    Args:
        data: DataFrame con datos de precios (debe incluir 'Open', 'High', 'Low', 'Close')
        title: Título del gráfico
        ma_periods: Lista de períodos para medias móviles
        save_path: Ruta para guardar el gráfico (None para mostrar)
    """
    set_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(title, fontsize=16)
    
    # Configurar índice de tiempo
    x = data.index
    if not isinstance(x, pd.DatetimeIndex):
        try:
            x = pd.to_datetime(x)
        except:
            x = range(len(data))
    
    # Gráfico de velas
    candlestick_width = 0.6
    up_color = '#66BB6A'
    down_color = '#EF5350'
    
    for i, (idx, row) in enumerate(data.iterrows()):
        color = up_color if row['Close'] >= row['Open'] else down_color
        ax1.add_patch(patches.Rectangle(
            (i - candlestick_width/2, row['Low']),
            candlestick_width,
            row['High'] - row['Low'],
            fill=False,
            edgecolor=color,
            linewidth=1
        ))
        ax1.add_patch(patches.Rectangle(
            (i - candlestick_width/2, min(row['Open'], row['Close'])),
            candlestick_width,
            abs(row['Close'] - row['Open']),
            fill=True,
            facecolor=color,
            edgecolor=color,
            alpha=0.8
        ))
    
    # Añadir medias móviles
    if ma_periods:
        for period in ma_periods:
            ma_name = f'SMA_{period}'
            if ma_name not in data.columns:
                data[ma_name] = data['Close'].rolling(window=period).mean()
            ax1.plot(range(len(data)), data[ma_name], label=f'SMA {period}')
    
    # Añadir EMAs si existen
    if 'FastEMA' in data.columns and 'SlowEMA' in data.columns:
        ax1.plot(range(len(data)), data['FastEMA'], label='Fast EMA', color='#FF5722')
        ax1.plot(range(len(data)), data['SlowEMA'], label='Slow EMA', color='#2196F3')
    
    # Volumen en el eje inferior
    volume_data = data['Volume'] if 'Volume' in data.columns else pd.Series(0, index=data.index)
    ax2.bar(range(len(data)), volume_data, width=0.8, alpha=0.5, color='#90A4AE')
    
    # Configurar leyenda y etiquetas
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax2.set_ylabel('Volume')
    
    # Configurar formato de eje x para fechas
    if isinstance(data.index, pd.DatetimeIndex):
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax2.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
    else:
        # Usar índices numéricos y mostrar algunos ticks
        ticks = np.linspace(0, len(data) - 1, min(10, len(data)))
        ax1.set_xticks(ticks)
        ax2.set_xticks(ticks)
        
        if len(data) > 10:
            ax1.set_xticklabels([])
    
    # Ajustar límites de los ejes
    ax1.set_xlim(-1, len(data))
    ax2.set_xlim(-1, len(data))
    
    plt.tight_layout()
    
    # Guardar o mostrar
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pattern_detection(data: pd.DataFrame, pattern_info: Dict, 
                         window_size: int = 50, title: Optional[str] = None,
                         save_path: Optional[str] = None):
    """
    Visualiza un patrón de precios detectado.
    
    Args:
        data: DataFrame con datos de precios
        pattern_info: Diccionario con información del patrón
        window_size: Tamaño de la ventana a mostrar
        title: Título del gráfico (None para automático)
        save_path: Ruta para guardar el gráfico (None para mostrar)
    """
    set_style()
    
    # Obtener información del patrón
    pattern_name = pattern_info.get('pattern', 'Unknown Pattern')
    confidence = pattern_info.get('confidence', 0.0)
    
    # Crear título si no se proporciona
    if title is None:
        title = f"{pattern_name} Pattern (Confidence: {confidence:.2f})"
    
    # Obtener datos recientes
    if len(data) > window_size:
        plot_data = data.iloc[-window_size:].copy()
    else:
        plot_data = data.copy()
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Dibujar velas
    candlestick_width = 0.6
    up_color = '#66BB6A'
    down_color = '#EF5350'
    
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        color = up_color if row['Close'] >= row['Open'] else down_color
        ax.add_patch(patches.Rectangle(
            (i - candlestick_width/2, row['Low']),
            candlestick_width,
            row['High'] - row['Low'],
            fill=False,
            edgecolor=color,
            linewidth=1
        ))
        ax.add_patch(patches.Rectangle(
            (i - candlestick_width/2, min(row['Open'], row['Close'])),
            candlestick_width,
            abs(row['Close'] - row['Open']),
            fill=True,
            facecolor=color,
            edgecolor=color,
            alpha=0.8
        ))
    
    # Añadir EMAs si existen
    if 'FastEMA' in plot_data.columns and 'SlowEMA' in plot_data.columns:
        ax.plot(range(len(plot_data)), plot_data['FastEMA'], label='Fast EMA', color='#FF5722')
        ax.plot(range(len(plot_data)), plot_data['SlowEMA'], label='Slow EMA', color='#2196F3')
    
    # Marcar puntos clave del patrón
    if pattern_name in ['DOUBLE_TOP', 'DOUBLE_BOTTOM']:
        # Marcar los dos picos/valles
        if 'high1_idx' in pattern_info and 'high2_idx' in pattern_info:  # Double Top
            high1_pos = plot_data.index.get_loc(pattern_info['high1_idx']) if pattern_info['high1_idx'] in plot_data.index else None
            high2_pos = plot_data.index.get_loc(pattern_info['high2_idx']) if pattern_info['high2_idx'] in plot_data.index else None
            
            if high1_pos is not None and high2_pos is not None:
                high1_val = plot_data.iloc[high1_pos]['High']
                high2_val = plot_data.iloc[high2_pos]['High']
                
                ax.plot(high1_pos, high1_val, 'ro', markersize=10, label='First Peak')
                ax.plot(high2_pos, high2_val, 'ro', markersize=10, label='Second Peak')
                
                if 'neckline' in pattern_info:
                    neckline = pattern_info['neckline']
                    ax.axhline(y=neckline, color='r', linestyle='--', alpha=0.7, label='Neckline')
                
                # Proyección
                if 'entry_price' in pattern_info and 'take_profit' in pattern_info:
                    ax.axhline(y=pattern_info['entry_price'], color='g', linestyle='-', alpha=0.7, label='Entry')
                    ax.axhline(y=pattern_info['take_profit'], color='b', linestyle='-', alpha=0.7, label='Target')
                    ax.axhline(y=pattern_info['stop_loss'], color='r', linestyle='-', alpha=0.7, label='Stop Loss')
        
        elif 'low1_idx' in pattern_info and 'low2_idx' in pattern_info:  # Double Bottom
            low1_pos = plot_data.index.get_loc(pattern_info['low1_idx']) if pattern_info['low1_idx'] in plot_data.index else None
            low2_pos = plot_data.index.get_loc(pattern_info['low2_idx']) if pattern_info['low2_idx'] in plot_data.index else None
            
            if low1_pos is not None and low2_pos is not None:
                low1_val = plot_data.iloc[low1_pos]['Low']
                low2_val = plot_data.iloc[low2_pos]['Low']
                
                ax.plot(low1_pos, low1_val, 'go', markersize=10, label='First Valley')
                ax.plot(low2_pos, low2_val, 'go', markersize=10, label='Second Valley')
                
                # Dibujar neckline
                if 'neckline' in pattern_info:
                    neckline = pattern_info['neckline']
                    ax.axhline(y=neckline, color='g', linestyle='--', alpha=0.7, label='Neckline')
                
                # Proyección
                if 'entry_price' in pattern_info and 'take_profit' in pattern_info:
                    ax.axhline(y=pattern_info['entry_price'], color='g', linestyle='-', alpha=0.7, label='Entry')
                    ax.axhline(y=pattern_info['take_profit'], color='b', linestyle='-', alpha=0.7, label='Target')
                    ax.axhline(y=pattern_info['stop_loss'], color='r', linestyle='-', alpha=0.7, label='Stop Loss')
    
    elif pattern_name in ['HEAD_AND_SHOULDERS', 'INV_HEAD_AND_SHOULDERS']:
        # Marcar los tres picos/valles
        if 'left_shoulder_idx' in pattern_info and 'head_idx' in pattern_info and 'right_shoulder_idx' in pattern_info:
            left_pos = plot_data.index.get_loc(pattern_info['left_shoulder_idx']) if pattern_info['left_shoulder_idx'] in plot_data.index else None
            head_pos = plot_data.index.get_loc(pattern_info['head_idx']) if pattern_info['head_idx'] in plot_data.index else None
            right_pos = plot_data.index.get_loc(pattern_info['right_shoulder_idx']) if pattern_info['right_shoulder_idx'] in plot_data.index else None
            
            if left_pos is not None and head_pos is not None and right_pos is not None:
                if pattern_name == 'HEAD_AND_SHOULDERS':
                    left_val = plot_data.iloc[left_pos]['High']
                    head_val = plot_data.iloc[head_pos]['High']
                    right_val = plot_data.iloc[right_pos]['High']
                    
                    ax.plot(left_pos, left_val, 'ro', markersize=10, label='Left Shoulder')
                    ax.plot(head_pos, head_val, 'mo', markersize=10, label='Head')
                    ax.plot(right_pos, right_val, 'ro', markersize=10, label='Right Shoulder')
                else:  # INV_HEAD_AND_SHOULDERS
                    left_val = plot_data.iloc[left_pos]['Low']
                    head_val = plot_data.iloc[head_pos]['Low']
                    right_val = plot_data.iloc[right_pos]['Low']
                    
                    ax.plot(left_pos, left_val, 'go', markersize=10, label='Left Shoulder')
                    ax.plot(head_pos, head_val, 'mo', markersize=10, label='Head')
                    ax.plot(right_pos, right_val, 'go', markersize=10, label='Right Shoulder')
                
                # Dibujar neckline
                if 'neckline' in pattern_info:
                    neckline = pattern_info['neckline']
                    line_color = 'r' if pattern_name == 'HEAD_AND_SHOULDERS' else 'g'
                    ax.axhline(y=neckline, color=line_color, linestyle='--', alpha=0.7, label='Neckline')
                
                # Proyección
                if 'entry_price' in pattern_info and 'take_profit' in pattern_info:
                    ax.axhline(y=pattern_info['entry_price'], color='g', linestyle='-', alpha=0.7, label='Entry')
                    ax.axhline(y=pattern_info['take_profit'], color='b', linestyle='-', alpha=0.7, label='Target')
                    ax.axhline(y=pattern_info['stop_loss'], color='r', linestyle='-', alpha=0.7, label='Stop Loss')
    
    # Configurar títulos y etiquetas
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    
    # Configurar formato de eje x para fechas
    if isinstance(plot_data.index, pd.DatetimeIndex):
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
    
    # Agregar leyenda
    ax.legend(loc='best')
    
    # Ajustar los límites del eje y para incluir proyecciones
    if 'take_profit' in pattern_info and 'stop_loss' in pattern_info:
        min_price = min(plot_data['Low'].min(), pattern_info['stop_loss'])
        max_price = max(plot_data['High'].max(), pattern_info['take_profit'])
        
        margin = (max_price - min_price) * 0.1
        ax.set_ylim(min_price - margin, max_price + margin)
    
    plt.tight_layout()
    
    # Guardar o mostrar
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_trades(data: pd.DataFrame, trades: List[Dict], title: str = 'Trading Performance',
               save_path: Optional[str] = None):
    """
    Visualiza operaciones de trading sobre los datos de precios.
    
    Args:
        data: DataFrame con datos de precios
        trades: Lista de diccionarios con información de trades
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico (None para mostrar)
    """
    set_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(title, fontsize=16)
    
    # Configurar índice de tiempo
    x = data.index
    if not isinstance(x, pd.DatetimeIndex):
        try:
            x = pd.to_datetime(x)
        except:
            x = range(len(data))
    
    # Gráfico de precios
    ax1.plot(range(len(data)), data['Close'], label='Close', color='#424242', alpha=0.5)
    
    # Añadir EMAs si existen
    if 'FastEMA' in data.columns and 'SlowEMA' in data.columns:
        ax1.plot(range(len(data)), data['FastEMA'], label='Fast EMA', color='#FF5722')
        ax1.plot(range(len(data)), data['SlowEMA'], label='Slow EMA', color='#2196F3')
    
    # Marcar operaciones
    entry_markers = []
    exit_markers = []
    colors = []
    patterns = []
    pnls = []
    annotations = []
    
    # Identificar entradas y salidas en los datos
    for trade in trades:
        # Buscar entrada en los datos
        if 'entry_time' in trade and trade['entry_time'] in data.index:
            entry_idx = data.index.get_loc(trade['entry_time'])
            entry_price = trade['entry_price']
            
            # Buscar salida en los datos
            if 'exit_time' in trade and trade['exit_time'] in data.index:
                exit_idx = data.index.get_loc(trade['exit_time'])
                exit_price = trade['exit_price']
                
                # Determinar si es ganador o perdedor
                pnl = trade.get('pnl', 0)
                color = 'g' if pnl > 0 else 'r'
                
                # Agregar a las listas para dibujo
                entry_markers.append((entry_idx, entry_price))
                exit_markers.append((exit_idx, exit_price))
                colors.append(color)
                patterns.append(trade.get('pattern', 'Unknown'))
                pnls.append(pnl)
                
                # Agregar línea de conexión
                ax1.plot([entry_idx, exit_idx], [entry_price, exit_price], color=color, alpha=0.7)
                
                # Anotar PnL
                annotations.append((exit_idx, exit_price, f"${pnl:.2f}"))
    
    # Dibujar marcadores de entrada y salida
    for (entry_idx, entry_price), (exit_idx, exit_price), color, pattern in zip(entry_markers, exit_markers, colors, patterns):
        ax1.plot(entry_idx, entry_price, 'o', color=color, markersize=8, alpha=0.8)
        ax1.plot(exit_idx, exit_price, 's', color=color, markersize=8, alpha=0.8)
        
        # Anotar patrón en la entrada
        ax1.annotate(pattern, (entry_idx, entry_price), xytext=(5, 5),
                     textcoords='offset points', fontsize=8)
    
    # Anotar PnL en las salidas
    for exit_idx, exit_price, annotation in annotations:
        ax1.annotate(annotation, (exit_idx, exit_price), xytext=(5, -15),
                    textcoords='offset points', fontsize=8)
    
    # Gráfico de PnL acumulado
    cumulative_pnl = np.cumsum(pnls)
    if len(cumulative_pnl) > 0:
        # Crear índices para los puntos de PnL
        pnl_indices = [exit_idx for entry_idx, entry_price in entry_markers]
        
        # Interpolar para tener PnL para cada barra
        full_pnl = np.zeros(len(data))
        full_pnl[pnl_indices] = pnls
        cumulative_full_pnl = np.cumsum(full_pnl)
        
        # Dibujar PnL acumulado
        ax2.plot(range(len(data)), cumulative_full_pnl, color='#2196F3', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        # Rellenar áreas positivas y negativas
        ax2.fill_between(range(len(data)), cumulative_full_pnl, 0, 
                         where=cumulative_full_pnl >= 0, color='g', alpha=0.3)
        ax2.fill_between(range(len(data)), cumulative_full_pnl, 0, 
                         where=cumulative_full_pnl < 0, color='r', alpha=0.3)
    
    # Configurar etiquetas y leyendas
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax2.set_ylabel('Cumulative P&L ($)')
    ax2.set_xlabel('Time')
    
    # Configurar formato de eje x para fechas
    if isinstance(data.index, pd.DatetimeIndex):
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax2.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
    else:
        # Usar índices numéricos y mostrar algunos ticks
        ticks = np.linspace(0, len(data) - 1, min(10, len(data)))
        ax1.set_xticks(ticks)
        ax2.set_xticks(ticks)
        
        if len(data) > 10:
            ax1.set_xticklabels([])
    
    # Mostrar estadísticas de trading
    if len(pnls) > 0:
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        avg_win = sum(p for p in pnls if p > 0) / sum(1 for p in pnls if p > 0) if any(p > 0 for p in pnls) else 0
        avg_loss = sum(p for p in pnls if p < 0) / sum(1 for p in pnls if p < 0) if any(p < 0 for p in pnls) else 0
        profit_factor = abs(sum(p for p in pnls if p > 0) / sum(p for p in pnls if p < 0)) if sum(p for p in pnls if p < 0) != 0 else float('inf')
        
        stats_text = (f"Total Trades: {len(pnls)}\n"
                     f"Win Rate: {win_rate:.1f}%\n"
                     f"Avg Win: ${avg_win:.2f}\n"
                     f"Avg Loss: ${avg_loss:.2f}\n"
                     f"Profit Factor: {profit_factor:.2f}\n"
                     f"Net P&L: ${sum(pnls):.2f}")
        
        # Colocar texto de estadísticas
        ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Guardar o mostrar
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pattern_distribution(patterns: List[Dict], title: str = 'Pattern Distribution Analysis',
                            save_path: Optional[str] = None):
    """
    Visualiza distribución y rendimiento de patrones detectados.
    
    Args:
        patterns: Lista de diccionarios con información de patrones
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico (None para mostrar)
    """
    set_style()
    
    # Extraer datos de patrones
    pattern_types = [p.get('pattern', 'Unknown') for p in patterns]
    confidences = [p.get('confidence', 0) for p in patterns]
    pnls = [p.get('pnl', 0) for p in patterns if 'pnl' in p]
    
    # Contar patrones por tipo
    pattern_counts = {}
    for pattern in pattern_types:
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    # Calcular rendimiento por tipo de patrón
    pattern_performance = {}
    for i, pattern in enumerate(pattern_types):
        if i < len(pnls):
            if pattern not in pattern_performance:
                pattern_performance[pattern] = []
            pattern_performance[pattern].append(pnls[i])
    
    # Calcular medias de rendimiento
    pattern_avg_pnl = {pattern: sum(pnls)/len(pnls) for pattern, pnls in pattern_performance.items() if pnls}
    
    # Crear figura con subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))
    fig.suptitle(title, fontsize=16)
    
    # 1. Distribución de patrones
    patterns_df = pd.DataFrame({'Pattern': list(pattern_counts.keys()), 
                               'Count': list(pattern_counts.values())})
    patterns_df = patterns_df.sort_values('Count', ascending=False)
    
    sns.barplot(x='Pattern', y='Count', data=patterns_df, ax=ax1, palette='viridis')
    ax1.set_title('Pattern Distribution')
    ax1.set_xlabel('Pattern Type')
    ax1.set_ylabel('Count')
    
    # Rotar etiquetas si hay muchas
    if len(patterns_df) > 5:
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Distribución de confianza
    sns.histplot(confidences, kde=True, ax=ax2, bins=20, color='#2196F3')
    ax2.set_title('Pattern Confidence Distribution')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    
    # 3. Rendimiento por patrón
    if pattern_avg_pnl:
        performance_df = pd.DataFrame({'Pattern': list(pattern_avg_pnl.keys()),
                                      'Average PnL': list(pattern_avg_pnl.values())})
        performance_df = performance_df.sort_values('Average PnL', ascending=False)
        
        bars = sns.barplot(x='Pattern', y='Average PnL', data=performance_df, ax=ax3, palette='coolwarm')
        
        # Colorear según rendimiento
        for i, bar in enumerate(bars.patches):
            if bar.get_height() < 0:
                bar.set_color('#EF5350')
            else:
                bar.set_color('#66BB6A')
        
        ax3.set_title('Average Performance by Pattern')
        ax3.set_xlabel('Pattern Type')
        ax3.set_ylabel('Average P&L ($)')
        
        # Línea de cero
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Rotar etiquetas si hay muchas
        if len(performance_df) > 5:
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Guardar o mostrar
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()