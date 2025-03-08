import argparse
import logging
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import json
import neurevo
from neurevo_trading.agents.neurevo_agent import NeurEvoTradingAgent
from neurevo_trading.environment.neurevo_adapter import NeurEvoEnvironmentAdapter

from neurevo_trading.agents.pattern_detector import PatternDetector
from neurevo_trading.environment.trading_env import TradingEnvironment
from neurevo_trading.environment.data_processor import DataProcessor
from neurevo_trading.utils.visualization import plot_trades, plot_pattern_detection, plot_pattern_distribution
from neurevo_trading.utils.feature_engineering import create_features, detect_patterns

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger("backtest")

class PatternTraderBacktest:
    """
    Backtester para estrategias de trading basadas en patrones de reversión.
    """
    
    def __init__(self, config_file=None, initial_capital=10000, 
                 commission=0.0, slippage=0.0, risk_per_trade=0.02,
                 min_confidence=0.7, min_risk_reward=1.5,
                 model_path=None):
        """
        Inicializa el backtester.
        
        Args:
            config_file: Ruta al archivo de configuración
            initial_capital: Capital inicial
            commission: Comisión por operación (porcentaje)
            slippage: Deslizamiento por operación (porcentaje)
            risk_per_trade: Porcentaje de capital a arriesgar por operación
            min_confidence: Confianza mínima para considerar un patrón válido
            min_risk_reward: Ratio mínimo de riesgo/recompensa
            model_path: Ruta al modelo entrenado de detección de patrones
        """
        # Cargar configuración si existe
        self.config = self._load_config(config_file)
        
        # Usar valores de configuración si existen, de lo contrario usar los parámetros
        self.initial_capital = self.config.get("initial_capital", initial_capital)
        self.commission = self.config.get("commission", commission)
        self.slippage = self.config.get("slippage", slippage)
        self.risk_per_trade = self.config.get("risk_per_trade", risk_per_trade)
        self.min_confidence = self.config.get("min_confidence", min_confidence)
        self.min_risk_reward = self.config.get("min_risk_reward", min_risk_reward)
        self.model_path = self.config.get("model_path", model_path)
        
        # Inicializar procesador de datos
        self.data_processor = DataProcessor(logger=logger)
        
        # Inicializar detector de patrones
        self.pattern_detector = PatternDetector(
            input_channels=5,  # OHLCV por defecto
            window_size=50
        )
        
        # Cargar modelo si se especifica
        if self.model_path and os.path.exists(self.model_path):
            self.pattern_detector.load(self.model_path)
            logger.info(f"Modelo de detección de patrones cargado desde {self.model_path}")
        
        # Resultados del backtest
        self.trades = []
        self.pattern_detections = []
        self.equity_curve = None
        self.statistics = {}
    
    def run_with_neurevo(self, data: pd.DataFrame, model_path=None, config=None, 
                    plot=True, save_results=True, results_dir="backtest_results"):
        """
        Ejecuta el backtest usando un cerebro NeurEvo.
        
        Args:
            data: DataFrame con datos de precios
            model_path: Ruta al modelo entrenado (opcional)
            config: Configuración para NeurEvo
            plot: Si es True, genera gráficos de resultados
            save_results: Si es True, guarda los resultados
            results_dir: Directorio para guardar resultados
            
        Returns:
            Dict con estadísticas del backtest
        """
        start_time = time.time()
        logger.info("Iniciando backtest con NeurEvo...")
        
        # Preparar datos
        self.data = self.data_processor.prepare_data(data, add_features=True)
        
        # Crear entorno de trading
        env = TradingEnvironment(
            data=self.data,
            window_size=self.window_size,
            initial_balance=self.initial_capital,
            commission=self.commission,
            slippage=self.slippage
        )
        
        # Configuración por defecto para NeurEvo
        default_config = {
            "hidden_layers": [256, 128, 64],
            "learning_rate": 0.00025,
            "batch_size": 64,
            "curiosity_weight": 0.1
        }
        
        if config:
            default_config.update(config)
        
        # Crear cerebro NeurEvo
        brain = neurevo.create_brain(default_config)
        
        # Crear adaptador para el entorno
        adapter = NeurEvoEnvironmentAdapter(env)
        env_id = adapter.register_with_brain(brain)
        
        # Crear agente
        agent_id = adapter.create_agent()
        
        # Cargar modelo si se proporciona
        if model_path and os.path.exists(model_path):
            logger.info(f"Cargando modelo desde {model_path}")
            brain.load(model_path)
        else:
            logger.warning("No se proporcionó modelo, los resultados serán aleatorios o basados en un modelo sin entrenar")
        
        # Ejecutar episodios de backtest
        rewards = []
        trades = []
        equity_curve = [self.initial_capital]
        
        # Ejecutar un episodio completo
        total_reward = adapter.run_episode(agent_id)
        rewards.append(total_reward)
        
        # Obtener información de trades
        self.trades = env.trade_history
        
        # Calcular equity curve
        for t in self.trades:
            equity_point = equity_curve[-1]
            if "pnl" in t:
                equity_point += t["pnl"]
            equity_curve.append(equity_point)
        
        # Calcular estadísticas de rendimiento
        self.statistics = self._calculate_statistics()
        
        # Guardar resultados
        if save_results:
            self._save_results(results_dir)
        
        # Generar gráficos
        if plot:
            self._generate_plots(results_dir if save_results else None)
        
        logger.info(f"Backtest con NeurEvo completado en {time.time() - start_time:.2f} segundos")
        logger.info(f"Estadísticas: {json.dumps(self.statistics, indent=2)}")
        
        return self.statistics
    
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
    
    def run(self, data: pd.DataFrame, plot=True, save_results=True, 
           results_dir="backtest_results"):
        """
        Ejecuta el backtest sobre los datos proporcionados.
        
        Args:
            data: DataFrame con datos de precios
            plot: Si es True, genera gráficos de resultados
            save_results: Si es True, guarda los resultados
            results_dir: Directorio para guardar resultados
            
        Returns:
            Dict con estadísticas del backtest
        """
        start_time = time.time()
        logger.info("Iniciando backtest...")
        
        # Preparar datos
        self.data = self.data_processor.prepare_data(data, add_features=True)
        
        # Inicializar variables de backtest
        self.capital = self.initial_capital
        self.position = 0  # 0 = sin posición, 1 = long, -1 = short
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.position_size = 0
        self.current_pattern = None
        
        self.trades = []
        self.pattern_detections = []
        self.equity_curve = [self.initial_capital]
        
        # Ventana mínima para detección de patrones
        window_size = 50
        
        # Recorrer los datos
        for i in range(window_size, len(self.data)):
            # Obtener datos hasta el punto actual
            current_data = self.data.iloc[:i+1].copy()
            
            # Verificar posición actual
            self._check_exits(i)
            
            # Si no hay posición abierta, buscar entradas
            if self.position == 0:
                self._check_entries(i)
            
            # Actualizar equity curve
            self.equity_curve.append(self._calculate_equity(i))
        
        # Cerrar posición final si existe
        if self.position != 0:
            self._close_position(len(self.data) - 1, "End of backtest")
        
        # Calcular estadísticas de rendimiento
        self.statistics = self._calculate_statistics()
        
        # Guardar resultados
        if save_results:
            self._save_results(results_dir)
        
        # Generar gráficos
        if plot:
            self._generate_plots(results_dir if save_results else None)
        
        logger.info(f"Backtest completado en {time.time() - start_time:.2f} segundos")
        logger.info(f"Estadísticas: {json.dumps(self.statistics, indent=2)}")
        
        return self.statistics
    
    def _check_entries(self, index: int):
        """
        Busca señales de entrada en el punto actual.
        
        Args:
            index: Índice actual en los datos
        """
        # Obtener datos recientes
        lookback = min(50, index)
        recent_data = self.data.iloc[index-lookback:index+1].copy()
        
        # Detectar patrones usando el detector
        pattern_info = self.pattern_detector.detect(recent_data)
        pattern_name = pattern_info.get("pattern_name", "NO_PATTERN")
        confidence = pattern_info.get("confidence", 0.0)
        
        # Si no se detectó con el detector neuronal, intentar con detección clásica
        if pattern_name == "NO_PATTERN" or confidence < self.min_confidence:
            patterns = detect_patterns(recent_data)
            if patterns:
                pattern_info = patterns[0]  # Usar el patrón con mayor confianza
                pattern_name = pattern_info.get("pattern", "NO_PATTERN")
                confidence = pattern_info.get("confidence", 0.0)
        
        # Registrar detección de patrón
        if pattern_name != "NO_PATTERN":
            self.pattern_detections.append({
                "index": index,
                "date": self.data.index[index],
                "pattern": pattern_name,
                "confidence": confidence,
                "price": self.data['Close'].iloc[index]
            })
        
        # Si se detectó un patrón con confianza suficiente
        if pattern_name != "NO_PATTERN" and confidence >= self.min_confidence:
            # Determinar dirección basada en el patrón
            signal = 0
            if pattern_name in ["DOUBLE_BOTTOM", "INV_HEAD_AND_SHOULDERS"]:
                signal = 1  # Long
            elif pattern_name in ["DOUBLE_TOP", "HEAD_AND_SHOULDERS"]:
                signal = -1  # Short
            
            # Si hay señal, configurar operación
            if signal != 0:
                # Obtener precio actual
                current_price = self.data['Close'].iloc[index]
                
                # Calcular stop loss y take profit
                if 'stop_loss' in pattern_info and 'take_profit' in pattern_info:
                    stop_loss = pattern_info['stop_loss']
                    take_profit = pattern_info['take_profit']
                else:
                    # Usar ATR para calcular SL/TP si no se proporcionan
                    atr = self.data['ATR'].iloc[index] if 'ATR' in self.data.columns else self.data['Close'].iloc[index] * 0.01
                    
                    if signal == 1:  # Long
                        stop_loss = current_price - 2 * atr
                        take_profit = current_price + 3 * atr
                    else:  # Short
                        stop_loss = current_price + 2 * atr
                        take_profit = current_price - 3 * atr
                
                # Calcular riesgo y ratio riesgo/recompensa
                risk = abs(current_price - stop_loss)
                reward = abs(take_profit - current_price)
                risk_reward = reward / risk if risk > 0 else 0
                
                # Verificar ratio mínimo de riesgo/recompensa
                if risk_reward >= self.min_risk_reward:
                    # Calcular tamaño de posición basado en riesgo
                    risk_amount = self.capital * self.risk_per_trade
                    position_size = risk_amount / risk if risk > 0 else 0
                    
                    # Abrir posición
                    self._open_position(
                        index=index,
                        signal=signal,
                        price=current_price,
                        size=position_size,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        pattern=pattern_name,
                        confidence=confidence,
                        risk_reward=risk_reward
                    )
    
    def _check_exits(self, index: int):
        """
        Verifica si se deben cerrar posiciones existentes.
        
        Args:
            index: Índice actual en los datos
        """
        if self.position == 0:
            return
        
        current_bar = self.data.iloc[index]
        
        # Verificar stop loss y take profit
        if self.position == 1:  # Long
            # Stop loss
            if current_bar['Low'] <= self.stop_loss:
                self._close_position(index, "Stop Loss", self.stop_loss)
            
            # Take profit
            elif current_bar['High'] >= self.take_profit:
                self._close_position(index, "Take Profit", self.take_profit)
        
        elif self.position == -1:  # Short
            # Stop loss
            if current_bar['High'] >= self.stop_loss:
                self._close_position(index, "Stop Loss", self.stop_loss)
            
            # Take profit
            elif current_bar['Low'] <= self.take_profit:
                self._close_position(index, "Take Profit", self.take_profit)
    
    def _open_position(self, index, signal, price, size, stop_loss, take_profit, 
                     pattern, confidence, risk_reward):
        """
        Abre una posición nueva.
        
        Args:
            index: Índice actual en los datos
            signal: Dirección de la señal (-1 = short, 1 = long)
            price: Precio de entrada
            size: Tamaño de la posición
            stop_loss: Nivel de stop loss
            take_profit: Nivel de take profit
            pattern: Tipo de patrón
            confidence: Confianza en el patrón (0-1)
            risk_reward: Ratio riesgo/recompensa
        """
        # Guardar información de la posición
        self.position = signal
        self.entry_price = price
        self.position_size = size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.current_pattern = pattern
        
        # Calcular valor de la posición
        position_value = price * size
        
        # Aplicar comisión
        commission_cost = position_value * self.commission
        self.capital -= commission_cost
        
        # Registrar operación
        self.trades.append({
            "entry_index": index,
            "entry_date": self.data.index[index],
            "entry_price": price,
            "position": "Long" if signal == 1 else "Short",
            "size": size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "pattern": pattern,
            "confidence": confidence,
            "risk_reward": risk_reward,
            "commission": commission_cost
        })
        
        logger.info(f"Abierta posición {self.trades[-1]['position']} en {self.data.index[index]} a {price:.2f}, "
                   f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}, Patrón: {pattern}, Confianza: {confidence:.2f}")
    
    def _close_position(self, index, reason, price=None):
        """
        Cierra la posición actual.
        
        Args:
            index: Índice actual en los datos
            reason: Razón del cierre
            price: Precio de cierre (None para usar el precio actual)
        """
        if self.position == 0:
            return
        
        # Usar precio proporcionado o precio actual
        exit_price = price if price is not None else self.data['Close'].iloc[index]
        
        # Calcular P&L
        if self.position == 1:  # Long
            pnl = (exit_price - self.entry_price) * self.position_size
        else:  # Short
            pnl = (self.entry_price - exit_price) * self.position_size
        
        # Calcular valor de la posición
        position_value = exit_price * self.position_size
        
        # Aplicar comisión y slippage
        commission_cost = position_value * self.commission
        slippage_cost = position_value * self.slippage
        net_pnl = pnl - commission_cost - slippage_cost
        
        # Actualizar capital
        self.capital += net_pnl
        
        # Actualizar último trade
        if self.trades:
            last_trade = self.trades[-1]
            last_trade["exit_index"] = index
            last_trade["exit_date"] = self.data.index[index]
            last_trade["exit_price"] = exit_price
            last_trade["exit_reason"] = reason
            last_trade["pnl"] = net_pnl
            last_trade["net_pnl"] = net_pnl
            last_trade["commission_exit"] = commission_cost
            last_trade["slippage"] = slippage_cost
            
            # Calcular duración
            entry_index = last_trade["entry_index"]
            duration = index - entry_index
            last_trade["duration"] = duration
        
        logger.info(f"Cerrada posición en {self.data.index[index]} a {exit_price:.2f}, "
                   f"Razón: {reason}, P&L: {net_pnl:.2f}")
        
        # Resetear variables de posición
        self.position = 0
        self.entry_price = 0
        self.position_size = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.current_pattern = None
    
    def _calculate_equity(self, index):
        """
        Calcula el equity en el punto actual.
        
        Args:
            index: Índice actual en los datos
            
        Returns:
            float: Valor del equity
        """
        # Si no hay posición, devolver capital actual
        if self.position == 0:
            return self.capital
        
        # Obtener precio actual
        current_price = self.data['Close'].iloc[index]
        
        # Calcular P&L no realizado
        if self.position == 1:  # Long
            unrealized_pnl = (current_price - self.entry_price) * self.position_size
        else:  # Short
            unrealized_pnl = (self.entry_price - current_price) * self.position_size
        
        # Actualizar equity
        return self.capital + unrealized_pnl
    
    def _calculate_statistics(self):
        """
        Calcula estadísticas de rendimiento del backtest.
        
        Returns:
            Dict con estadísticas
        """
        if not self.trades:
            return {"error": "No se realizaron operaciones durante el backtest"}
        
        # Extraer resultados de trades
        pnls = [trade.get("pnl", 0) for trade in self.trades]
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl <= 0]
        
        # Calcular equity curve
        equity_array = np.array(self.equity_curve)
        
        # Rendimiento general
        total_return = (self.equity_curve[-1] / self.initial_capital - 1) * 100
        total_pnl = sum(pnls)
        
        # Estadísticas de trades
        num_trades = len(self.trades)
        num_wins = len(wins)
        num_losses = len(losses)
        win_rate = (num_wins / num_trades * 100) if num_trades > 0 else 0
        
        # Estadísticas de P&L
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        # Ratios
        profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf')
        expectancy = ((win_rate / 100) * avg_win - (1 - win_rate / 100) * abs(avg_loss)) if num_trades > 0 else 0
        
        # Estadísticas de drawdown
        max_equity = np.maximum.accumulate(equity_array)
        drawdown = (max_equity - equity_array) / max_equity * 100
        max_drawdown = np.max(drawdown)
        
        # Estadísticas por patrón
        patterns = {}
        for trade in self.trades:
            pattern = trade.get("pattern", "Unknown")
            if pattern not in patterns:
                patterns[pattern] = {
                    "count": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0,
                    "avg_pnl": 0
                }
            
            patterns[pattern]["count"] += 1
            pnl = trade.get("pnl", 0)
            patterns[pattern]["total_pnl"] += pnl
            
            if pnl > 0:
                patterns[pattern]["wins"] += 1
            else:
                patterns[pattern]["losses"] += 1
        
        # Calcular promedio y tasa de victorias por patrón
        for pattern in patterns:
            count = patterns[pattern]["count"]
            wins = patterns[pattern]["wins"]
            patterns[pattern]["win_rate"] = (wins / count * 100) if count > 0 else 0
            patterns[pattern]["avg_pnl"] = patterns[pattern]["total_pnl"] / count if count > 0 else 0
        
        return {
            "initial_capital": self.initial_capital,
            "final_capital": self.equity_curve[-1],
            "total_return_pct": total_return,
            "total_pnl": total_pnl,
            "total_trades": num_trades,
            "winning_trades": num_wins,
            "losing_trades": num_losses,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "max_drawdown_pct": max_drawdown,
            "patterns": patterns
        }
    
    def _save_results(self, results_dir):
        """
        Guarda los resultados del backtest.
        
        Args:
            results_dir: Directorio para guardar resultados
        """
        # Crear directorio si no existe
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convertir trades a DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # Convertir detecciones a DataFrame
        detections_df = pd.DataFrame(self.pattern_detections)
        
        # Convertir equity curve a DataFrame
        equity_df = pd.DataFrame({
            "Date": self.data.index[:len(self.equity_curve)],
            "Equity": self.equity_curve
        })
        
        # Guardar CSVs
        trades_df.to_csv(os.path.join(results_dir, f"trades_{timestamp}.csv"), index=False)
        detections_df.to_csv(os.path.join(results_dir, f"patterns_{timestamp}.csv"), index=False)
        equity_df.to_csv(os.path.join(results_dir, f"equity_{timestamp}.csv"), index=False)
        
        # Guardar estadísticas como JSON
        with open(os.path.join(results_dir, f"statistics_{timestamp}.json"), 'w') as f:
            json.dump(self.statistics, f, indent=4)
        
        logger.info(f"Resultados guardados en {results_dir}")
    
    def _generate_plots(self, save_dir=None):
        """
        Genera gráficos de resultados.
        
        Args:
            save_dir: Directorio para guardar gráficos (None para mostrar)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Gráfico de trades
        if self.trades:
            plot_trades(
                data=self.data,
                trades=self.trades,
                title=f"Backtest Results - {len(self.trades)} Trades",
                save_path=os.path.join(save_dir, f"trades_{timestamp}.png") if save_dir else None
            )
        
        # 2. Gráfico de equity curve
        plt.figure(figsize=(14, 7))
        plt.plot(self.data.index[:len(self.equity_curve)], self.equity_curve, color='#2196F3', linewidth=2)
        plt.title("Equity Curve", fontsize=16)
        plt.xlabel("Time")
        plt.ylabel("Capital ($)")
        plt.grid(True, alpha=0.3)
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"equity_{timestamp}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # 3. Distribución de patrones
        if self.pattern_detections:
            plot_pattern_distribution(
                patterns=self.trades,
                title="Pattern Performance Analysis",
                save_path=os.path.join(save_dir, f"pattern_distribution_{timestamp}.png") if save_dir else None
            )

def main():
    """Función principal para ejecutar backtest desde línea de comandos."""
    parser = argparse.ArgumentParser(description="AITraderPro - Backtest de patrones de reversión")
    
    parser.add_argument("--data", type=str, required=True, help="Ruta al archivo CSV con datos")
    parser.add_argument("--config", type=str, help="Ruta al archivo de configuración")
    parser.add_argument("--capital", type=float, default=10000, help="Capital inicial")
    parser.add_argument("--commission", type=float, default=0.0, help="Comisión por operación (porcentaje)")
    parser.add_argument("--risk", type=float, default=0.02, help="Porcentaje de riesgo por operación")
    parser.add_argument("--confidence", type=float, default=0.7, help="Confianza mínima para señales")
    parser.add_argument("--risk-reward", type=float, default=1.5, help="Ratio mínimo de riesgo/recompensa")
    parser.add_argument("--model", type=str, help="Ruta al modelo entrenado")
    parser.add_argument("--no-plot", action="store_true", help="No generar gráficos")
    parser.add_argument("--output", type=str, default="backtest_results", help="Directorio para resultados")
    
    args = parser.parse_args()
    
    # Cargar datos
    data_processor = DataProcessor(logger=logger)
    try:
        data = data_processor.load_csv(args.data)
    except Exception as e:
        logger.error(f"Error al cargar datos: {e}")
        return
    
    # Inicializar backtester
    backtester = PatternTraderBacktest(
        config_file=args.config,
        initial_capital=args.capital,
        commission=args.commission,
        risk_per_trade=args.risk,
        min_confidence=args.confidence,
        min_risk_reward=args.risk_reward,
        model_path=args.model
    )
    
    # Ejecutar backtest
    results = backtester.run(
        data=data,
        plot=not args.no_plot,
        save_results=True,
        results_dir=args.output
    )
    
    # Mostrar resumen
    print("\n===== Resumen de Backtest =====")
    print(f"Capital inicial: ${results['initial_capital']:.2f}")
    print(f"Capital final: ${results['final_capital']:.2f}")
    print(f"Retorno total: {results['total_return_pct']:.2f}%")
    print(f"Trades totales: {results['total_trades']}")
    print(f"Win rate: {results['win_rate']:.2f}%")
    print(f"Factor de beneficio: {results['profit_factor']:.2f}")
    print(f"Drawdown máximo: {results['max_drawdown_pct']:.2f}%")
    print("=============================\n")
    
    # Mostrar rendimiento por patrón
    print("===== Rendimiento por Patrón =====")
    patterns = results['patterns']
    for pattern, stats in patterns.items():
        print(f"{pattern}: {stats['count']} trades, Win rate: {stats['win_rate']:.2f}%, PnL Promedio: ${stats['avg_pnl']:.2f}")
    print("=================================")

if __name__ == "__main__":
    main()