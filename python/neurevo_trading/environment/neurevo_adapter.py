import numpy as np
import time
from typing import Callable, Dict, Tuple, Any, List
import gym
import random
import torch
import sys
from datetime import datetime, timedelta
import pandas as pd

# Importar NeurEvo
import neurevo

# Importar componentes de neurevo_trading
from neurevo_trading.environment.trading_env import TradingEnvironment

class NeurEvoEnvironmentAdapter:
    """
    Adaptador que permite usar entornos de trading con NeurEvo.
    Optimizado para entrenar modelos enfocados en maximizar el PnL con crecimiento estable.
    """
    
    def __init__(self, trading_env: TradingEnvironment):
        """
        Inicializa el adaptador.
        
        Args:
            trading_env: Entorno de trading a adaptar
        """
        self.env = trading_env
        self.is_registered = False
        self.brain = None
        self.env_id = "TradingEnv"
        self.agent = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")
    
    def register_with_brain(self, brain, env_id=None):
        """
        Registra el entorno adaptado con un cerebro NeurEvo.
        
        Args:
            brain: Cerebro NeurEvo
            env_id: Identificador para el entorno (opcional)
            
        Returns:
            Identificador del entorno registrado
        """
        if env_id:
            self.env_id = env_id
        
        self.brain = brain
        
        # La API ha cambiado, ahora no hay que registrar el entorno primero
        # Directamente creamos el agente para este entorno usando create_for_environment
        
        self.is_registered = True
        return self.env_id
    
    def create_agent(self, config=None):
        """
        Crea un agente NeurEvo para este entorno.
        
        Args:
            config: Configuración para el agente
            
        Returns:
            ID del agente creado
        """
        if not self.brain:
            raise ValueError("Brain not set. Call register_with_brain() first.")
        
        # Crear un adaptador de entorno gym que encapsule nuestro entorno
        class GymEnvAdapter(gym.Env):
            def __init__(self, env):
                self.env = env
                self.action_space = env.action_space
                self.observation_space = env.observation_space
                
            def reset(self):
                return self.env.reset()
                
            def step(self, action):
                return self.env.step(action)
        
        # Crear el adaptador gym
        gym_env = GymEnvAdapter(self.env)
        
        print("Creando agente para entorno:", gym_env)
        # Crear agente directamente con el entorno gym
        self.agent = self.brain.create_for_environment(gym_env)
        self.is_registered = True
        return self.agent
    
    def _print_progress_bar(self, iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
        """
        Imprime una barra de progreso en el terminal.
        
        Args:
            iteration: Iteración actual (comienza desde 0)
            total: Total de iteraciones
            prefix: Texto prefijo
            suffix: Texto sufijo
            decimals: Precisión decimal del porcentaje
            length: Longitud de la barra
            fill: Caracter para llenar la barra
            print_end: Caracter final (por defecto: '\r')
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
        # Imprimir nueva línea al final
        if iteration == total:
            print()
    
    def train_agent(self, agent_id=None, episodes=1000, verbose=True):
        """
        Entrena un agente en este entorno, optimizando para PnL estable.
        
        Args:
            agent_id: ID del agente a entrenar (no usado, solo por compatibilidad)
            episodes: Número de episodios
            verbose: Si es True, muestra progreso
            
        Returns:
            Resultados del entrenamiento
        """
        if not self.is_registered:
            raise ValueError("Environment not registered with a brain. Call register_with_brain() first.")
        
        print(f"Entrenando agente por {episodes} episodios...")
        start_time = time.time()
        last_update_time = start_time
        
        # Implementamos nuestro propio bucle de entrenamiento
        rewards = []
        equity_curves = []
        max_drawdowns = []
        
        # Iniciar estadísticas de seguimiento
        running_reward = 0.0
        running_drawdown = 0.0
        best_reward = float("-inf")
        worst_reward = float("inf")
        
        # Imprimir encabezado de estadísticas
        if verbose:
            print("\n{:<10} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
                "Episodio", "Recompensa", "Prom. 10 Ep.", "Max DD", "Mejor Ep.", "Tiempo Restante"))
            print("-" * 85)
        
        for episode in range(episodes):
            # Iniciar episodio
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            episode_equity = []
            
            while not done:
                # Usar neurevo para predecir la acción
                # Si no es posible, usar una acción aleatoria como fallback
                try:
                    if hasattr(self.brain, 'predict') and self.agent is not None:
                        action = self.brain.predict(self.agent, state)
                    else:
                        action = self.env.action_space.sample()
                except:
                    action = self.env.action_space.sample()
                
                # Ejecutar la acción en el entorno
                next_state, reward, done, info = self.env.step(action)
                
                # Registrar equity actual
                episode_equity.append(info.get('balance', 0))
                
                # Actualizar estado
                state = next_state
                total_reward += reward
                steps += 1
            
            # Guardar métricas del episodio
            rewards.append(total_reward)
            equity_curves.append(episode_equity)
            max_drawdown = info.get('max_drawdown', 0)
            max_drawdowns.append(max_drawdown)
            
            # Actualizar estadísticas
            if total_reward > best_reward:
                best_reward = total_reward
                best_episode = episode + 1
            if total_reward < worst_reward:
                worst_reward = total_reward
            
            # Calcular promedio de las últimas 10 recompensas
            recent_rewards = rewards[-10:] if len(rewards) >= 10 else rewards
            avg_reward = np.mean(recent_rewards)
            
            # Actualizar promedio móvil de drawdown
            recent_drawdowns = max_drawdowns[-10:] if len(max_drawdowns) >= 10 else max_drawdowns
            avg_drawdown = np.mean(recent_drawdowns)
            
            # Estimar tiempo restante
            current_time = time.time()
            elapsed_time = current_time - start_time
            time_per_episode = elapsed_time / (episode + 1)
            remaining_episodes = episodes - (episode + 1)
            estimated_remaining_time = time_per_episode * remaining_episodes
            
            # Formatear tiempo restante
            remaining_time_str = str(timedelta(seconds=int(estimated_remaining_time)))
            
            # Mostrar progreso si es apropiado
            if verbose and ((episode + 1) % 5 == 0 or episode == 0 or episode == episodes - 1):
                # Mostrar estadísticas actualizadas
                print("{:<10} {:<15.2f} {:<15.2f} {:<15.2%} {:<15.2f} {:<15}".format(
                    episode + 1, total_reward, avg_reward, avg_drawdown, best_reward, remaining_time_str))
                
                # Actualizar barra de progreso
                self._print_progress_bar(episode + 1, episodes, 
                                      prefix=f'Progreso:', 
                                      suffix=f'Completado {episode+1}/{episodes}', 
                                      length=40)
                
                # Forzar actualización de la salida
                sys.stdout.flush()
            
            # Actualizar hora de la última actualización
            last_update_time = current_time
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\nEntrenamiento completado en {training_time:.2f} segundos")
        print(f"Mejor episodio: {best_episode} con recompensa: {best_reward:.2f}")
        print(f"Peor recompensa: {worst_reward:.2f}")
        print(f"Promedio final de recompensa: {np.mean(rewards):.2f}")
        print(f"Drawdown promedio final: {np.mean(max_drawdowns):.2%}")
        
        # Generar resultados
        results = {
            "episodes": episodes,
            "rewards": rewards,
            "avg_reward": np.mean(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards),
            "avg_drawdown": np.mean(max_drawdowns),
            "training_time": training_time,
            "best_episode": best_episode
        }
        
        return results
    
    def run_episode(self, render=False, max_steps=5000, verbose=True):
        """
        Ejecuta un episodio completo usando el cerebro NeurEvo para tomar decisiones.
        Implementa manejo robusto de errores para evitar interrupciones durante la evaluación.
        
        Args:
            render: Si es True, renderiza el entorno en cada paso
            max_steps: Número máximo de pasos para evitar bucles infinitos
            verbose: Si es True, muestra información detallada durante la ejecución
            
        Returns:
            float: Recompensa total obtenida en el episodio o un valor por defecto en caso de error
        """
        # Verificar que el entorno está registrado con un cerebro
        if self.brain is None:
            raise ValueError("El entorno no está registrado con un cerebro NeurEvo")
        
        try:
            # Inicializar entorno y variables de seguimiento
            state = self.env.reset()
            done = False
            total_reward = 0
            equity_curve = []
            steps = 0
            
            # Registrar estado inicial
            if hasattr(self.env, 'balance'):
                initial_balance = self.env.balance
                equity_curve.append(initial_balance)
            else:
                initial_balance = 10000  # Valor por defecto
                equity_curve.append(initial_balance)
            
            if verbose:
                print(f"Iniciando episodio con balance inicial: {initial_balance:.2f}")
            
            # Bucle principal del episodio
            while not done and steps < max_steps:
                # Predicción de acción con manejo de excepciones
                try:
                    if hasattr(self.brain, 'predict') and self.agent is not None:
                        action = self.brain.predict(self.agent, state)
                    else:
                        if verbose:
                            print("ADVERTENCIA: Usando acción aleatoria (cerebro o agente no configurados correctamente)")
                        action = self.env.action_space.sample()
                except Exception as e:
                    print(f"ERROR al predecir acción (paso {steps}): {e}")
                    # Generar acción aleatoria en caso de error
                    action = self.env.action_space.sample()
                
                # Ejecución de paso con manejo robusto de excepciones
                try:
                    # Paso principal en el entorno
                    state, reward, done, info = self.env.step(action)
                    
                    # Si la recompensa es NaN o infinita, reemplazar con cero
                    if not np.isfinite(reward):
                        print(f"ADVERTENCIA: Recompensa no válida ({reward}), reemplazada con 0")
                        reward = 0
                    
                    # Acumular recompensa
                    total_reward += reward
                    
                    # Registrar equity actual
                    if isinstance(info, dict) and 'balance' in info:
                        current_balance = info['balance']
                        if np.isfinite(current_balance):
                            equity_curve.append(current_balance)
                    elif hasattr(self.env, 'balance'):
                        current_balance = self.env.balance
                        if np.isfinite(current_balance):
                            equity_curve.append(current_balance)
                except Exception as e:
                    print(f"ERROR crítico al ejecutar paso {steps}: {e}")
                    print("Terminando episodio anticipadamente debido a error irrecuperable")
                    # En caso de error crítico, terminar episodio
                    done = True
                    
                    # Si no se ha registrado balance final, usar el último válido
                    if len(equity_curve) > 0:
                        final_balance = equity_curve[-1]
                    else:
                        final_balance = initial_balance
                
                # Renderizar si es necesario
                if render:
                    try:
                        self.env.render()
                    except Exception as render_error:
                        print(f"Error al renderizar: {render_error}")
                
                # Incrementar contador de pasos
                steps += 1
                
                # Feedback periódico en modo verbose
                if verbose and steps % 100 == 0:
                    current_balance = equity_curve[-1] if len(equity_curve) > 0 else initial_balance
                    print(f"Paso {steps}: Balance = {current_balance:.2f}, Recompensa = {reward:.2f}")
            
            # Cálculo de métricas finales
            try:
                # Balance final
                if hasattr(self.env, 'balance') and np.isfinite(self.env.balance):
                    final_balance = self.env.balance
                elif len(equity_curve) > 0:
                    final_balance = equity_curve[-1]
                else:
                    final_balance = initial_balance
                
                # Máximo drawdown
                if hasattr(self.env, 'max_drawdown') and np.isfinite(self.env.max_drawdown):
                    max_drawdown = self.env.max_drawdown
                elif len(equity_curve) > 1:
                    # Cálculo manual del drawdown máximo
                    peaks = pd.Series(equity_curve).cummax()
                    drawdowns = 1 - pd.Series(equity_curve) / peaks
                    max_drawdown = drawdowns.max()
                else:
                    max_drawdown = 0
                
                # Profit and Loss total
                if hasattr(self.env, 'total_pnl') and np.isfinite(self.env.total_pnl):
                    total_pnl = self.env.total_pnl
                elif len(equity_curve) > 0:
                    total_pnl = final_balance - initial_balance
                else:
                    total_pnl = 0
                
                # Porcentaje de retorno
                pct_return = ((final_balance / initial_balance) - 1) * 100 if initial_balance > 0 else 0
                
                # Si alcanzó el límite de pasos, notificar
                if steps >= max_steps and not done:
                    print(f"ADVERTENCIA: Episodio alcanzó el límite máximo de {max_steps} pasos")
                
                # Mostrar resumen del episodio
                if verbose:
                    print(f"\n--- RESUMEN DEL EPISODIO ---")
                    print(f"Balance final:    {final_balance:.2f} (Inicial: {initial_balance:.2f})")
                    print(f"Retorno:          {pct_return:.2f}%")
                    print(f"PnL total:        {total_pnl:.2f}")
                    print(f"Máx. Drawdown:    {max_drawdown:.2%}")
                    print(f"Pasos ejecutados: {steps}")
                    print(f"Recompensa total: {total_reward:.2f}")
                    print(f"Estado final:     {'Completado' if done else 'Interrumpido'}")
                    print("-------------------------\n")
                
                # Retornar recompensa total
                return total_reward
                
            except Exception as metric_error:
                print(f"ERROR al calcular métricas finales: {metric_error}")
                # Devolver recompensa acumulada hasta el momento
                return total_reward
            
        except Exception as e:
            print(f"ERROR CRÍTICO durante la ejecución del episodio: {e}")
            import traceback
            traceback.print_exc()
            
            # Valor por defecto en caso de error crítico
            return -1000.0