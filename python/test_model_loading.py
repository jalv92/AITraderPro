"""
Script para probar la carga del modelo NeuroEvo y diagnosticar problemas de compatibilidad.
"""

import os
import sys
import pickle
import torch
import traceback
import numpy as np

# Agregar directorio principal al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_with_pickle(model_path):
    """Intenta cargar el modelo directamente con pickle."""
    print(f"Intentando cargar con pickle: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    print("¡Éxito! Modelo cargado con pickle")
    return model_data

def load_with_torch(model_path, weights_only=False):
    """Intenta cargar el modelo con torch.load."""
    print(f"Intentando cargar con torch.load (weights_only={weights_only}): {model_path}")
    model_data = torch.load(model_path, map_location='cpu', weights_only=weights_only)
    print("¡Éxito! Modelo cargado con torch.load")
    return model_data

def examine_model_data(model_data):
    """Examina el contenido del modelo cargado."""
    print("\n=== Contenido del modelo ===")
    
    # Ver las claves principales
    print(f"Claves principales: {list(model_data.keys()) if isinstance(model_data, dict) else 'No es un diccionario'}")
    
    # Si es un diccionario y tiene un adapter, examinar
    if isinstance(model_data, dict) and 'adapter' in model_data:
        adapter = model_data['adapter']
        print(f"\nTipo del adapter: {type(adapter)}")
        
        # Ver métodos disponibles
        methods = [attr for attr in dir(adapter) if callable(getattr(adapter, attr)) and not attr.startswith('_')]
        print(f"Métodos disponibles: {methods}")
        
        # Si tiene configuración, mostrarla
        if 'config' in model_data:
            print(f"\nConfiguración: {model_data['config']}")
    
    print("==========================\n")

def try_load_model(model_path):
    """Intenta cargar el modelo de diversas maneras."""
    methods = [
        lambda: load_with_pickle(model_path),
        lambda: load_with_torch(model_path, weights_only=False),
        lambda: load_with_torch(model_path, weights_only=True)
    ]
    
    for i, method in enumerate(methods):
        print(f"\n[Método {i+1}]")
        try:
            model_data = method()
            examine_model_data(model_data)
            print(f"Método {i+1} exitoso!")
            return model_data
        except Exception as e:
            print(f"Error: {e}")
            print("Traza de error:")
            traceback.print_exc()
            print("\nProbando siguiente método...\n")
    
    print("Todos los métodos de carga fallaron.")
    return None

def simulate_prediction(model_data):
    """Simula una predicción con el modelo cargado."""
    print("\n=== Simulando predicción ===")
    
    # Crear datos de prueba
    features = np.random.rand(5, 10)  # 5 características, 10 puntos de tiempo
    print(f"Datos de prueba shape: {features.shape}")
    
    try:
        if isinstance(model_data, dict) and 'adapter' in model_data:
            adapter = model_data['adapter']
            
            # Intentar diferentes métodos
            prediction_methods = [
                lambda: adapter.predict(features),
                lambda: adapter.run_episode(None),
                lambda: adapter.agent.act(features) if hasattr(adapter, 'agent') else None
            ]
            
            for i, method in enumerate(prediction_methods):
                try:
                    print(f"\nIntentando método de predicción {i+1}...")
                    result = method()
                    print(f"Resultado: {result}")
                    if result is not None:
                        print(f"¡Predicción exitosa con método {i+1}!")
                        return result
                except Exception as e:
                    print(f"Error en método {i+1}: {e}")
            
            print("No se pudo realizar una predicción con ningún método.")
        else:
            print("Formato de modelo no reconocido para predicción.")
    except Exception as e:
        print(f"Error al simular predicción: {e}")
    
    print("===========================\n")
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnóstico de carga de modelos NeuroEvo")
    parser.add_argument("--model", type=str, required=True, help="Ruta al archivo del modelo")
    
    args = parser.parse_args()
    
    print(f"=== Diagnóstico de carga de modelo: {args.model} ===\n")
    
    # Verificar que el archivo existe
    if not os.path.exists(args.model):
        print(f"Error: El archivo {args.model} no existe")
        return
    
    # Intentar cargar el modelo
    model_data = try_load_model(args.model)
    
    if model_data:
        # Simular una predicción
        simulate_prediction(model_data)

if __name__ == "__main__":
    main() 