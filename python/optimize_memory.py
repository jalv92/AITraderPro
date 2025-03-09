#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import psutil
import argparse
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import torch


def print_memory_usage(message: str) -> None:
    """Imprime información sobre el uso actual de memoria."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    print(f"{message}: {memory_mb:.2f} MB")


def optimize_dataframe(df: pd.DataFrame, categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Optimiza el uso de memoria de un DataFrame reduciendo los tipos de datos."""
    # Copia para evitar modificar el original
    result = df.copy()
    
    # Columnas para ignorar (ej: fechas ya optimizadas)
    ignore_columns = set()
    
    # Convertir fechas a datetime optimizado si existen
    datetime_columns = [col for col in result.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in datetime_columns:
        if pd.api.types.is_object_dtype(result[col]):
            try:
                result[col] = pd.to_datetime(result[col])
                ignore_columns.add(col)
            except:
                pass
    
    # Optimizar columnas categóricas
    if categorical_columns:
        for col in categorical_columns:
            if col in result.columns and col not in ignore_columns:
                result[col] = result[col].astype('category')
                ignore_columns.add(col)
    
    # Optimizar columnas numéricas
    for col in result.columns:
        if col in ignore_columns:
            continue
            
        col_type = result[col].dtype
        
        # Enteros
        if pd.api.types.is_integer_dtype(col_type):
            c_min = result[col].min()
            c_max = result[col].max()
            
            if c_min >= 0:
                if c_max < 255:
                    result[col] = result[col].astype(np.uint8)
                elif c_max < 65535:
                    result[col] = result[col].astype(np.uint16)
                elif c_max < 4294967295:
                    result[col] = result[col].astype(np.uint32)
                else:
                    result[col] = result[col].astype(np.uint64)
            else:
                if c_min > -128 and c_max < 127:
                    result[col] = result[col].astype(np.int8)
                elif c_min > -32768 and c_max < 32767:
                    result[col] = result[col].astype(np.int16)
                elif c_min > -2147483648 and c_max < 2147483647:
                    result[col] = result[col].astype(np.int32)
                else:
                    result[col] = result[col].astype(np.int64)
        
        # Flotantes
        elif pd.api.types.is_float_dtype(col_type):
            c_min = result[col].min()
            c_max = result[col].max()
            
            # Verificar si podemos convertir a float de menor precisión
            if np.isfinite(c_min) and np.isfinite(c_max):
                if np.abs(c_min) < 1e10 and np.abs(c_max) < 1e10:
                    result[col] = result[col].astype(np.float32)
    
    return result


def clean_gpu_memory() -> None:
    """Limpia la memoria de la GPU si está disponible."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Memoria de GPU liberada")


def optimize_training_data(
    data_file: str, 
    output_file: Optional[str] = None,
    max_rows: Optional[int] = None
) -> str:
    """Optimiza un archivo de datos para entrenar modelos, reduciendo su huella de memoria."""
    print(f"Optimizando archivo: {data_file}")
    print_memory_usage("Memoria inicial")
    
    # Determinar el archivo de salida
    if output_file is None:
        base, ext = os.path.splitext(data_file)
        output_file = f"{base}_optimized{ext}"
    
    # Cargar datos (con límite de filas opcional)
    print(f"Cargando datos...")
    if max_rows:
        df = pd.read_csv(data_file, nrows=max_rows)
        print(f"Cargadas {max_rows} filas de datos")
    else:
        df = pd.read_csv(data_file)
        print(f"Cargadas {len(df)} filas de datos")
    
    print_memory_usage("Memoria después de cargar datos")
    
    # Optimizar el DataFrame
    print("Optimizando tipos de datos...")
    df_optimized = optimize_dataframe(df)
    
    # Liberar memoria del DataFrame original
    del df
    gc.collect()
    print_memory_usage("Memoria después de optimizar")
    
    # Guardar resultado optimizado
    print(f"Guardando datos optimizados en: {output_file}")
    df_optimized.to_csv(output_file, index=False)
    print(f"Archivo optimizado guardado con éxito")
    
    # Liberar toda la memoria
    del df_optimized
    gc.collect()
    clean_gpu_memory()
    print_memory_usage("Memoria final")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Optimiza archivos de datos para uso eficiente de memoria")
    parser.add_argument("--input", "-i", required=True, help="Archivo CSV de entrada a optimizar")
    parser.add_argument("--output", "-o", help="Archivo CSV de salida (opcional)")
    parser.add_argument("--max-rows", "-m", type=int, help="Número máximo de filas a procesar (opcional)")
    
    args = parser.parse_args()
    
    optimize_training_data(
        data_file=args.input,
        output_file=args.output,
        max_rows=args.max_rows
    )


if __name__ == "__main__":
    main() 