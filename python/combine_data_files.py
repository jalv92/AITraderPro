#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para combinar múltiples archivos CSV de datos en uno solo.
Mantiene el orden cronológico y elimina duplicados.
"""

import os
import glob
import pandas as pd
from datetime import datetime
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="AITraderPro - Combinador de Archivos de Datos")
    
    parser.add_argument("--input-dir", type=str, default="data/raw", 
                        help="Carpeta donde se encuentran los archivos CSV a combinar")
    parser.add_argument("--output-file", type=str, default="data/combined_data.csv", 
                        help="Archivo de salida con todos los datos combinados")
    parser.add_argument("--pattern", type=str, default="MNQ_*.csv", 
                        help="Patrón para identificar los archivos a combinar")
    parser.add_argument("--timestamp-col", type=str, default="Timestamp", 
                        help="Nombre de la columna de fecha/hora")
    parser.add_argument("--verbose", action="store_true", 
                        help="Mostrar información detallada durante el procesamiento")
    
    return parser.parse_args()

def main():
    # Parsear argumentos de línea de comandos
    args = parse_args()
    
    print("=== AITraderPro - Combinador de Archivos de Datos ===")
    print(f"Buscando archivos en: {args.input_dir}")
    print(f"Patrón de búsqueda: {args.pattern}")
    
    # Encontrar todos los archivos que coincidan con el patrón
    search_pattern = os.path.join(args.input_dir, args.pattern)
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"ERROR: No se encontraron archivos que coincidan con {search_pattern}")
        return
    
    print(f"Encontrados {len(files)} archivos para combinar")
    
    # Lista para almacenar todos los DataFrames
    all_data = []
    total_rows = 0
    
    # Procesar cada archivo
    for i, file in enumerate(files):
        try:
            if args.verbose:
                print(f"Procesando archivo {i+1}/{len(files)}: {os.path.basename(file)}")
            
            # Leer archivo CSV
            df = pd.read_csv(file)
            
            if args.verbose:
                print(f"  - Leídas {len(df)} filas")
            
            # Convertir la columna de timestamp a datetime para ordenar después
            if args.timestamp_col in df.columns:
                df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col])
            
            all_data.append(df)
            total_rows += len(df)
            
        except Exception as e:
            print(f"ERROR al procesar archivo {file}: {str(e)}")
    
    if not all_data:
        print("No se pudo procesar ningún archivo. Abortando.")
        return
    
    print(f"Combinando {len(all_data)} DataFrames con un total de {total_rows} filas...")
    
    # Combinar todos los DataFrames
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Ordenar por timestamp
    if args.timestamp_col in combined_df.columns:
        print("Ordenando por fecha y hora...")
        combined_df.sort_values(by=args.timestamp_col, inplace=True)
    
    # Eliminar duplicados si existen
    initial_rows = len(combined_df)
    combined_df.drop_duplicates(inplace=True)
    final_rows = len(combined_df)
    
    duplicate_rows = initial_rows - final_rows
    if duplicate_rows > 0:
        print(f"Eliminadas {duplicate_rows} filas duplicadas")
    
    # Crear directorio de salida si no existe
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Guardar el DataFrame combinado
    combined_df.to_csv(args.output_file, index=False)
    
    print(f"\n¡Combinación completada!")
    print(f"Total de archivos procesados: {len(files)}")
    print(f"Total de filas en archivo combinado: {final_rows}")
    print(f"Archivo guardado en: {args.output_file}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1) 