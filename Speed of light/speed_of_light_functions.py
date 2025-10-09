"""
Funciones para análisis de la velocidad de la luz
================================================

Este módulo contiene todas las funciones necesarias para procesar y visualizar
los datos de los experimentos de velocidad de la luz usando análisis sinusoidal
avanzado con FFT, filtrado y ajuste de mínimos cuadrados.

Autor: Daniel Montejo con ayuda de Chat
Fecha: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit


def process_sinusoidal_signals(t, y1, y2):
    """
    Procesa señales sinusoidales usando FFT, filtrado y ajuste sinusoidal.
    
    Args:
        t: array de tiempo
        y1: señal de referencia (CHN1)
        y2: señal ruidosa (CHN2)
    
    Returns:
        dict: Resultados del procesamiento incluyendo fases y señales limpias
    """
    # 1) Sampling & reference frequency from CHN1
    dt = np.median(np.diff(t))
    fs = 1/dt
    Y1 = rfft(y1 - y1.mean())
    freqs = rfftfreq(len(y1), d=dt)
    f0 = freqs[np.argmax(np.abs(Y1))]  # Frecuencia dominante
    
    # 2) Zero-phase band-pass (±20% around f0) → isolate the tone
    bw_frac = 0.20
    low = (f0*(1-bw_frac)) / (fs/2)
    high = (f0*(1+bw_frac)) / (fs/2)
    
    # Asegurar que los valores estén en el rango válido [0, 1]
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    b, a = butter(4, [low, high], btype="band")
    
    y1_bp = filtfilt(b, a, y1)
    y2_bp = filtfilt(b, a, y2)
    
    # 4) Global sinusoid fit at f0 → denoised signals
    w = 2*np.pi*f0
    
    # Definir función sinusoidal para curve_fit
    def sinusoidal_func(t, A, phi, C):
        return A * np.sin(w*t + phi) + C
    
    # Estimaciones iniciales basadas en los datos
    A1_guess = np.std(y1) * np.sqrt(2)
    A2_guess = np.std(y2) * np.sqrt(2)
    C1_guess = np.mean(y1)
    C2_guess = np.mean(y2)
    phi1_guess = 0.0
    phi2_guess = np.pi/2
    
    # Fit CHN1
    popt1, pcov1 = curve_fit(sinusoidal_func, t, y1, 
                            p0=[A1_guess, phi1_guess, C1_guess],
                            maxfev=5000)
    A1, phi1, C1 = popt1
    
    # Normalizar CHN1: asegurar amplitud positiva
    if A1 < 0:
        A1 = -A1
        phi1 = phi1 + np.pi
    
    # Normalizar fase CHN1 a rango [0, 2π)
    # phi1 = phi1 % (2 * np.pi)
    
    y1_clean = sinusoidal_func(t, A1, phi1, C1)
    
    # Calcular errores de CHN1
    sigma1 = np.sqrt(np.diag(pcov1))
    A1_err, phi1_err, C1_err = sigma1
    
    # Fit CHN2
    popt2, pcov2 = curve_fit(sinusoidal_func, t, y2,
                            p0=[A2_guess, phi2_guess, C2_guess],
                            maxfev=5000)
    A2, phi2, C2 = popt2
    
    # Normalizar CHN2: asegurar amplitud positiva
    if A2 < 0:
        A2 = -A2
        phi2 = phi2 + np.pi
    
    # Normalizar fase CHN2 a rango [0, 2π)
    # phi2 = phi2 % (2 * np.pi)
    
    y2_clean = sinusoidal_func(t, A2, phi2, C2)
    
    # Calcular errores de CHN2
    sigma2 = np.sqrt(np.diag(pcov2))
    A2_err, phi2_err, C2_err = sigma2
    
    # Error en la diferencia de fase
    phase_error_rad = np.sqrt(phi1_err**2 + phi2_err**2)       
    
    # Calcular diferencia de fase como CHN1 - CHN2 (CHN2 leading cuando es positivo)
    phase_diff_fit_rad = abs(phi1 - phi2)
    # Para el ratio, usar valor absoluto como antes
    phase_ratio = (phase_diff_fit_rad/np.pi)*100
    
    # Convertir diferencias negativas a positivas (agregar 2π si es negativo)
    # Esto convierte "CHN2 lagging" a "CHN2 leading" equivalente
    # if phase_diff_fit_rad < 0:
    #     phase_diff_fit_rad += 2 * np.pi
    
    
    
    return {
        'f0': f0,
        'phase_fit_rad': phase_diff_fit_rad,
        'phase_error_rad': phase_error_rad,
        'phase_ratio': phase_ratio,
        'y1_clean': y1_clean,
        'y2_clean': y2_clean,
        'y1_filtered': y1_bp,
        'y2_filtered': y2_bp,
        'A1': A1,
        'A2': A2,
        'phi1': phi1,
        'phi2': phi2,
        'C1': C1,
        'C2': C2,
    }


def plot_all_ldn_files():
    """
    Visualiza todos los archivos L_Dn en subplots organizados.
    """
    # Carpeta de datos
    data_folder = "data"
    
    # Buscar todos los archivos L_Dn
    files = [f for f in os.listdir(data_folder) if f.startswith('L_D') and f.endswith('.csv')]
    files.sort()
    
    # Crear figura con subplots
    n_files = len(files)
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    fig.suptitle('Ondas de los archivos L_Dn - Velocidad de la Luz', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()  # Convertir a array 1D para fácil indexación
    
    for i, filename in enumerate(files):
        # Leer datos
        filepath = os.path.join(data_folder, filename)
        data = pd.read_csv(filepath)
        
        # Verificar columnas y usar las apropiadas
        if 'CHN1' in data.columns and 'CHN2' in data.columns:
            x_data, y_data = data['CHN1'], data['CHN2']
            x_label, y_label = 'CHN1', 'CHN2'
            if 'time' in data.columns:
                time_data = data['time'] * 1e6  # Convertir a microsegundos
                has_time = True
            else:
                has_time = False
        elif 'X' in data.columns and 'Y' in data.columns:
            x_data, y_data = data['X'], data['Y']
            x_label, y_label = 'Canal 1', 'Canal 2'
            has_time = False
        else:
            print(f"Warning: Columnas no reconocidas en {filename}")
            print(f"Columnas disponibles: {list(data.columns)}")
            continue
        
        # Obtener el subplot actual
        ax = axes[i]
        
        if has_time:
            # Graficar vs tiempo si está disponible
            ax.plot(time_data, x_data, 'b-', linewidth=1.2, label=x_label)
            ax.plot(time_data, y_data, 'r-', linewidth=1.2, label=y_label)
            ax.set_xlabel('Tiempo (μs)')
            ax.set_ylabel('Amplitud (V)')
        else:
            # Graficar como Lissajous si no hay tiempo
            ax.plot(x_data, y_data, 'g-', linewidth=1.2, alpha=0.7)
            ax.set_xlabel(f'{x_label} (V)')
            ax.set_ylabel(f'{y_label} (V)')
            ax.set_aspect('equal', adjustable='box')
        
        # Configurar ejes
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.7)  # Línea horizontal en y=0
        if has_time:
            ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.7)  # Línea vertical en x=0
        ax.spines['top'].set_visible(False)       # Ocultar eje superior
        ax.spines['right'].set_visible(False)     # Ocultar eje derecho
        
        # Grid y formato
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(f'{filename}', fontweight='bold')
        if has_time:
            ax.legend(loc='upper right', fontsize=9)
        
        # Ajustar márgenes verticales para mejor visualización
        y_min, y_max = ax.get_ylim()
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


def plot_all_sdn_files():
    """
    Visualiza todos los archivos S_Dn en subplots organizados.
    """
    # Carpeta de datos
    data_folder = "data"
    
    # Buscar todos los archivos S_Dn
    files = [f for f in os.listdir(data_folder) if f.startswith('S_D') and f.endswith('.csv')]
    files.sort()
    
    # Crear figura con subplots
    n_files = len(files)
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    fig.suptitle('Ondas de los archivos S_Dn - Velocidad de la Luz', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()  # Convertir a array 1D para fácil indexación
    
    for i, filename in enumerate(files):
        # Leer datos
        filepath = os.path.join(data_folder, filename)
        data = pd.read_csv(filepath)
        
        # Convertir tiempo a microsegundos para mejor lectura
        time_microseconds = data['time'] * 1e6
        
        # Obtener el subplot actual
        ax = axes[i]
        
        # Graficar las dos ondas
        ax.plot(time_microseconds, data['CHN1'], 'b-', linewidth=1.2, label='CHN1')
        ax.plot(time_microseconds, data['CHN2'], 'r-', linewidth=1.2, label='CHN2')
        
        # Configurar ejes para centrar el eje x
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.7)  # Línea horizontal en y=0
        # ax.spines['bottom'].set_position('zero')  # Mover eje x al centro
        ax.spines['top'].set_visible(False)       # Ocultar eje superior
        ax.spines['right'].set_visible(False)     # Ocultar eje derecho
        
        # Grid y formato
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('Tiempo (μs)')
        ax.set_ylabel('Amplitud (V)')
        ax.set_title(f'{filename}', fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        
        # Ajustar márgenes verticales para mejor visualización
        y_min, y_max = ax.get_ylim()
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


def plot_all_sdn_files_with_sinusoidal():
    """
    Visualiza todos los archivos S_Dn con aproximaciones sinusoidales superpuestas.
    Muestra señales ruidosas originales y sus ajustes sinusoidales limpios.
    """
    # Carpeta de datos
    data_folder = "data"
    
    # Buscar todos los archivos S_Dn
    files = [f for f in os.listdir(data_folder) if f.startswith('S_D') and f.endswith('.csv')]
    files.sort()
    
    # Crear figura con subplots
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    fig.suptitle('Ondas S_Dn: Señales Originales vs Ajustes Sinusoidales', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()  # Convertir a array 1D para fácil indexación

    f0 = 0
    
    for i, filename in enumerate(files):
        # Leer datos
        filepath = os.path.join(data_folder, filename)
        data = pd.read_csv(filepath)
        
        # Convertir tiempo a microsegundos para visualización
        time_seconds = data['time'].values.astype(float)
        time_microseconds = time_seconds * 1e6
        y1 = data['CHN1'].values.astype(float)
        y2 = data['CHN2'].values.astype(float)
        
        # Aplicar análisis sinusoidal
        try:
            results = process_sinusoidal_signals(time_seconds, y1, y2)
            
            # Obtener el subplot actual
            ax = axes[i]
            
            # Graficar señales originales (ruidosas) con transparencia
            ax.plot(time_microseconds, y1, 'b-', linewidth=1, alpha=0.6, label='CHN1 Original')
            ax.plot(time_microseconds, y2, 'r-', linewidth=1, alpha=0.6, label='CHN2 Original')
            
            # Superponer ajustes sinusoidales limpios
            ax.plot(time_microseconds, results['y1_clean'], 'navy', linewidth=2.5, label='CHN1 Sinusoidal')
            ax.plot(time_microseconds, results['y2_clean'], 'darkred', linewidth=2.5, label='CHN2 Sinusoidal')
            
            # Configurar ejes para centrar el eje x
            ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.7)  # Línea horizontal en y=0
            # ax.spines['bottom'].set_position('zero')  # Mover eje x al centro
            ax.spines['top'].set_visible(False)       # Ocultar eje superior
            ax.spines['right'].set_visible(False)     # Ocultar eje derecho
            
            # Grid y formato
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlabel('Tiempo (μs)')
            ax.set_ylabel('Amplitud (V)')
            
            # Título con información de fase
            phase_info = f'Fase: {results["phase_fit_rad"]:.2f} ± {results["phase_error_rad"]:.2f} rad'
            ax.set_title(f'{filename} - {phase_info}', fontweight='bold', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            
            # Ajustar márgenes verticales para mejor visualización
            y_min, y_max = ax.get_ylim()
            margin = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - margin, y_max + margin)

            f0 = results['f0'] if results['f0'] != 0 else f0
            
        except Exception as e:
            # En caso de error, mostrar solo señales originales
            ax = axes[i]
            ax.plot(time_microseconds, y1, 'b-', linewidth=1.2, label='CHN1')
            ax.plot(time_microseconds, y2, 'r-', linewidth=1.2, label='CHN2')
            ax.set_title(f'{filename} - Error en análisis', fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            print(f"Error procesando {filename}: {e}")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
    print("\nLeyenda:")
    print("• Líneas claras (transparentes): Señales originales ruidosas")
    print("• Líneas gruesas oscuras: Ajustes sinusoidales limpios")
    print("• Fase: Diferencia de fase entre CHN1 y CHN2 con incertidumbre")
    print(f"• Frecuencia dominante (f0): {f0} kHz")


def plot_sdn_with_sinusoidal_fit(file_number):
    """
    Visualiza un archivo S_Dn con ajuste sinusoidal y análisis de fase.
    
    Args:
        file_number (int): Número del archivo (0-9)
    """
    filename = f"S_D{file_number}.csv"
    filepath = os.path.join("data", filename)
    
    if not os.path.exists(filepath):
        print(f"Error: No se encontró el archivo {filepath}")
        return
    
    # Leer datos
    data = pd.read_csv(filepath)
    time_seconds = data['time'].values.astype(float)
    time_microseconds = time_seconds * 1e6  # Para visualización
    y1 = data['CHN1'].values.astype(float)
    y2 = data['CHN2'].values.astype(float)
    
    # Procesar señales
    results = process_sinusoidal_signals(time_seconds, y1, y2)
    
    # Crear figura con subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Señales temporales
    ax1.plot(time_microseconds, y1, 'b-', linewidth=1, alpha=0.7, label='CHN1 Original')
    ax1.plot(time_microseconds, results['y1_clean'], 'r-', linewidth=2, label='CHN1 Ajuste Sinusoidal')
    # ax1.axhline(y=0, color='k', linewidth=0.8, alpha=0.7)
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylabel('Amplitud CHN1 (V)', fontsize=12)
    ax1.set_title(f'CHN1 - {filename}', fontweight='bold')
    ax1.legend(fontsize=10)
    
    # Plot 2: CHN2
    ax2.plot(time_microseconds, y2, 'g-', linewidth=1, alpha=0.7, label='CHN2 Original')
    ax2.plot(time_microseconds, results['y2_clean'], 'm-', linewidth=2, label='CHN2 Ajuste Sinusoidal')
    # ax2.axhline(y=0, color='k', linewidth=0.8, alpha=0.7)
    ax2.spines['bottom'].set_position('zero')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylabel('Amplitud CHN2 (V)', fontsize=12)
    ax2.set_title(f'CHN2 - {filename}', fontweight='bold')
    ax2.legend(fontsize=10)
    
    # Plot 3: Lissajous original
    ax3.plot(y1, y2, 'b.', markersize=2, alpha=0.6, label='Original')
    ax3.set_xlabel('CHN1 (V)')
    ax3.set_ylabel('CHN2 (V)')
    ax3.set_title('Lissajous Original')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    # Plot 4: Lissajous limpia
    ax4.plot(results['y1_clean'], results['y2_clean'], 'r.', markersize=2, alpha=0.8, label='Ajuste Sinusoidal')
    ax4.set_xlabel('CHN1 Limpia (V)')
    ax4.set_ylabel('CHN2 Limpia (V)')
    ax4.set_title('Lissajous Ajuste Sinusoidal')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar información del análisis
    print(f"\n--- Análisis de {filename} ---")
    print(f"Frecuencia de muestreo: {results['fs']/1e6:.2f} MHz")
    print(f"Frecuencia dominante: {results['f0']/1e3:.2f} kHz")
    print(f"Desfase en radianes: {results['phase_fit_rad']:.4f} ± {results['phase_error_rad']:.4f} rad")
    print(f"Desfase en grados: {results['phase_fit_rad']*180/np.pi:.2f}° ± {results['phase_error_rad']*180/np.pi:.2f}°")
    print(f"Amplitud CHN1 (normalizada): {results['A1']:.6f} V")
    print(f"Amplitud CHN2 (normalizada): {results['A2']:.6f} V")
    print(f"Fase CHN1 (normalizada): {results['phi1']:.4f} rad ({results['phi1']*180/np.pi:.2f}°)")
    print(f"Fase CHN2 (normalizada): {results['phi2']:.4f} rad ({results['phi2']*180/np.pi:.2f}°)")
    
    return {
        'data': data,
        'time_us': time_microseconds,
        'results': results
    }


def process_all_sdn_files_sinusoidal():
    """
    Procesa todos los archivos S_DN con análisis sinusoidal avanzado.
    
    Returns:
        dict: Diccionario con resultados procesados para cada archivo
    """
    data_folder = "data"
    s_files = [f for f in os.listdir(data_folder) if f.startswith('S_D') and f.endswith('.csv')]
    s_files.sort()
    
    results = {}
    
    print("Procesando archivos S_DN con análisis sinusoidal...")
    
    for filename in s_files:
        file_number = int(filename.split('_D')[1].split('.')[0])
        print(f"Procesando {filename}...")
        
        # Leer datos
        filepath = os.path.join(data_folder, filename)
        data = pd.read_csv(filepath)
        time_seconds = data['time']
        
        # Aplicar análisis sinusoidal
        sinusoidal_results = process_sinusoidal_signals(
            time_seconds, data['CHN1'], data['CHN2']
        )
        
        results[file_number] = {
            'filename': filename,
            'sinusoidal_results': sinusoidal_results
        }
    
    print(f"Completado procesamiento sinusoidal de {len(results)} archivos.")
    return results


def plot_lissajous_comparison_sinusoidal(s_results, file_number):
    """
    Crea figuras de Lissajous comparando datos sinusoidales S_DN con datos ruidosos L_DN.
    
    Args:
        s_results: resultados procesados de archivos S_DN con análisis sinusoidal
        file_number: número del archivo a visualizar
    """
    if file_number not in s_results:
        print(f"Error: No se encontraron datos procesados para S_D{file_number}")
        return
    
    # Datos sinusoidales de S_DN
    s_data = s_results[file_number]
    results = s_data['sinusoidal_results']
    
    # Cargar datos ruidosos de L_DN
    l_filename = f"L_D{file_number}.csv"
    l_filepath = os.path.join("data", l_filename)
    
    if not os.path.exists(l_filepath):
        print(f"Error: No se encontró el archivo {l_filepath}")
        return
    
    l_data = pd.read_csv(l_filepath)
    
    # Verificar columnas y usar las apropiadas
    if 'CHN1' in l_data.columns and 'CHN2' in l_data.columns:
        l_chn1, l_chn2 = l_data['CHN1'], l_data['CHN2']
    elif 'X' in l_data.columns and 'Y' in l_data.columns:
        l_chn1, l_chn2 = l_data['X'], l_data['Y']
    else:
        print(f"Error: Columnas no reconocidas en {l_filename}")
        print(f"Columnas disponibles: {list(l_data.columns)}")
        return
    
    # Crear figura con subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Subplot 1: Lissajous ruidosa (L_DN)
    ax1.plot(l_chn1, l_chn2, 'b-', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Canal 1 (V)')
    ax1.set_ylabel('Canal 2 (V)')
    ax1.set_title(f'Lissajous Ruidosa - {l_filename}')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.008, 0.008)
    ax1.set_ylim(-0.008, 0.008)
    ax1.set_aspect('equal', adjustable='box')
    
    # Subplot 2: Lissajous filtrada (S_DN)
    ax2.plot(results['y1_filtered'], results['y2_filtered'], 'g-', alpha=0.8, linewidth=2)
    ax2.set_xlabel('CHN1 Filtrada (V)')
    ax2.set_ylabel('CHN2 Filtrada (V)')
    ax2.set_title(f'Lissajous Filtrada - S_D{file_number}')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.008, 0.008)
    ax2.set_ylim(-0.008, 0.008)
    ax2.set_aspect('equal', adjustable='box')
    
    # Subplot 3: Lissajous ajuste sinusoidal (S_DN)
    ax3.plot(results['y1_clean'], results['y2_clean'], 'r-', alpha=0.8, linewidth=2)
    ax3.set_xlabel('CHN1 Ajuste (V)')
    ax3.set_ylabel('CHN2 Ajuste (V)')
    ax3.set_title(f'Lissajous Ajuste Sinusoidal - S_D{file_number}')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.008, 0.008)
    ax3.set_ylim(-0.008, 0.008)
    ax3.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar información del análisis
    print(f"\n--- Comparación S_D{file_number} vs L_D{file_number} ---")
    print(f"Frecuencia dominante: {results['f0']/1e3:.2f} kHz")
    print(f"Desfase (ajuste sinusoidal): {results['phase_fit_rad']:.2f} ± {results['phase_error_rad']:.2f} rad")
    print(f"Relación de fases: {results['phase_ratio']:.2f}%")
    return results


def plot_all_lissajous_comparison_sinusoidal(s_results):
    """
    Crea una figura con todas las comparaciones de Lissajous usando análisis sinusoidal.
    """
    n_files = len(s_results)
    cols = 2
    rows = (n_files + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Comparación de Figuras Lissajous - Análisis Sinusoidal', fontsize=16, fontweight='bold')
    
    phase_differences_fit = []
    phase_errors = []
    
    for i, file_number in enumerate(sorted(s_results.keys())):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # Datos sinusoidales
        s_data = s_results[file_number]
        results = s_data['sinusoidal_results']
        
        # Cargar datos ruidosos
        l_filename = f"L_D{file_number}.csv"
        l_filepath = os.path.join("data", l_filename)
        
        if os.path.exists(l_filepath):
            l_data = pd.read_csv(l_filepath)
            
            # Verificar columnas y usar las apropiadas
            if 'CHN1' in l_data.columns and 'CHN2' in l_data.columns:
                l_chn1, l_chn2 = l_data['CHN1'], l_data['CHN2']
            elif 'X' in l_data.columns and 'Y' in l_data.columns:
                l_chn1, l_chn2 = l_data['X'], l_data['Y']
            else:
                print(f"Warning: Columnas no reconocidas en {l_filename}")
                continue
            
            # Graficar ambas curvas
            ax.plot(l_chn1, l_chn2, 'b-', alpha=0.4, linewidth=1, label='Ruidosa')
            ax.plot(results['y1_clean'], results['y2_clean'], 'r-', alpha=0.8, linewidth=1.5, label='Sinusoidal')
            
            phase_differences_fit.append(results['phase_fit_rad'])
            phase_errors.append(results['phase_error_rad'])
        else:
            ax.plot(results['y1_clean'], results['y2_clean'], 'r-', alpha=0.8, linewidth=1.5, label='Sinusoidal')
        
        ax.set_xlabel('Canal 1 (V)')
        ax.set_ylabel('Canal 2 (V)')
        ax.set_title(f'D{file_number} - Porcentaje de similitud: {results['phase_ratio']:.2f}%')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_aspect('equal', adjustable='box')
    
    # Ocultar subplots vacíos
    for i in range(n_files, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()
    
    # Resumen de desfases y estadísticas
    if phase_differences_fit:
        print(f"\n--- Resumen de Análisis Sinusoidal ---")
        print(f"Desfase promedio: {np.mean(phase_differences_fit):.2f}° ± {np.std(phase_differences_fit):.2f}°")
        print(f"Error promedio de medición: {np.mean(phase_errors):.2f}° ± {np.std(phase_errors):.2f}°")
        print(f"Rango de fases: [{np.min(phase_differences_fit):.1f}°, {np.max(phase_differences_fit):.1f}°]")
    
    return {
        'phase_fit': phase_differences_fit,
        'phase_errors': phase_errors,
    }