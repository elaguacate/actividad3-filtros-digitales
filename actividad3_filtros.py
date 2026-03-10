"""Actividad formativa 3: Implementación y evaluación de filtros digitales
Autor: Sergio Ponce
Lenguaje: Python (SciPy, NumPy, Matplotlib)

Objetivo:
- Diseñar filtros FIR e IIR pasa bajos, pasa altos y pasa bandas.
- Analizar su respuesta en frecuencia.
- Aplicarlos sobre una señal con ruido y comparar antes/después.
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# ==============================
# 1. Definición de la señal de entrada
# ==============================

fs = 1000.0  # Frecuencia de muestreo [Hz]
T = 1.0      # Duración de la señal [s]
N = int(T * fs)
t = np.arange(N) / fs

# Ruido blanco gaussiano (para simular ruido en la medición)
np.random.seed(0)  # Para reproducibilidad
noise = np.random.normal(0, 1, N)

# Señal compuesta: 50 Hz (baja), 200 Hz (alta) + ruido
x_low = np.sin(2 * np.pi * 50 * t)          # componente de baja frecuencia
x_high = 0.5 * np.sin(2 * np.pi * 200 * t)  # componente de alta frecuencia
x = x_low + x_high + noise                  # señal total con ruido

# ==============================
# 2. Diseño de filtros FIR (ventana de Hamming)
# ==============================

numtaps = 101  # orden+1 del filtro FIR

# Pasa bajos FIR (corta por encima de 100 Hz)
cut_lp = 100 / (fs / 2)  # frecuencia normalizada (0–1)
fir_lp = signal.firwin(numtaps, cut_lp, window='hamming')

# Pasa altos FIR (deja frecuencias por encima de 150 Hz)
cut_hp = 150 / (fs / 2)
fir_hp = signal.firwin(numtaps, cut_hp, pass_zero=False, window='hamming')

# Pasa bandas FIR (80–120 Hz, alrededor de 100 Hz)
cut_bp = [80 / (fs / 2), 120 / (fs / 2)]
fir_bp = signal.firwin(numtaps, cut_bp, pass_zero=False, window='hamming')

# ==============================
# 3. Diseño de filtros IIR Butterworth (orden 4)
# ==============================

# Pasa bajos IIR
b_lp, a_lp = signal.butter(4, 100 / (fs / 2), btype='low')

# Pasa altos IIR
b_hp, a_hp = signal.butter(4, 150 / (fs / 2), btype='high')

# Pasa bandas IIR
b_bp, a_bp = signal.butter(4, [80 / (fs / 2), 120 / (fs / 2)], btype='band')

# ==============================
# 4. Aplicación de filtros a la señal
# ==============================

# FIR (convolución causal)
y_fir_lp = signal.lfilter(fir_lp, 1.0, x)
y_fir_hp = signal.lfilter(fir_hp, 1.0, x)
y_fir_bp = signal.lfilter(fir_bp, 1.0, x)

# IIR (filtros de fase cero para evitar desfase adicional)
y_iir_lp = signal.filtfilt(b_lp, a_lp, x)
y_iir_hp = signal.filtfilt(b_hp, a_hp, x)
y_iir_bp = signal.filtfilt(b_bp, a_bp, x)

# ==============================
# 5. Funciones auxiliares de graficación
# ==============================

def plot_time_domain(t, x, y, title):
    """Compara señal original y filtrada en el tiempo."""
    plt.figure(figsize=(8, 4))
    plt.plot(t, x, label='Señal original', alpha=0.6)
    plt.plot(t, y, label='Señal filtrada', alpha=0.8)
    plt.xlim(0, 0.1)  # solo primeros 100 ms para ver detalle
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.title(title)
    plt.legend()
    plt.grid(True)


def plot_freq_response(b, a=1, fs=fs, title='Respuesta en frecuencia'):
    """Grafica la respuesta en frecuencia (magnitud) del filtro."""
    w, h = signal.freqz(b, a, worN=2048)
    f = w * fs / (2 * np.pi)
    plt.figure(figsize=(8, 4))
    plt.semilogx(f, 20 * np.log10(np.abs(h) + 1e-6))
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud [dB]')
    plt.title(title)
    plt.grid(True, which='both')


def plot_spectrum(x, fs, title):
    """Espectro de magnitud de una señal (FFT)."""
    N = len(x)
    win = np.hanning(N)
    X = np.fft.rfft(x * win)
    f = np.fft.rfftfreq(N, 1 / fs)
    plt.figure(figsize=(8, 4))
    plt.plot(f, 20 * np.log10(np.abs(X) + 1e-6))
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud [dB]')
    plt.title(title)
    plt.grid(True)


# ==============================
# 6. Ejecución principal (para generar figuras)
# ==============================

if __name__ == '__main__':
    # Espectro de la señal original
    plot_spectrum(x, fs, 'Espectro de la señal original (ruido + 50 Hz + 200 Hz)')

    # Respuestas en frecuencia FIR
    plot_freq_response(fir_lp, 1, fs, 'FIR pasa bajos (Hamming, fc=100 Hz)')
    plot_freq_response(fir_hp, 1, fs, 'FIR pasa altos (Hamming, fc=150 Hz)')
    plot_freq_response(fir_bp, 1, fs, 'FIR pasa bandas (Hamming, 80–120 Hz)')

    # Respuestas en frecuencia IIR Butterworth
    plot_freq_response(b_lp, a_lp, fs, 'IIR Butterworth pasa bajos (fc=100 Hz)')
    plot_freq_response(b_hp, a_hp, fs, 'IIR Butterworth pasa altos (fc=150 Hz)')
    plot_freq_response(b_bp, a_bp, fs, 'IIR Butterworth pasa bandas (80–120 Hz)')

    # Dominio del tiempo: ejemplos FIR vs IIR
    plot_time_domain(t, x, y_fir_lp, 'Dominio del tiempo: FIR pasa bajos')
    plot_time_domain(t, x, y_iir_lp, 'Dominio del tiempo: IIR pasa bajos')

    plot_time_domain(t, x, y_fir_hp, 'Dominio del tiempo: FIR pasa altos')
    plot_time_domain(t, x, y_iir_hp, 'Dominio del tiempo: IIR pasa altos')

    plot_time_domain(t, x, y_fir_bp, 'Dominio del tiempo: FIR pasa bandas')
    plot_time_domain(t, x, y_iir_bp, 'Dominio del tiempo: IIR pasa bandas')

    # Espectros antes y después (ejemplo con pasa bajos)
    plot_spectrum(y_fir_lp, fs, 'Espectro después de FIR pasa bajos')
    plot_spectrum(y_iir_lp, fs, 'Espectro después de IIR pasa bajos')

    plt.show()
