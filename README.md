# actividad3-filtros-digitales
Implementación de filtros digitales FIR e IIR en Python para una señal con ruido.
# Actividad Formativa 3: Implementación y Evaluación de Filtros Digitales

Este repositorio contiene la implementación en Python de filtros digitales
FIR e IIR (pasa bajos, pasa altos y pasa bandas) para procesar una señal
compuesta por dos sinusoides contaminadas con ruido blanco.

## 🧪 Descripción

- Lenguaje: Python 3.x  
- Librerías: NumPy, SciPy, Matplotlib  
- Señal de prueba:
  - Frecuencia de muestreo: 1000 Hz
  - Componentes: 50 Hz, 200 Hz y ruido blanco gaussiano

Se diseñan filtros:

- FIR con ventana de Hamming:
  - Pasa bajos (fc = 100 Hz)
  - Pasa altos (fc = 150 Hz)
  - Pasa bandas (80–120 Hz)
- IIR Butterworth (orden 4) con las mismas frecuencias de corte.

Se comparan las señales antes y después del filtrado en el dominio del tiempo
y en el dominio de la frecuencia.

## 📂 Archivos

- `actividad3_filtros.py`: código principal de la actividad.
- `imagenes/`: gráficas generadas (espectros y respuestas en el tiempo) *(opcional)*.
- `presentacion/`: archivo PDF de la presentación *(opcional)*.

## ▶️ Ejecución

1. Clonar o descargar este repositorio.
2. Instalar dependencias:

```bash
pip install numpy scipy matplotlib

