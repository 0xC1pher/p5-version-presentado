# Changelog



Este archivo documenta los cambios realizados en el código del proyecto de simulación de comunicación OFDM con codificación de canal.
del archivo 

# py-version-presentado.py 

```
---

## [1.1.0] - 2025-01-10

### Añadido
- **Codificación Convolucional**:
  - Se implementaron las funciones `convolutional_encoding` y `convolutional_decoding` para aplicar codificación y decodificación convolucional a los bits de la imagen.
  - La codificación convolucional se añadió antes de la modulación QAM/PSK.
  - La decodificación convolucional se añadió después de la demodulación y el demapping.

- **Turbo Codes**:
  - Se implementaron las funciones `turbo_encoding` y `turbo_decoding` para simular la codificación y decodificación Turbo utilizando dos codificadores convolucionales.
  - La codificación Turbo se añadió después de la codificación convolucional y antes de la modulación.
  - La decodificación Turbo se añadió después de la demodulación y antes de la decodificación convolucional.

- **Integración en el Flujo Principal**:
  - Se integraron las funciones de codificación y decodificación en el flujo principal del código, manteniendo la coherencia del sistema.
  - Se añadió la lógica para manejar los bits codificados y decodificados en el proceso de transmisión y recepción.

### Cambios
- **Mejoras en la Simulación**:
  - Se ajustó el cálculo del BER para incluir los efectos de la codificación y decodificación.
  - Se mejoró la visualización de la constelación y la imagen reconstruida.

---

## [1.0.0] - 2025-01-10

### Añadido
- **Funcionalidad Básica**:
  - Carga y conversión de imágenes a escala de grises.
  - Conversión de imágenes a bits y viceversa.
  - Modulación QPSK, 16-QAM y 64-QAM.
  - Simulación de transmisión OFDM con prefijo cíclico (CP).
  - Estimación y ecualización del canal.
  - Demodulación y demapping de símbolos QAM.
  - Reconstrucción de la imagen a partir de los bits recibidos.

- **Simulación Monte Carlo**:
  - Implementación de una simulación Monte Carlo para calcular el BER en función del SNR.
  - Soporte para múltiples tipos de modulación (QPSK, 16-QAM, 64-QAM).

- **Visualización**:
  - Gráficos de la constelación modulada y recibida.
  - Comparación de la imagen original y reconstruida.
  - Gráfico de BER vs SNR para diferentes modulaciones.

---

## [0.1.0] - 2025-01-10

### Añadido
- **Estructura Inicial del Proyecto**:
  - Implementación inicial de funciones para carga de imágenes, modulación y demodulación.

---
```
# p6 new release

```
---

# Changelog

Este archivo documenta los cambios realizados al código original para cumplir con los requisitos del proyecto.

---

## [0.1.0] - 2025-01-12

### Cambios realizados

#### Nuevas funciones agregadas:
1. **`get_bandwidth_and_spacing`**:
   - Solicita al usuario el ancho de banda (`BW`) y el espaciado entre subportadoras (`Δf`).
   - Valida que el espaciado sea `15 KHz` o `7.5 KHz`.

2. **`get_snr`**:
   - Solicita al usuario el valor de SNR en dB.
   - Valida que el valor de SNR sea mayor o igual a 0.

3. **`get_longitud_CP`**:
   - Solicita al usuario la longitud del prefijo cíclico (normal o extendido).
   - Ajusta el valor del prefijo cíclico según el espaciado entre subportadoras (`Δf`).

4. **`get_porcentaje_pilotos`**:
   - Solicita al usuario el porcentaje de subportadoras que serán piloto.
   - Valida que el porcentaje esté entre 0 y 100.

5. **`distribuir_pilotos_uniformemente`**:
   - Distribuye uniformemente las subportadoras piloto en el rango total de subportadoras.
   - Calcula los índices de las subportadoras piloto y de datos.

6. **`monte_carlo_simulation`**:
   - Realiza una simulación Monte Carlo para calcular el BER de diferentes modulaciones y valores de SNR.
   - Genera gráficas de BER vs SNR para QPSK, 16-QAM y 64-QAM.

#### Mejoras y ajustes:
1. **Validación de entradas**:
   - Se agregaron bucles `while` y manejo de excepciones en las funciones que solicitan entradas por teclado.
   - Mensajes de error claros cuando el usuario ingresa valores no válidos.

2. **Integración en el flujo principal**:
   - Se aseguró que todas las funciones se llamen en el orden correcto en el flujo principal (`main`).
   - Se agregaron mensajes de depuración para facilitar la comprensión del código.

3. **Cálculo de la IFFT**:
   - Se mejoró la función `calculate_ifft_size` para calcular el tamaño de la IFFT como la potencia de 2 más cercana que sea mayor o igual a `N_c`.

4. **Simulación del canal**:
   - Se mejoró la función `channel` para simular un canal móvil con ruido aleatorio y no uniforme.

5. **Reconstrucción de la imagen**:
   - Se mejoró la función `reconstruct_image_from_bits` para manejar casos en los que faltan bits.

#### Resultados esperados:
- **Imagen en blanco y negro transmitida**: Se muestra la imagen original en blanco y negro.
- **Imagen reconstruida**: Se muestra la imagen reconstruida después de la transmisión.
- **Gráfica de BER vs SNR**: Se muestra la gráfica de BER vs SNR para las diferentes modulaciones.

---

## [0.1.0] - 2025-01-12

### Versión inicial
- Código original proporcionado.
- Funciones básicas para cargar una imagen, convertirla a blanco y negro, y transformarla a bits.
- Funciones para modular y demodular señales OFDM.
- Funciones para agregar y eliminar el prefijo cíclico (CP).
- Funciones para estimar el canal y ecualizar las señales.

---

## Notas
- **No se eliminó ninguna función del código original**.
- Todas las funciones originales se mantuvieron intactas.
- Se agregaron nuevas funciones y se realizaron ajustes menores para mejorar la claridad y la funcionalidad.

---
```
# p7 new release

```
---
# Changelog

Este archivo documenta los cambios realizados en el código del proyecto de simulación de comunicación OFDM con codificación de canal.
del archivo 

# py-version-presentado.py 

---

## [1.2.0] - 2025-01-11

### Añadido
- **Implementación de SC-FDM**:
  - Se añadió la funcionalidad de **pre-codificación (DFT)** antes de la modulación OFDM para convertir los símbolos del dominio del tiempo al dominio de la frecuencia.
  - Se añadió la funcionalidad de **post-codificación (IDFT)** después de la demodulación OFDM para convertir los símbolos de vuelta al dominio del tiempo.
  - Se implementaron las funciones `apply_dft` y `apply_idft` para manejar estas transformaciones.

- **Cálculo del PAPR**:
  - Se añadió la función `calculate_papr` para calcular el **Peak-to-Average Power Ratio (PAPR)** de las señales OFDM y SC-FDM.
  - Se implementó la función `plot_ccdf` para graficar la **Complementary Cumulative Distribution Function (CCDF)** del PAPR y comparar OFDM y SC-FDM.

- **Integración en el Flujo Principal**:
  - Se integraron las funciones de pre-codificación y post-codificación en el flujo principal del código.
  - Se añadió la lógica para calcular y graficar el PAPR en el proceso de transmisión y recepción.

---

## [1.1.0] - 2025-01-10

### Añadido
- **Codificación Convolucional**:
  - Se implementaron las funciones `convolutional_encoding` y `convolutional_decoding` para aplicar codificación y decodificación convolucional a los bits de la imagen.
  - La codificación convolucional se añadió antes de la modulación QAM/PSK.
  - La decodificación convolucional se añadió después de la demodulación y el demapping.

- **Turbo Codes**:
  - Se implementaron las funciones `turbo_encoding` y `turbo_decoding` para simular la codificación y decodificación Turbo utilizando dos codificadores convolucionales.
  - La codificación Turbo se añadió después de la codificación convolucional y antes de la modulación.
  - La decodificación Turbo se añadió después de la demodulación y antes de la decodificación convolucional.

- **Integración en el Flujo Principal**:
  - Se integraron las funciones de codificación y decodificación en el flujo principal del código, manteniendo la coherencia del sistema.
  - Se añadió la lógica para manejar los bits codificados y decodificados en el proceso de transmisión y recepción.

### Cambios
- **Mejoras en la Simulación**:
  - Se ajustó el cálculo del BER para incluir los efectos de la codificación y decodificación.
  - Se mejoró la visualización de la constelación y la imagen reconstruida.

---

## [1.0.0] - 2025-01-10

### Añadido
- **Funcionalidad Básica**:
  - Carga y conversión de imágenes a escala de grises.
  - Conversión de imágenes a bits y viceversa.
  - Modulación QPSK, 16-QAM y 64-QAM.
  - Simulación de transmisión OFDM con prefijo cíclico (CP).
  - Estimación y ecualización del canal.
  - Demodulación y demapping de símbolos QAM.
  - Reconstrucción de la imagen a partir de los bits recibidos.

- **Simulación Monte Carlo**:
  - Implementación de una simulación Monte Carlo para calcular el BER en función del SNR.
  - Soporte para múltiples tipos de modulación (QPSK, 16-QAM, 64-QAM).

- **Visualización**:
  - Gráficos de la constelación modulada y recibida.
  - Comparación de la imagen original y reconstruida.
  - Gráfico de BER vs SNR para diferentes modulaciones.

---

## [0.1.0] - 2025-01-10

### Añadido
- **Estructura Inicial del Proyecto**:
  - Implementación inicial de funciones para carga de imágenes, modulación y demodulación.

---

## Notas
- Este archivo `CHANGELOG.md` sigue el formato estándar de [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/).
- Las versiones siguen el esquema de [Versionado Semántico](https://semver.org/lang/es/).
