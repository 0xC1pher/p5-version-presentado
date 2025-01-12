## 0. **Conclusión**
Este código implementa un sistema completo de transmisión y recepción OFDM con modulación QAM, incluyendo funciones para la modulación, demodulación, estimación de canal, y simulación de BER. Cada función tiene un propósito claro y está diseñada para trabajar en conjunto con las demás. este sistema podría ser aún más robusto y eficiente.
```  **Markdown**:

```markdown
# Explicación de Funciones y Métodos

## 1. **Funciones Principales**

### **1.1. `load_image(image_path)`**
- **Propósito:** Carga una imagen desde una ruta específica.
- **Uso:** Se utiliza para cargar la imagen que se va a transmitir.
- **Parámetros:**
  - `image_path`: Ruta del archivo de imagen.
- **Retorno:** Un objeto `PIL.Image` si la imagen se carga correctamente, o `None` si no se encuentra la imagen.

```python
def load_image(image_path):
    try:
        image = Image.open(image_path)
        return image
    except FileNotFoundError:
        print(f"No se pudo encontrar la imagen en la ruta: {image_path}")
        return None
```

---

### **1.2. `convert_to_grayscale(image)`**
- **Propósito:** Convierte una imagen a escala de grises.
- **Uso:** Transforma la imagen cargada a blanco y negro para simplificar el procesamiento.
- **Parámetros:**
  - `image`: Imagen en formato `PIL.Image`.
- **Retorno:** Imagen en escala de grises.

```python
def convert_to_grayscale(image):
    imagen_bn = image.convert('L')
    return imagen_bn
```

---

### **1.3. `image_to_bits_bw(image)`**
- **Propósito:** Convierte una imagen en escala de grises a un tren de bits.
- **Uso:** Transforma los píxeles de la imagen en una secuencia de bits para su transmisión.
- **Parámetros:**
  - `image`: Imagen en escala de grises.
- **Retorno:** Un array de bits, el ancho y el alto de la imagen.

```python
def image_to_bits_bw(image):
    array = np.array(image)
    alto, ancho = array.shape    
    array = array.astype(np.uint8)
    bits = np.unpackbits(array.flatten())
    return bits, ancho, alto
```

---

### **1.4. `select_modulation()`**
- **Propósito:** Permite al usuario seleccionar el tipo de modulación (QPSK, 16-QAM, 64-QAM).
- **Uso:** Define el esquema de modulación a utilizar en la transmisión.
- **Retorno:** Un string que indica el tipo de modulación seleccionado.

```python
def select_modulation():
    print("Seleccione el tipo de modulación:")
    print("1. QPSK")
    print("2. 16-QAM")
    print("3. 64-QAM")
    choice = input("Ingrese el número de la modulación deseada (1, 2, o 3): ").strip()
    
    if choice == "1":
        return "qpsk"
    elif choice == "2":
        return "16qam"
    elif choice == "3":
        return "64qam"
    else:
        print("Opción no válida. Seleccionando QPSK por defecto.")
        return "qpsk"
```

---

### **1.5. `get_qam_constellation_mapping(modulation_type)`**
- **Propósito:** Devuelve el mapeo de constelación para el tipo de modulación seleccionado.
- **Uso:** Define cómo se mapean los bits a símbolos en la constelación QAM.
- **Parámetros:**
  - `modulation_type`: Tipo de modulación (1 para QPSK, 2 para 16-QAM, 3 para 64-QAM).
- **Retorno:** Un diccionario que mapea grupos de bits a símbolos complejos.

```python
def get_qam_constellation_mapping(modulation_type):
    if modulation_type == 1:
        # QPSK
        constellation_mapping = {
            (0,0) : -1-1j,
            (0,1) : -1+1j,
            (1,0) : 1-1j,
            (1,1) : 1+1j,
        }
    elif modulation_type == 2:
        # 16-QAM
        constellation_mapping = {
            (0,0,0,0) : -3-3j,
            (0,0,0,1) : -3-1j,
            # ... (resto de mapeos)
        }
    elif modulation_type == 3:
        # 64-QAM
        constellation_mapping = {
            (0,0,0,0,0,0) : -7-7j,
            (0,0,0,0,0,1) : -7-5j,
            # ... (resto de mapeos)
        }
    else:
        raise ValueError("Tipo de modulación no válido")
    return constellation_mapping
```

---

### **1.6. `modulate_with_mapping(bits, modulation_type)`**
- **Propósito:** Modula un tren de bits utilizando el mapeo de constelación.
- **Uso:** Transforma los bits en símbolos modulados según el esquema QAM seleccionado.
- **Parámetros:**
  - `bits`: Tren de bits a modular.
  - `modulation_type`: Tipo de modulación ("qpsk", "16qam", "64qam").
- **Retorno:** Símbolos modulados en formato complejo.

```python
def modulate_with_mapping(bits, modulation_type):
    if modulation_type == "qpsk":
        constellation_mapping = get_qam_constellation_mapping(1)
    elif modulation_type == "16qam":
        constellation_mapping = get_qam_constellation_mapping(2)
    elif modulation_type == "64qam":
        constellation_mapping = get_qam_constellation_mapping(3)
    else:
        raise ValueError("Tipo de modulación no soportado.")

    bits_per_symbol = len(next(iter(constellation_mapping.keys())))
    if len(bits) % bits_per_symbol != 0:
        bits = np.append(bits, [0] * (bits_per_symbol - len(bits) % bits_per_symbol))  

    bit_groups = bits.reshape(-1, bits_per_symbol)
    symbols = np.array([constellation_mapping[tuple(group)] for group in bit_groups])
    return symbols
```

---

### **1.7. `OFDM_symbol(QAM_payload, pilotCarriers, dataCarriers, pilotValue, nc)`**
- **Propósito:** Genera un símbolo OFDM a partir de los datos QAM.
- **Uso:** Construye un símbolo OFDM asignando los símbolos QAM a las subportadoras de datos y los valores piloto a las subportadoras piloto.
- **Parámetros:**
  - `QAM_payload`: Símbolos QAM para las subportadoras de datos.
  - `pilotCarriers`: Índices de las subportadoras piloto.
  - `dataCarriers`: Índices de las subportadoras de datos.
  - `pilotValue`: Valor asignado a las subportadoras piloto.
  - `nc`: Número total de subportadoras.
- **Retorno:** Un símbolo OFDM en el dominio de la frecuencia.

```python
def OFDM_symbol(QAM_payload, pilotCarriers, dataCarriers, pilotValue, nc):
    symbol = np.zeros(nc, dtype=complex)
    symbol[pilotCarriers] = pilotValue
    if len(QAM_payload) < len(dataCarriers):
        QAM_payload = np.pad(QAM_payload, (0, len(dataCarriers) - len(QAM_payload)), 'constant', constant_values=0)
    symbol[dataCarriers] = QAM_payload[:len(dataCarriers)]
    return symbol
```

---

### **1.8. `IDFT(OFDM_data)`**
- **Propósito:** Aplica la Transformada Inversa de Fourier (IDFT) a los datos OFDM.
- **Uso:** Convierte los símbolos OFDM del dominio de la frecuencia al dominio del tiempo.
- **Parámetros:**
  - `OFDM_data`: Símbolos OFDM en el dominio de la frecuencia.
- **Retorno:** Símbolos OFDM en el dominio del tiempo.

```python
def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data) * np.sqrt(len(OFDM_data))
```

---

### **1.9. `addCP(OFDM_time, CP)`**
- **Propósito:** Añade un prefijo cíclico (CP) a un símbolo OFDM.
- **Uso:** Mejora la robustez del sistema frente a la interferencia entre símbolos (ISI).
- **Parámetros:**
  - `OFDM_time`: Símbolo OFDM en el dominio del tiempo.
  - `CP`: Longitud del prefijo cíclico.
- **Retorno:** Símbolo OFDM con prefijo cíclico.

```python
def addCP(OFDM_time, CP):
    cp = OFDM_time[-CP:]
    return np.hstack([cp, OFDM_time])
```

---

### **1.10. `channel(ofdm_signal, snr_db, noise_prob=0.1)`**
- **Propósito:** Simula un canal con ruido aleatorio y no uniforme.
- **Uso:** Añade ruido a la señal OFDM para simular condiciones reales de transmisión.
- **Parámetros:**
  - `ofdm_signal`: Señal OFDM en el dominio del tiempo.
  - `snr_db`: Relación señal-ruido en decibelios (SNR).
  - `noise_prob`: Probabilidad de que un símbolo se vea altamente afectado por ruido.
- **Retorno:** Señal OFDM afectada por ruido.

```python
def channel(ofdm_signal, snr_db, noise_prob=0.1):
    channelResponse = np.array([1, 0, 0.3 + 0.3j])
    convolved = np.convolve(ofdm_signal, channelResponse)
    signal_power = np.mean(np.abs(convolved)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(ofdm_signal)) + 1j * np.random.randn(len(ofdm_signal)))
    high_noise_indices = np.random.choice([0, 1], size=len(ofdm_signal), p=[1 - noise_prob, noise_prob])
    high_noise = high_noise_indices * (np.random.randn(len(ofdm_signal)) + 1j * np.random.randn(len(ofdm_signal))) * 10 * np.sqrt(noise_power)
    received_signal = ofdm_signal + noise + high_noise
    return received_signal
```

---

## 2. **Funciones de Decodificación**

### **2.1. `removeCP(signal, CP, N)`**
- **Propósito:** Elimina el prefijo cíclico de un símbolo OFDM.
- **Uso:** Prepara la señal para la transformada de Fourier.
- **Parámetros:**
  - `signal`: Símbolo OFDM con prefijo cíclico.
  - `CP`: Longitud del prefijo cíclico.
  - `N`: Tamaño del símbolo OFDM sin prefijo.
- **Retorno:** Símbolo OFDM sin prefijo cíclico.

```python
def removeCP(signal, CP, N):
    return signal[CP:CP+N]
```

---

### **2.2. `DFT(OFDM_RX)`**
- **Propósito:** Aplica la Transformada de Fourier (DFT) a un símbolo OFDM.
- **Uso:** Convierte la señal OFDM del dominio del tiempo al dominio de la frecuencia.
- **Parámetros:**
  - `OFDM_RX`: Símbolo OFDM en el dominio del tiempo.
- **Retorno:** Símbolo OFDM en el dominio de la frecuencia.

```python
def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)
```

---

### **2.3. `channelEstimate(OFDM_demod, pilotCarriers, pilotValue, allCarriers, H_exact)`**
- **Propósito:** Estima la respuesta del canal usando las subportadoras piloto.
- **Uso:** Corrige los efectos del canal en la señal recibida.
- **Parámetros:**
  - `OFDM_demod`: Símbolo OFDM en el dominio de la frecuencia.
  - `pilotCarriers`: Índices de las subportadoras piloto.
  - `pilotValue`: Valor transmitido en las subportadoras piloto.
  - `allCarriers`: Índices de todas las subportadoras.
  - `H_exact`: Respuesta exacta del canal.
- **Retorno:** Estimación del canal en las subportadoras piloto y en todas las subportadoras.

```python
def channelEstimate(OFDM_demod, pilotCarriers, pilotValue, allCarriers, H_exact):
    pilotCarriers = pilotCarriers[pilotCarriers < len(OFDM_demod)]
    pilots = OFDM_demod[pilotCarriers]
    Hest_at_pilots = pilots / pilotValue
    Hest_abs = interp1d(pilotCarriers, np.abs(Hest_at_pilots), kind='linear', fill_value="extrapolate")(np.arange(len(OFDM_demod)))
    Hest_phase = interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear', fill_value="extrapolate")(np.arange(len(OFDM_demod)))
    Hest = Hest_abs * np.exp(1j * Hest_phase)
    return Hest_at_pilots, Hest
```

---

### **2.4. `equalize(OFDM_demod, Hest)`**
- **Propósito:** Corrige las subportadoras activas usando la estimación del canal.
- **Uso:** Compensa los efectos del canal en la señal recibida.
- **Parámetros:**
  - `OFDM_demod`: Símbolo OFDM en el dominio de la frecuencia.
  - `Hest`: Estimación del canal.
- **Retorno:** Subportadoras activas corregidas.

```python
def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest
```

---

### **2.5. `get_payload(equalized, dataCarriers)`**
- **Propósito:** Extrae las subportadoras de datos del símbolo equalizado.
- **Uso:** Recupera los símbolos QAM correspondientes a los datos transmitidos.
- **Parámetros:**
  - `equalized`: Subportadoras equalizadas.
  - `dataCarriers`: Índices de las subportadoras de datos.
- **Retorno:** Símbolos QAM correspondientes a los datos.

```python
def get_payload(equalized, dataCarriers):
    valid_dataCarriers = dataCarriers[dataCarriers < len(equalized)]
    return equalized[valid_dataCarriers]
```

---

### **2.6. `Demapping(QAM, demapping_table)`**
- **Propósito:** Realiza el demapeo de los símbolos QAM a bits.
- **Uso:** Convierte los símbolos QAM recibidos en bits.
- **Parámetros:**
  - `QAM`: Símbolos QAM recuperados.
  - `demapping_table`: Tabla de mapeo inverso (símbolos a bits).
- **Retorno:** Bits recuperados y decisiones duras.

```python
def Demapping(QAM, demapping_table):
    constellation = np.array([x for x in demapping_table.keys()])
    dists = abs(QAM.reshape((-1, 1)) - constellation.reshape((1, -1)))
    const_index = dists.argmin(axis=1)
    hardDecision = constellation[const_index]
    PS_est = np.vstack([demapping_table[C] for C in hardDecision])
    return PS_est, hardDecision
```

---

### **2.7. `reconstruct_image_from_bits(bits, width, height)`**
- **Propósito:** Reconstruye una imagen a partir de una secuencia de bits.
- **Uso:** Convierte los bits recuperados en una imagen en escala de grises.
- **Parámetros:**
  - `bits`: Secuencia de bits recuperados.
  - `width`: Ancho de la imagen original.
  - `height`: Alto de la imagen original.
- **Retorno:** Imagen reconstruida en formato de matriz.

```python
def reconstruct_image_from_bits(bits, width, height):
    total_bits = width * height * 8
    if len(bits) < total_bits:
        padding = np.zeros(total_bits - len(bits), dtype=int)
        bits = np.concatenate((bits, padding))
    bits = bits[:total_bits]
    bytes_array = np.packbits(bits)
    image_matrix = bytes_array.reshape((height, width))
    return image_matrix, bits
```

---

## 3. **Funciones de Simulación y Visualización**

### **3.1. `monte_carlo_simulation(...)`**
- **Propósito:** Realiza una simulación Monte Carlo para calcular el BER de diferentes modulaciones y valores de SNR.
- **Uso:** Evalúa el rendimiento del sistema bajo diferentes condiciones de ruido.
- **Parámetros:** Varios, incluyendo el número de subportadoras, bits de la imagen, y parámetros del canal.
- **Retorno:** Resultados del BER para cada modulación y SNR.

```python
def monte_carlo_simulation(nc, bitsbn, H_exact, allCarriers, pilotCarriers, dataCarriers, pilotValue, n, modulation_types, SNR_range, channelResponse, CP):
    ber_results = {mod: [] for mod in modulation_types}
    # ... (código de simulación)
    return ber_results
```

---

### **3.2. `plot_ber_vs_snr(ber_results, snr_range)`**
- **Propósito:** Grafica el BER vs SNR para cada modulación.
- **Uso:** Visualiza el rendimiento del sistema en función del SNR.
- **Parámetros:**
  - `ber_results`: Resultados del BER para cada modulación.
  - `snr_range`: Rango de valores de SNR.
- **Retorno:** Ninguno (muestra una gráfica).

```python
def plot_ber_vs_snr(ber_results, snr_range):
    plt.figure(figsize=(10, 6))
    for modulation, ber_values in ber_results.items():
        plt.semilogy(snr_range, ber_values, label=modulation.upper())
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("BER vs SNR para Modulaciones")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()
```

---
Si necesitas más detalles sobre alguna función en particular,me escribes. 😊
