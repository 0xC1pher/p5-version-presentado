import matplotlib
matplotlib.use('TkAgg')  # Establece un backend que permite mostrar ventanas
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
from scipy.interpolate import interp1d
from commpy.filters import rcosfilter
from commpy.channelcoding import Trellis, conv_encode, viterbi_decode

def load_image(image_path):
    """
    Carga una imagen desde una ruta específica.
    """
    try:
        image = Image.open(image_path)
        return image
    except FileNotFoundError:
        print(f"No se pudo encontrar la imagen en la ruta: {image_path}")
        return None

def convert_to_grayscale(image):
    """
    Convierte una imagen a escala de grises.
    """
    imagen_bn = image.convert('L')
    return imagen_bn

def image_to_bits_bw(image):
    """
    Convierte una imagen en blanco y negro (escala de grises) a un tren de bits.
    
    Args:
        image (PIL.Image): Imagen en blanco y negro.
    
    Returns:
        numpy array: Tren de bits (0 y 1).
    """
    # Convertir la imagen a un array numpy
    array = np.array(image)
    # Obtener dimensiones de imagen en blanco y negro
    alto, ancho = array.shape    
    
    # Asegurar que los valores sean de 8 bits
    array = array.astype(np.uint8)

    # Convertir cada valor de 0-255 a 8 bits
    bits = np.unpackbits(array.flatten())

    return bits, ancho, alto

def select_modulation():
    """
    Permite al usuario elegir el tipo de modulación por teclado.
    """
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

def get_qam_constellation_mapping(modulation_type):
    if modulation_type == 1:
        # QAM
        constellation_mapping ={
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
            (0,0,1,0) : -3+3j,
            (0,0,1,1) : -3+1j,
            (0,1,0,0) : -1-3j,
            (0,1,0,1) : -1-1j,
            (0,1,1,0) : -1+3j,
            (0,1,1,1) : -1+1j,
            (1,0,0,0) :  3-3j,
            (1,0,0,1) :  3-1j,
            (1,0,1,0) :  3+3j,
            (1,0,1,1) :  3+1j,
            (1,1,0,0) :  1-3j,
            (1,1,0,1) :  1-1j,
            (1,1,1,0) :  1+3j,
            (1,1,1,1) :  1+1j
        }
    elif modulation_type == 3:
        # 64-QAM
        constellation_mapping ={
            (0,0,0,0,0,0) : -7-7j,
            (0,0,0,0,0,1) : -7-5j,
            (0,0,0,0,1,0) : -7-3j,
            (0,0,0,0,1,1) : -7-1j,
            (0,0,0,1,0,0) : -7+1j,
            (0,0,0,1,0,1) : -7+3j,
            (0,0,0,1,1,0) : -7+5j,
            (0,0,0,1,1,1) : -7+7j,
            (0,0,1,0,0,0) : -5-7j,
            (0,0,1,0,0,1) : -5-5j,
            (0,0,1,0,1,0) : -5-3j,
            (0,0,1,0,1,1) : -5-1j,
            (0,0,1,1,0,0) : -5+1j,
            (0,0,1,1,0,1) : -5+3j,
            (0,0,1,1,1,0) : -5+5j,
            (0,0,1,1,1,1) : -5+7j,
            (0,1,0,0,0,0) : -3-7j,
            (0,1,0,0,0,1) : -3-5j,
            (0,1,0,0,1,0) : -3-3j,
            (0,1,0,0,1,1) : -3-1j,
            (0,1,0,1,0,0) : -3+1j,
            (0,1,0,1,0,1) : -3+3j,
            (0,1,0,1,1,0) : -3+5j,
            (0,1,0,1,1,1) : -3+7j,
            (0,1,1,0,0,0) : -1-7j,
            (0,1,1,0,0,1) : -1-5j,
            (0,1,1,0,1,0) : -1-3j,
            (0,1,1,0,1,1) : -1-1j,
            (0,1,1,1,0,0) : -1+1j,
            (0,1,1,1,0,1) : -1+3j,
            (0,1,1,1,1,0) : -1+5j,
            (0,1,1,1,1,1) : -1+7j,
            (1,0,0,0,0,0) : +1-7j,
            (1,0,0,0,0,1) : +1-5j,
            (1,0,0,0,1,0) : +1-3j,
            (1,0,0,0,1,1) : +1-1j,
            (1,0,0,1,0,0) : +1+1j,
            (1,0,0,1,0,1) : +1+3j,
            (1,0,0,1,1,0) : +1+5j,
            (1,0,0,1,1,1) : +1+7j,
            (1,0,1,0,0,0) : +3-7j,
            (1,0,1,0,0,1) : +3-5j,
            (1,0,1,0,1,0) : +3-3j,
            (1,0,1,0,1,1) : +3-1j,
            (1,0,1,1,0,0) : +3+1j,
            (1,0,1,1,0,1) : +3+3j,
            (1,0,1,1,1,0) : +3+5j,
            (1,0,1,1,1,1) : +3+7j,
            (1,1,0,0,0,0) : +5-7j,
            (1,1,0,0,0,1) : +5-5j,
            (1,1,0,0,1,0) : +5-3j,
            (1,1,0,0,1,1) : +5-1j,
            (1,1,0,1,0,0) : +5+1j,
            (1,1,0,1,0,1) : +5+3j,
            (1,1,0,1,1,0) : +5+5j,
            (1,1,0,1,1,1) : +5+7j,
            (1,1,1,0,0,0) : +7-7j,
            (1,1,1,0,0,1) : +7-5j,
            (1,1,1,0,1,0) : +7-3j,
            (1,1,1,0,1,1) : +7-1j,
            (1,1,1,1,0,0) : +7+1j,
            (1,1,1,1,0,1) : +7+3j,
            (1,1,1,1,1,0) : +7+5j,
            (1,1,1,1,1,1) : +7+7j
        }
    else:
        raise ValueError("Tipo de modulación no válido")
    return constellation_mapping

def modulate_with_mapping(bits, modulation_type):
    """
    Modula un tren de bits utilizando un mapeo explícito de constelación.

    Args:
        bits (numpy array): Tren de bits de la imagen.
        modulation_type (str): Tipo de modulación ("qpsk", "16qam", "64qam").

    Returns:
        numpy array: Símbolos modulados.
    """
    print(modulation_type)
    
    # Convertir el tipo de modulación al índice para la constelación
    if modulation_type == "qpsk":
        constellation_mapping = get_qam_constellation_mapping(1)
    elif modulation_type == "16qam":
        constellation_mapping = get_qam_constellation_mapping(2)
    elif modulation_type == "64qam":
        constellation_mapping = get_qam_constellation_mapping(3)
    else:
        raise ValueError("Tipo de modulación no soportado.")

    # Determinar el número de bits por símbolo
    bits_per_symbol = len(next(iter(constellation_mapping.keys())))

    # Asegurarse de que el número de bits sea divisible por bits_per_symbol. Realiza cero padding
    if len(bits) % bits_per_symbol != 0:
        bits = np.append(bits, [0] * (bits_per_symbol - len(bits) % bits_per_symbol))  

    # Dividir los bits en grupos de tamaño bits_per_symbol
    bit_groups = bits.reshape(-1, bits_per_symbol)

    # Mapear cada grupo de bits a un símbolo correspondiente al diccionario de constelacion
    symbols = np.array([constellation_mapping[tuple(group)] for group in bit_groups])
    
    return symbols

def apply_mimo(symbols, num_tx_antennas, num_rx_antennas):
    """
    Aplica multiplexación espacial MIMO para transmitir flujos de datos en paralelo.
    
    Args:
        symbols (numpy array): Símbolos modulados.
        num_tx_antennas (int): Número de antenas en transmisor.
        num_rx_antennas (int): Número de antenas en receptor.
    
    Returns:
        numpy array: Símbolos transmitidos desde cada antena.
    """
    # Número máximo de flujos independientes
    num_streams = min(num_tx_antennas, num_rx_antennas)
    
    # Dividir los símbolos en flujos independientes
    symbols_split = np.array_split(symbols, num_streams)
    
    # Transmitir cada flujo desde una antena diferente
    mimo_symbols = np.zeros((num_tx_antennas, len(symbols_split[0])), dtype=complex)
    for i in range(num_streams):
        mimo_symbols[i] = symbols_split[i]
    
    return mimo_symbols

def get_bandwidth_and_spacing():
    """
    Solicita al usuario el ancho de banda y el espaciado entre subportadoras.

    Returns:
        tuple: Ancho de banda (BW) y espaciado (\Delta f) en Hz.
    """
    bw = float(input("Ingrese el ancho de banda (Hz): "))
    delta_f = float(input("Ingrese el espaciado entre subportadoras (Hz): "))
    return bw, delta_f

def get_longitud_CP():
    
    longitud_CP = float(input("Escoger Prefijo ciclico: 1. Normal 2. Extendido: "))
    
    return longitud_CP

def get_snr():
    snr_dB = float(input("Ingresar el valor de SNR en dB: "))
    
    return snr_dB

def calculate_nc(bw, delta_f):
    """
    Calcula el número de subportadoras activas (N_c) a partir del ancho de banda y el espaciado entre subportadoras.

    Args:
        bw (float): Ancho de banda en Hz.
        delta_f (float): Espaciado entre subportadoras en Hz.

    Returns:
        int: Número de subportadoras activas (N_c).
    """
    return bw / delta_f

def OFDM_symbol(QAM_payload, pilotCarriers, dataCarriers, pilotValue, nc):
    """
    Genera un símbolo OFDM a partir de los datos QAM, las subportadoras piloto y datos.
    Args:
        QAM_payload: Datos QAM para las subportadoras de datos.
        pilotCarriers: Índices de las subportadoras piloto.
        dataCarriers: Índices de las subportadoras de datos.
        pilotValue: Valor asignado a las subportadoras piloto.
        nc: Número total de subportadoras (FFT size).
    Returns:
        symbol: Vector con las subportadoras OFDM.
    """
    symbol = np.zeros(nc, dtype=complex)  # Inicializar todas las subportadoras en 0
    symbol[pilotCarriers] = pilotValue  # Asignar valores a las subportadoras piloto
    
    # Verificar si el tamaño del QAM_payload coincide con las subportadoras de datos
    if len(QAM_payload) < len(dataCarriers):
        # Rellenar con ceros si QAM_payload tiene menos datos
        QAM_payload = np.pad(QAM_payload, (0, len(dataCarriers) - len(QAM_payload)), 'constant', constant_values=0)
    
    # Asignar los símbolos QAM a las subportadoras de datos
    symbol[dataCarriers] = QAM_payload[:len(dataCarriers)]  # Cortar si hay datos sobrantes
    
    return symbol

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data) * np.sqrt(len(OFDM_data))

def addCP(OFDM_time,CP):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

def calculate_ifft_size(nc):
    """
    Calcula el tamaño de la IFFT (N) como la potencia de 2 más cercana que sea >= N_c.

    Args:
        nc (int): Número de subportadoras activas (N_c).

    Returns:
        int: Tamaño de la IFFT (N).
    """

    m = 0
    potencia_de_dos = 2 ** m

    while potencia_de_dos <= nc:  #128
        m += 1
        potencia_de_dos = 2 ** m

    return potencia_de_dos

def removeCP(signal, CP, N):
    """
    Elimina el prefijo cíclico de un símbolo OFDM.

    Args:
        signal (numpy array): Símbolo OFDM con prefijo cíclico.
        CP (int): Longitud del prefijo cíclico en muestras.
        N (int): Tamaño del símbolo OFDM sin prefijo.

    Returns:
        numpy array: Símbolo OFDM sin prefijo cíclico.
    """
    return signal[CP:CP+N]

def DFT(OFDM_RX):
    """
    Aplica la FFT a un símbolo OFDM.

    Args:
        OFDM_RX (numpy array): Símbolo OFDM en el dominio del tiempo.

    Returns:
        numpy array: Símbolo OFDM en el dominio de la frecuencia.
    """
    return np.fft.fft(OFDM_RX)

def channelEstimate(OFDM_demod, pilotCarriers, pilotValue, allCarriers, H_exact):
    """
    Estima la respuesta del canal usando las subportadoras piloto.

    Args:
        OFDM_demod (numpy array): Símbolo OFDM en el dominio de la frecuencia.
        pilotCarriers (numpy array): Índices de las subportadoras piloto.
        pilotValue (complex): Valor transmitido en las subportadoras piloto.
        allCarriers (numpy array): Índices de todas las subportadoras.
        H_exact (numpy array): Canal exacto calculado.

    Returns:
        tuple: Estimación del canal en las subportadoras piloto y en todas las subportadoras.
    """

    # Filtrar índices de subportadoras piloto que estén dentro del rango
    pilotCarriers = pilotCarriers[pilotCarriers < len(OFDM_demod)]
    
    # Valores recibidos en las subportadoras piloto
    pilots = OFDM_demod[pilotCarriers]
    
    # Estimar la respuesta del canal en las subportadoras piloto
    Hest_at_pilots = pilots / pilotValue
    
    # Interpolar para estimar el canal en todas las subportadoras
    Hest_abs = interp1d(pilotCarriers, np.abs(Hest_at_pilots), kind='linear', fill_value="extrapolate")(np.arange(len(OFDM_demod)))
    Hest_phase = interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear', fill_value="extrapolate")(np.arange(len(OFDM_demod)))
    Hest = Hest_abs * np.exp(1j * Hest_phase)
    
    return Hest_at_pilots, Hest

def equalize(OFDM_demod, Hest):
    """
    Corrige las subportadoras activas usando la estimación del canal.

    Args:
        OFDM_demod (numpy array): Símbolo OFDM en el dominio de la frecuencia.
        Hest (numpy array): Estimación del canal.

    Returns:
        numpy array: Subportadoras activas corregidas.
    """
    return OFDM_demod / Hest

def get_payload(equalized, dataCarriers):
    """
    Extrae las subportadoras de datos del símbolo equalizado.

    Args:
        equalized (numpy array): Subportadoras equalizadas.
        dataCarriers (numpy array): Índices de las subportadoras de datos.

    Returns:
        numpy array: Símbolos QAM correspondientes a los datos.
    """
    # Ajustar índices para asegurarse de que están dentro del rango
    valid_dataCarriers = dataCarriers[dataCarriers < len(equalized)]
    return equalized[valid_dataCarriers]

def Demapping(QAM, demapping_table):
    """
    Realiza el demapping de los símbolos QAM estimados a bits utilizando operaciones vectorizadas.

    Args:
        QAM (numpy array): Símbolos QAM recuperados.
        demapping_table (dict): Tabla de mapeo inverso (símbolos a bits).

    Returns:
        tuple: PS_est (símbolos mapeados a bits en forma de bits), hardDecision (decisiones duras).
    """
    # Array de los puntos de la constelación
    constellation = np.array([x for x in demapping_table.keys()])
    
    # Calcular la distancia de cada punto recibido al punto de la constelación más cercano
    dists = abs(QAM.reshape((-1, 1)) - constellation.reshape((1, -1)))

    # Índices del punto de la constelación más cercano para cada símbolo recibido
    const_index = dists.argmin(axis=1)

    # Decisión dura: símbolos QAM más cercanos
    hardDecision = constellation[const_index]

    # Transformar los puntos QAM en grupos de bits
    PS_est = np.vstack([demapping_table[C] for C in hardDecision])

    return PS_est, hardDecision

def Serie_paralelo(symbols, block_size):
    """
    Divide los símbolos QAM en bloques para las subportadoras de datos.
    # """
    # num_blocks = len(symbols) // block_size
    # grupos = symbols[:num_blocks * block_size].reshape(-1, block_size)
    
    longitud_datos = len(symbols)
    num_blocks = (longitud_datos + block_size - 1) // block_size
    total_elements = num_blocks * block_size
    
    # Rellenar con ceros si es necesario
    if total_elements > longitud_datos:
        symbols = np.append(symbols, np.zeros(total_elements - longitud_datos, dtype=symbols.dtype))
    
    # Dividir en bloques
    grupos = symbols.reshape(num_blocks, block_size)
    
    return grupos

def reconstruct_image_from_bits(bits, width, height):
    """
    Reconstruye una imagen a partir de una secuencia de bits. 
    Si faltan bits, los rellena con ceros.
    
    Parameters:
        bits (numpy array): Secuencia de bits (0 y 1) recuperados.
        width (int): Ancho de la imagen original.
        height (int): Alto de la imagen original.
    
    Returns:
        numpy array: Imagen reconstruida en formato de matriz.
    """
    # Número total de bits necesarios
    total_bits = width * height * 8  # Cada píxel tiene 8 bits
    
    # Si los bits recuperados son menores al total, rellenar con ceros
    if len(bits) < total_bits:
        print(f"Faltan {total_bits - len(bits)} bits. Rellenando con ceros.")
        padding = np.zeros(total_bits - len(bits), dtype=int)
        bits = np.concatenate((bits, padding))
    
    # Truncar si hay más bits de los necesarios
    bits = bits[:total_bits]
    
    # Asegurar que los bits sean un múltiplo de 8
    if len(bits) % 8 != 0:
        raise ValueError("La cantidad de bits no es múltiplo de 8 incluso después del ajuste.")
    
    # Convertir los bits a bytes
    bytes_array = np.packbits(bits)
    
    # Reshape a las dimensiones de la imagen
    image_matrix = bytes_array.reshape((height, width))
    
    return image_matrix, bits

def monte_carlo_simulation(nc, bitsbn, H_exact, allCarriers, pilotCarriers, dataCarriers, pilotValue, n, modulation_types, SNR_range, channelResponse, CP):
    """
    Realiza una simulación Monte Carlo para calcular el BER de diferentes modulaciones y valores de SNR.
    
    Args:
        nc (int): Número de subportadoras activas.
        bitsbn (numpy array): Bits originales de la imagen.
        ancho (int): Ancho de la imagen.
        alto (int): Alto de la imagen.
        pilotCarriers (numpy array): Subportadoras piloto.
        dataCarriers (numpy array): Subportadoras de datos.
        pilotValue (complex): Valor de las subportadoras piloto.
        n (int): Tamaño de la IFFT.
        modulation_types (list): Tipos de modulación ("qpsk", "16qam", "64qam").
        SNR_range (list): Valores de SNR a simular (en dB).
        channelResponse (numpy array): Respuesta del canal.
        CP (int): Longitud del prefijo cíclico.

    Returns:
        dict: BER para cada modulación y SNR.
    """
    ber_results = {mod: [] for mod in modulation_types}

    for modulation in modulation_types:
        print(f"\nSimulando para modulación: {modulation}")
        constellation_mapping = get_qam_constellation_mapping({"qpsk": 1, "16qam": 2, "64qam": 3}[modulation])
        demapping_table = {v: k for k, v in constellation_mapping.items()}

        # Precalcular símbolos y OFDM
        symbols = modulate_with_mapping(bitsbn, modulation)
        grupos_resultantes = Serie_paralelo(symbols, abs(nc - len(pilotCarriers)))

        for SNRdb in SNR_range:
            print(f"SNR: {SNRdb} dB")
            
            # Generar señal OFDM
            ofdm_canal = []
            ofdm_signal = []  # Para calcular PAPR

            for grupo in grupos_resultantes:
                OFDM_data = OFDM_symbol(grupo, pilotCarriers, dataCarriers, pilotValue, n)
                OFDM_time = IDFT(OFDM_data)
                OFDM_withCP = addCP(OFDM_time, CP)
                OFDM_TX = OFDM_withCP
                
                noise_prob = 0.1
                OFDM_RX = channel(OFDM_TX, SNRdb, noise_prob)
                ofdm_canal.append(OFDM_RX)  # Almacenar el resultado del canal
                
                ofdm_signal.append(OFDM_withCP)

            # Decodificar
            rx_bits_array_total = []
            symbols_esti = []
            
            for grupo_rx in ofdm_canal:
                OFDM_RX_noCP = removeCP(grupo_rx, CP, n)
                OFDM_demod = DFT(OFDM_RX_noCP)
                OFDM_demod_datos = OFDM_demod[:nc]
                Hest_at_pilots, Hest = channelEstimate(OFDM_demod_datos, pilotCarriers, pilotValue, allCarriers, H_exact)
                equalized_Hest = equalize(OFDM_demod_datos, Hest)
                QAM_est = get_payload(equalized_Hest, dataCarriers)
                PS_est, hardDecision = Demapping(QAM_est, demapping_table)
                # Bits recuperados
                rx_bits_array = PS_est.flatten()
                symbols_esti.append(rx_bits_array)

            # Agrupación de los bloques de bits
            rx_bits_array_total = np.concatenate(symbols_esti)
            
            # Comparar con bits originales
            Bits_recuperados = rx_bits_array_total[:len(bitsbn)]
            errors = np.sum(bitsbn != Bits_recuperados)
            ber = errors / len(bitsbn)
            ber_results[modulation].append(ber)

            print(f"SNR: {SNRdb} dB, BER: {ber}")

    return ber_results

def distribuir_pilotos_uniformemente(nc, porcentaje_pilotos):
    """
    Distribuye uniformemente las subportadoras piloto en el rango total de subportadoras.

    Args:
        nc (int): Número total de subportadoras (N_c).
        porcentaje_pilotos (float): Porcentaje de subportadoras que serán piloto (ejemplo: 5.0 para 5%).

    Returns:
        tuple: (pilotCarriers, dataCarriers)
            - pilotCarriers: Índices de las subportadoras piloto.
            - dataCarriers: Índices de las subportadoras de datos.
    """
    # Convertir porcentaje a proporción
    pilot_ratio = porcentaje_pilotos / 100

    # Calcular el número de pilotos
    num_pilots = int(np.floor(nc * pilot_ratio))  # Redondear hacia abajo para evitar exceder

    # Distribuir uniformemente los pilotos
    pilotCarriers = np.linspace(0, nc - 1, num=num_pilots, dtype=int)  # Índices de subportadoras piloto

    # Subportadoras de datos
    allCarriers = np.arange(nc)  # Total de subportadoras (N_c)
    dataCarriers = np.delete(allCarriers, pilotCarriers)  # Eliminar las subportadoras piloto

    return pilotCarriers, dataCarriers

def channel(ofdm_signal, snr_db, noise_prob=0.1):
    """
    Simula un canal con ruido aleatorio y no uniforme.

    Args:
        ofdm_signal (np.array): Señal OFDM en el dominio del tiempo.
        snr_db (float): Relación señal-ruido en decibelios (SNR).
        noise_prob (float): Probabilidad de que un símbolo se vea altamente afectado (0 a 1).

    Returns:
        np.array: Señal OFDM afectada por ruido no uniforme.
    """
    channelResponse = np.array([1, 0, 0.3 + 0.3j])  # Respuesta impulsiva
    
    convolved = np.convolve(ofdm_signal, channelResponse)
    
    # Calcular la potencia de la señal
    signal_power = np.mean(np.abs(convolved)**2)
    
    # Convertir SNR de dB a escala lineal
    snr_linear = 10**(snr_db / 10)
    
    # Calcular la potencia base del ruido
    noise_power = signal_power / snr_linear
    
    # Generar ruido gaussiano complejo base
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(ofdm_signal)) + 1j * np.random.randn(len(ofdm_signal)))
    
    # Generar un patrón aleatorio de símbolos muy ruidosos
    high_noise_indices = np.random.choice([0, 1], size=len(ofdm_signal), p=[1 - noise_prob, noise_prob])
    
    # Aumentar el nivel de ruido en las posiciones seleccionadas
    high_noise = high_noise_indices * (np.random.randn(len(ofdm_signal)) + 1j * np.random.randn(len(ofdm_signal))) * 10 * np.sqrt(noise_power)
    
    # Señal recibida: añadir ruido base + ruido alto aleatorio
    received_signal = ofdm_signal + noise + high_noise
    
    return received_signal

def convolutional_encoding(bits):
    """
    Aplica codificación convolucional a los bits de entrada.
    
    Args:
        bits (numpy array): Tren de bits a codificar.
    
    Returns:
        numpy array: Bits codificados.
    """
    # Definir el trellis para la codificación convolucional (ejemplo: tasa 1/2, K=3)
    trellis = Trellis(memory=np.array([2]), g_matrix=np.array([[0o7, 0o5]]))
    
    # Codificar los bits
    encoded_bits = conv_encode(bits, trellis)
    
    return encoded_bits

def convolutional_decoding(encoded_bits):
    """
    Aplica decodificación convolucional a los bits codificados.
    
    Args:
        encoded_bits (numpy array): Tren de bits codificados.
    
    Returns:
        numpy array: Bits decodificados.
    """
    # Definir el trellis para la decodificación convolucional (debe ser el mismo que en la codificación)
    trellis = Trellis(memory=np.array([2]), g_matrix=np.array([[0o7, 0o5]]))
    
    # Decodificar los bits usando el algoritmo de Viterbi
    decoded_bits = viterbi_decode(encoded_bits, trellis)
    
    return decoded_bits

def turbo_encoding(bits):
    """
    Simula la codificación Turbo utilizando dos codificadores convolucionales.
    
    Args:
        bits (numpy array): Tren de bits a codificar.
    
    Returns:
        numpy array: Bits codificados.
    """
    # Primer codificador convolucional
    trellis1 = Trellis(memory=np.array([2]), g_matrix=np.array([[0o7, 0o5]]))
    encoded_bits1 = conv_encode(bits, trellis1)
    
    # Interleaving (mezcla de bits)
    interleaved_bits = bits[np.random.permutation(len(bits))]
    
    # Segundo codificador convolucional
    trellis2 = Trellis(memory=np.array([2]), g_matrix=np.array([[0o7, 0o5]]))
    encoded_bits2 = conv_encode(interleaved_bits, trellis2)
    
    # Combinar los bits codificados
    turbo_encoded_bits = np.concatenate((encoded_bits1, encoded_bits2))
    
    return turbo_encoded_bits

def turbo_decoding(encoded_bits):
    """
    Simula la decodificación Turbo utilizando dos decodificadores convolucionales.
    
    Args:
        encoded_bits (numpy array): Tren de bits codificados.
    
    Returns:
        numpy array: Bits decodificados.
    """
    # Dividir los bits codificados en dos partes
    half_length = len(encoded_bits) // 2
    encoded_bits1 = encoded_bits[:half_length]
    encoded_bits2 = encoded_bits[half_length:]
    
    # Primer decodificador convolucional
    trellis1 = Trellis(memory=np.array([2]), g_matrix=np.array([[0o7, 0o5]]))
    decoded_bits1 = viterbi_decode(encoded_bits1, trellis1)
    
    # De-interleaving (desmezcla de bits)
    deinterleaved_bits = decoded_bits1[np.random.permutation(len(decoded_bits1))]
    
    # Segundo decodificador convolucional
    trellis2 = Trellis(memory=np.array([2]), g_matrix=np.array([[0o7, 0o5]]))
    decoded_bits2 = viterbi_decode(encoded_bits2, trellis2)
    
    # Combinar los bits decodificados
    turbo_decoded_bits = np.concatenate((decoded_bits1, decoded_bits2))
    
    return turbo_decoded_bits

def apply_dft(symbols):
    """
    Aplica la DFT a los símbolos modulados.
    
    Args:
        symbols (numpy array): Símbolos modulados en el dominio del tiempo.
    
    Returns:
        numpy array: Símbolos en el dominio de la frecuencia.
    """
    return np.fft.fft(symbols)

def apply_idft(symbols):
    """
    Aplica la IDFT a los símbolos recibidos.
    
    Args:
        symbols (numpy array): Símbolos en el dominio de la frecuencia.
    
    Returns:
        numpy array: Símbolos en el dominio del tiempo.
    """
    return np.fft.ifft(symbols)

def calculate_papr(signal):
    """
    Calcula el PAPR de una señal.
    
    Args:
        signal (numpy array): Señal en el dominio del tiempo.
    
    Returns:
        float: Valor del PAPR.
    """
    peak_power = np.max(np.abs(signal)**2)
    avg_power = np.mean(np.abs(signal)**2)
    return 10 * np.log10(peak_power / avg_power)

def plot_ccdf(papr_values, label):
    """
    Grafica la CCDF del PAPR.
    
    Args:
        papr_values (list): Lista de valores de PAPR.
        label (str): Etiqueta para la gráfica.
    """
    papr_values = np.array(papr_values)
    papr_values.sort()
    ccdf = 1 - np.arange(1, len(papr_values) + 1) / len(papr_values)
    plt.semilogy(papr_values, ccdf, label=label)
    plt.xlabel('PAPR (dB)')
    plt.ylabel('CCDF')
    plt.title('CCDF del PAPR')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

def maximum_ratio_combining(signals, snr_values):
    """
    Combina las señales recibidas en múltiples antenas usando Maximum-Ratio Combining (MRC).
    
    Args:
        signals (list): Lista de señales recibidas en cada antena.
        snr_values (list): Lista de valores de SNR para cada antena.
    
    Returns:
        numpy array: Señal combinada usando MRC.
    """
    # Calcular los pesos de MRC basados en el SNR
    weights = np.sqrt(snr_values)
    weights /= np.sum(weights)  # Normalizar los pesos
    
    # Combinar las señales usando los pesos
    combined_signal = np.zeros_like(signals[0])
    for signal, weight in zip(signals, weights):
        combined_signal += signal * weight
    
    return combined_signal

def process_signals_with_diversity(ofdm_canal, snr_values, num_antennas):
    """
    Procesa las señales recibidas en múltiples antenas usando diversidad.
    
    Args:
        ofdm_canal (list): Lista de señales OFDM recibidas en cada antena.
        snr_values (list): Lista de valores de SNR para cada antena.
        num_antennas (int): Número de antenas a utilizar (2, 4, 8).
    
    Returns:
        numpy array: Señal combinada usando MRC.
    """
    # Seleccionar las señales de las antenas correspondientes
    signals = ofdm_canal[:num_antennas]
    snr = snr_values[:num_antennas]
    
    # Aplicar MRC
    combined_signal = maximum_ratio_combining(signals, snr)
    
    return combined_signal

def reconstruct_image_with_diversity(bits_recuperados, ancho, alto):
    """
    Reconstruye la imagen a partir de los bits recuperados usando diversidad.
    
    Args:
        bits_recuperados (numpy array): Bits recuperados después de la combinación MRC.
        ancho (int): Ancho de la imagen.
        alto (int): Alto de la imagen.
    
    Returns:
        numpy array: Imagen reconstruida.
    """
    reconstructed_image, _ = reconstruct_image_from_bits(bits_recuperados, ancho, alto)
    return reconstructed_image

def plot_ber_vs_snr(ber_results, snr_range):
    """
    Grafica el BER vs SNR para cada antena en una misma gráfica.
    
    Args:
        ber_results (dict): Diccionario con los resultados de BER para cada antena.
        snr_range (list): Rango de valores de SNR.
    """
    plt.figure(figsize=(10, 6))
    for antenna, ber_values in ber_results.items():
        plt.semilogy(snr_range, ber_values, label=f'Antena {antenna}')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('BER vs SNR para Cada Antena')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()

def lte_convolutional_encoder(bits):
    """
    Codificador convolucional LTE (tasa 1/3, polinomios [13, 15, 17] en octal).
    
    Args:
        bits (numpy array): Bits de entrada.
    
    Returns:
        numpy array: Bits codificados.
    """
    # Definir el trellis para el codificador convolucional LTE
    trellis = Trellis(memory=np.array([3]), g_matrix=np.array([[0o13, 0o15, 0o17]]))
    
    # Codificar los bits
    encoded_bits = conv_encode(bits, trellis)
    
    return encoded_bits

def lte_turbo_encoder(bits):
    """
    Codificador Turbo LTE.
    
    Args:
        bits (numpy array): Bits de entrada.
    
    Returns:
        numpy array: Bits codificados.
    """
    # Primer codificador convolucional (mismo que el LTE)
    trellis = Trellis(memory=np.array([3]), g_matrix=np.array([[0o13, 0o15, 0o17]]))
    encoded_bits1 = conv_encode(bits, trellis)
    
    # Entrelazado (interleaving)
    interleaved_bits = bits[np.random.permutation(len(bits))]
    
    # Segundo codificador convolucional
    encoded_bits2 = conv_encode(interleaved_bits, trellis)
    
    # Combinar los bits codificados (sistemáticos, paridad 1, paridad 2)
    turbo_encoded_bits = np.concatenate((bits, encoded_bits1[1::2], encoded_bits2[1::2]))
    
    return turbo_encoded_bits

def lte_viterbi_decoder(encoded_bits):
    """
    Decodificador Viterbi para la codificación convolucional LTE.
    
    Args:
        encoded_bits (numpy array): Bits codificados.
    
    Returns:
        numpy array: Bits decodificados.
    """
    # Definir el trellis para el decodificador
    trellis = Trellis(memory=np.array([3]), g_matrix=np.array([[0o13, 0o15, 0o17]]))
    
    # Decodificar los bits usando Viterbi
    decoded_bits = viterbi_decode(encoded_bits, trellis)
    
    return decoded_bits

def lte_turbo_decoder(encoded_bits):
    """
    Decodificador Turbo LTE.
    
    Args:
        encoded_bits (numpy array): Bits codificados.
    
    Returns:
        numpy array: Bits decodificados.
    """
    # Dividir los bits codificados en sistemáticos, paridad 1 y paridad 2
    systematic_bits = encoded_bits[::3]
    parity1_bits = encoded_bits[1::3]
    parity2_bits = encoded_bits[2::3]
    
    # Definir el trellis para el decodificador
    trellis = Trellis(memory=np.array([3]), g_matrix=np.array([[0o13, 0o15, 0o17]]))
    
    # Decodificar usando Viterbi (primer codificador)
    decoded_bits1 = viterbi_decode(np.concatenate((systematic_bits, parity1_bits)), trellis)
    
    # Entrelazado (interleaving)
    interleaved_bits = systematic_bits[np.random.permutation(len(systematic_bits))]
    
    # Decodificar usando Viterbi (segundo codificador)
    decoded_bits2 = viterbi_decode(np.concatenate((interleaved_bits, parity2_bits)), trellis)
    
    # Combinar los resultados (decisión dura)
    turbo_decoded_bits = np.logical_and(decoded_bits1, decoded_bits2).astype(int)
    
    return turbo_decoded_bits

def main():
    # Ruta de la imagen
    image_path = "pluto.jpg"
    
    # Cargar la imagen
    image = load_image(image_path)
    if image is None:
        return
    
    # Convertir a blanco y negro
    image_bw = convert_to_grayscale(image)
    
    # Obtener bits de la imagen en blanco y negro
    bitsbn, ancho, alto = image_to_bits_bw(image_bw)
    print(f"Dimensiones de imagen B/N: Ancho: {ancho}, Alto: {alto}")

    # Aplicar codificación convolucional LTE
    encoded_bits = lte_convolutional_encoder(bitsbn)
    
    # Aplicar codificación Turbo LTE (opcional)
    turbo_encoded_bits = lte_turbo_encoder(encoded_bits)
    
    # Solicitar tipo de modulación qpsk, 16qam, 64qam
    modulation_type = select_modulation()  # se obtiene "qpsk", "16qam" o "64qam"
    
    # Realiza Modulación QPSK, 16-QAM o 64-QAM
    symbols = modulate_with_mapping(turbo_encoded_bits, modulation_type)
    
    # Aplicar MIMO 8x2
    num_tx_antennas = 8
    num_rx_antennas = 2
    mimo_symbols = apply_mimo(symbols, num_tx_antennas, num_rx_antennas)
    
    # Procesar símbolos para cada antena
    ofdm_canal = []
    for i in range(num_tx_antennas):
        OFDM_data = OFDM_symbol(mimo_symbols[i], pilotCarriers, dataCarriers, pilotValue, n)
        OFDM_time = IDFT(OFDM_data)
        OFDM_withCP = addCP(OFDM_time, CP)
        OFDM_TX = OFDM_withCP
        OFDM_RX = channel(OFDM_TX, SNRdb, noise_prob)
        ofdm_canal.append(OFDM_RX)
    
    # Combinar las señales recibidas desde ambas antenas
    combined_signals = []
    for rx_ant1, rx_ant2 in zip(ofdm_canal[:num_rx_antennas], ofdm_canal[num_rx_antennas:]):
        combined_signal = maximum_ratio_combining([rx_ant1, rx_ant2], [SNRdb, SNRdb])
        combined_signals.append(combined_signal)
    
    # DECODIFICACIÓN OFDM con señales combinadas
    symbols_esti = []
    all_QAM_est = []
    all_hardDecision = []
    
    for grupo_rx in combined_signals:
        OFDM_RX_noCP = removeCP(grupo_rx, CP, n)
        OFDM_demod = DFT(OFDM_RX_noCP)
        OFDM_demod_datos = OFDM_demod[:nc]
        Hest_at_pilots, Hest = channelEstimate(OFDM_demod_datos, pilotCarriers, pilotValue, allCarriers, H_exact)
        equalized_Hest = equalize(OFDM_demod_datos, Hest)
        QAM_est = get_payload(equalized_Hest, dataCarriers)
        PS_est, hardDecision = Demapping(QAM_est, demapping_table)
        rx_bits_array = PS_est.flatten()
        symbols_esti.append(rx_bits_array)
    
    # Agrupación de los bloques de bits
    rx_bits_array_total = np.concatenate(symbols_esti)
    
    # Aplicar decodificación Turbo LTE (opcional)
    turbo_decoded_bits = lte_turbo_decoder(rx_bits_array_total)
    
    # Aplicar decodificación convolucional LTE
    decoded_bits = lte_viterbi_decoder(turbo_decoded_bits)
    
    Bits_recuperados = decoded_bits[:len(bitsbn)]
    
    # Validar resultados
    print(f"Número total de bits recuperados: {len(rx_bits_array_total)}")
    print(f"Número de bits útiles recuperados: {len(Bits_recuperados)}")

    reconstructed_image, bitsrec = reconstruct_image_from_bits(Bits_recuperados, ancho, alto)
        
    # Cálculo del BER
    errors = np.sum(bitsbn != bitsrec)
    total_bits = len(bitsbn)
    ber = errors / total_bits
    print(f"bits erróneos: {errors}")
    print(f"BER: {ber}")

    # Configuración de la simulación
    modulation_types = ["qpsk", "16qam", "64qam"]
    SNR_range = range(0, 60, 5)  # De 0 a 30 dB

    # Simulación Monte Carlo
    ber_results = monte_carlo_simulation(nc, bitsbn, H_exact, allCarriers, pilotCarriers, dataCarriers, pilotValue, n, modulation_types, SNR_range, channelResponse, CP)

    # Crear una figura para contener todo
    fig = plt.figure(figsize=(15, 6))  # Ajusta el tamaño de la ventana

    # Subplot 1: Imagen original
    ax1 = fig.add_subplot(2, 3, 1)  # 1 fila, 3 columnas, posición 1
    ax1.imshow(image)
    ax1.axis('off')  # Ocultar ejes
    ax1.set_title("Imagen Original")

    # Subplot 2: Imagen en blanco y negro
    ax2 = fig.add_subplot(2, 3, 2)  # 1 fila, 3 columnas, posición 2
    ax2.imshow(image_bw, cmap="gray")
    ax2.axis('off')  # Ocultar ejes
    ax2.set_title("Imagen en Blanco y Negro")

    # Subplot 3: Constelación
    ax3 = fig.add_subplot(2, 3, 3)  # 1 fila, 3 columnas, posición 3
    ax3.scatter(symbols.real, symbols.imag, color='blue', alpha=0.7, s=5)  # Reducir tamaño de puntos
    ax3.axhline(0, color='black', linewidth=0.5, linestyle="--")
    ax3.axvline(0, color='black', linewidth=0.5, linestyle="--")
    ax3.grid(True, linestyle="--", alpha=0.7)
    ax3.set_title(f"Constelación Modulada ({modulation_type.upper()})")
    ax3.set_xlabel("Eje Real")
    ax3.set_ylabel("Eje Imaginario")
    ax3.set_aspect("equal", adjustable="datalim")  # Mantener proporciones
    
    # Subplot 4: Imagen en blanco y negro reconstruida
    ax4 = fig.add_subplot(2, 3, 4)  # 2 fila, 3 columnas, posición 1
    ax4.imshow(reconstructed_image, cmap="gray")
    ax4.axis('off')  # Ocultar ejes
    ax4.set_title("Imagen en Blanco y Negro Reconstruida")
    
    # Subplot 5: Constelación con decisiones duras
    ax5 = fig.add_subplot(2, 3, 5)  # 2 filas, 3 columnas, posición 5
    # Graficar la constelación de todos los símbolos QAM recibidos en ax5
    ax5.scatter(all_QAM_est.real, all_QAM_est.imag, color='blue', s=5, alpha=0.5)
    ax5.plot(all_hardDecision.real, all_hardDecision.imag, 'ro')  # Puntos rojos en las decisiones duras
    ax5.axhline(0, color='black', linewidth=0.5, linestyle="--")
    ax5.axvline(0, color='black', linewidth=0.5, linestyle="--")
    ax5.set_title("Constelación de Todos los Símbolos QAM Recibidos")
    ax5.set_xlabel("Eje Real")
    ax5.set_ylabel("Eje Imaginario")
    ax5.grid(True, linestyle="--", alpha=0.7)

    # Subplot 6: BER vs SNR
    ax6 = fig.add_subplot(2, 3, 6)  # 2 filas, 3 columnas, posición 6
    for modulation, ber_values in ber_results.items():
        ax6.semilogy(SNR_range, ber_values, label=modulation.upper())
    ax6.set_xlabel("SNR (dB)")
    ax6.set_ylabel("BER")
    ax6.set_title("BER vs SNR para Modulaciones")
    ax6.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax6.legend()
    
    # Ajustar el diseño para evitar superposiciones
    plt.tight_layout()

    # Mostrar la figura
    plt.show()

if __name__ == "__main__":
    main()
