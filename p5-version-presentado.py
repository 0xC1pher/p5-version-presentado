import matplotlib
matplotlib.use('TkAgg')  # Establece un backend que permite mostrar ventanas
import matplotlib.pyplot as plt
from PIL import Image
import math
import numpy as np
from scipy.interpolate import interp1d

# --- Funciones Auxiliares ---

def cargar_imagen(ruta_imagen):
    """
    Carga una imagen desde una ruta específica.

    Args:
        ruta_imagen (str): Ruta de la imagen.

    Returns:
        PIL.Image: Imagen cargada o None si no se encuentra.
    """
    try:
        imagen = Image.open(ruta_imagen)
        return imagen
    except FileNotFoundError:
        print(f"Error: No se encontró la imagen en {ruta_imagen}")
        return None

def convertir_a_escala_de_grises(imagen):
    """
    Convierte una imagen a escala de grises.

    Args:
        imagen (PIL.Image): Imagen a convertir.

    Returns:
        PIL.Image: Imagen en escala de grises.
    """
    return imagen.convert('L')

def imagen_a_bits(imagen):
    """
    Convierte una imagen en escala de grises a un tren de bits.

    Args:
        imagen (PIL.Image): Imagen en escala de grises.

    Returns:
        tuple: (bits, ancho, alto), donde bits es un array de bits,
               ancho y alto son las dimensiones de la imagen.
    """
    array = np.array(imagen)
    alto, ancho = array.shape
    bits = np.unpackbits(array.flatten())
    return bits, ancho, alto

def seleccionar_modulacion():
    """
    Permite al usuario seleccionar el tipo de modulación.

    Returns:
        int: Tipo de modulación (1: QPSK, 2: 16-QAM, 3: 64-QAM).
    """
    while True:
        try:
            eleccion = int(input("Seleccione la modulación (1: QPSK, 2: 16-QAM, 3: 64-QAM): "))
            if 1 <= eleccion <= 3:
                return eleccion
            else:
                print("Elección inválida.")
        except ValueError:
            print("Entrada inválida. Por favor, ingrese un número.")

def obtener_constelacion_qam(tipo_modulacion):
    """
    Devuelve el mapeo de constelación para el tipo de modulación seleccionado.

    Args:
        tipo_modulacion (int): Tipo de modulación (1: QPSK, 2: 16-QAM, 3: 64-QAM).

    Returns:
        dict: Mapeo de constelación o mensaje de error si no es soportado.
    """
    constelaciones = {
        1: {  # QPSK
            0: -1 - 1j, 1: -1 + 1j, 2: 1 - 1j, 3: 1 + 1j
        },
        2: {  # 16-QAM
            0: -3 - 3j, 1: -3 - 1j, 2: -3 + 1j, 3: -3 + 3j,
            4: -1 - 3j, 5: -1 - 1j, 6: -1 + 1j, 7: -1 + 3j,
            8: 1 - 3j, 9: 1 - 1j, 10: 1 + 1j, 11: 1 + 3j,
            12: 3 - 3j, 13: 3 - 1j, 14: 3 + 1j, 15: 3 + 3j
        },
        3: {  # 64-QAM
            # Implementación de 64-QAM (completar según sea necesario)
            # ...
        }
    }
    try:
        return constelaciones[tipo_modulacion]
    except KeyError:
        return "Error: Tipo de modulación no soportado."

def modular_con_mapeo(bits, tipo_modulacion):
    """
    Modula un tren de bits utilizando un mapeo de constelación.

    Args:
        bits (numpy array): Tren de bits.
        tipo_modulacion (int): Tipo de modulación (1: QPSK, 2: 16-QAM, 3: 64-QAM).

    Returns:
        numpy array: Símbolos modulados o mensaje de error.
    """
    if not bits.size:
        raise ValueError("El array de bits está vacío.")
    mapeo = obtener_constelacion_qam(tipo_modulacion)
    if isinstance(mapeo, str):
        return mapeo  # Devuelve mensaje de error si el tipo de modulación es inválido

    bits_por_simbolo = int(np.log2(len(mapeo)))
    if len(bits) % bits_por_simbolo != 0:
        padding = bits_por_simbolo - (len(bits) % bits_por_simbolo)
        bits = np.pad(bits, (0, padding), 'constant')

    indices = np.packbits(bits.reshape(-1, bits_por_simbolo), axis=1)
    simbolos = np.array([mapeo[i] for i in indices])
    return simbolos

def obtener_ancho_de_banda_y_espaciado():
    """
    Solicita al usuario el ancho de banda y el espaciado entre subportadoras.

    Returns:
        tuple: (ancho_de_banda, espaciado) en Hz.
    """
    while True:
        try:
            ancho_de_banda = float(input("Ingrese el ancho de banda (Hz): "))
            espaciado = float(input("Ingrese el espaciado entre subportadoras (Hz): "))
            if ancho_de_banda > 0 and espaciado > 0:
                return ancho_de_banda, espaciado
            else:
                print("El ancho de banda y el espaciado deben ser positivos.")
        except ValueError:
            print("Entrada inválida. Por favor, ingrese números.")

def obtener_longitud_cp():
    """
    Solicita al usuario la longitud del prefijo cíclico.

    Returns:
        int: Longitud del prefijo cíclico (1: Normal, 2: Extendido).
    """
    while True:
        try:
            eleccion = int(input("Prefijo cíclico (1: Normal, 2: Extendido): "))
            if eleccion in (1, 2):
                return eleccion
            else:
                print("Elección inválida.")
        except ValueError:
            print("Entrada inválida. Por favor, ingrese 1 o 2.")

def obtener_snr():
    """
    Solicita al usuario la relación señal-ruido (SNR).

    Returns:
        float: SNR en dB.
    """
    while True:
        try:
            snr_db = float(input("Ingrese el SNR (dB): "))
            return snr_db
        except ValueError:
            print("Entrada inválida. Por favor, ingrese un número.")

def calcular_nc(ancho_de_banda, espaciado):
    """
    Calcula el número de subportadoras activas.

    Args:
        ancho_de_banda (float): Ancho de banda en Hz.
        espaciado (float): Espaciado entre subportadoras en Hz.

    Returns:
        int: Número de subportadoras activas.
    """
    return int(ancho_de_banda // espaciado)

def calcular_tamano_ifft(nc):
    """
    Calcula el tamaño de la IFFT como la potencia de 2 más cercana mayor o igual a nc.

    Args:
        nc (int): Número de subportadoras activas.

    Returns:
        int: Tamaño de la IFFT.
    """
    n = 1
    while n < nc:
        n *= 2
    return n

def modular_ofdm(datos, portadoras_piloto, portadoras_datos, valor_piloto, n):
    """
    Modula los datos en un símbolo OFDM.

    Args:
        datos (numpy array): Datos a modular.
        portadoras_piloto (numpy array): Índices de las portadoras piloto.
        portadoras_datos (numpy array): Índices de las portadoras de datos.
        valor_piloto (complex): Valor de las portadoras piloto.
        n (int): Tamaño de la IFFT.

    Returns:
        numpy array: Símbolo OFDM.
    """
    simbolo_ofdm = np.zeros(n, dtype=complex)
    simbolo_ofdm[portadoras_piloto] = valor_piloto
    simbolo_ofdm[portadoras_datos] = datos
    return simbolo_ofdm

def demodular_ofdm(senal_recibida, portadoras_piloto, valor_piloto, portadoras_datos, n):
    """
    Demodula un símbolo OFDM.

    Args:
        senal_recibida (numpy array): Señal recibida.
        portadoras_piloto (numpy array): Índices de las portadoras piloto.
        valor_piloto (complex): Valor de las portadoras piloto.
        portadoras_datos (numpy array): Índices de las portadoras de datos.
        n (int): Tamaño de la IFFT.

    Returns:
        numpy array: Señal demodulada.
    """
    simbolo_ofdm = np.fft.fft(senal_recibida)
    estimaciones_piloto = simbolo_ofdm[portadoras_piloto] / valor_piloto
    estimacion_canal = np.interp(np.arange(n), portadoras_piloto, estimaciones_piloto)
    senal_ecualizada = simbolo_ofdm / estimacion_canal
    return senal_ecualizada[portadoras_datos]

def canal(senal, snr_db):
    """
    Simula un canal con ruido AWGN.

    Args:
        senal (numpy array): Señal a transmitir.
        snr_db (float): Relación señal-ruido en dB.

    Returns:
        numpy array: Señal con ruido.
    """
    if snr_db is None or snr_db < 0:
        raise ValueError("El SNR debe ser un número no negativo.")
    snr = 10**(snr_db / 10)
    potencia_senal = np.mean(np.abs(senal)**2)
    potencia_ruido = potencia_senal / snr
    ruido = np.sqrt(potencia_ruido / 2) * (np.random.randn(*senal.shape) + 1j * np.random.randn(*senal.shape))
    return senal + ruido

def demapear_simbolos(simbolos_recibidos, tipo_modulacion):
    """
    Convierte símbolos recibidos a bits.

    Args:
        simbolos_recibidos (numpy array): Símbolos recibidos.
        tipo_modulacion (int): Tipo de modulación (1: QPSK, 2: 16-QAM, 3: 64-QAM).

    Returns:
        numpy array: Bits demodulados o mensaje de error.
    """
    mapeo = obtener_constelacion_qam(tipo_modulacion)
    if isinstance(mapeo, str):
        return mapeo  # Devuelve mensaje de error si el tipo de modulación es inválido

    distancias = np.abs(simbolos_recibidos[:, np.newaxis] - np.array(list(mapeo.values())))**2
    indices_decodificados = np.argmin(distancias, axis=1)
    bits_por_simbolo = int(np.log2(len(mapeo)))
    bits_decodificados = np.unpackbits(indices_decodificados.astype(np.uint8).reshape(-1,1), axis=1)[:,:bits_por_simbolo].flatten()
    return bits_decodificados

def reconstruir_imagen_desde_bits(bits, ancho, alto):
    """
    Reconstruye una imagen a partir de un tren de bits.

    Args:
        bits (numpy array): Tren de bits.
        ancho (int): Ancho de la imagen.
        alto (int): Alto de la imagen.

    Returns:
        numpy array: Imagen reconstruida.
    """
    total_bits = ancho * alto * 8
    if len(bits) < total_bits:
        print(f"Advertencia: Menos bits de los esperados. Rellenando con ceros.")
        bits = np.pad(bits, (0, total_bits - len(bits)), 'constant')
    elif len(bits) > total_bits:
        bits = bits[:total_bits]
    imagen_array = np.packbits(bits).reshape(alto, ancho)
    return imagen_array

def distribuir_pilotos(n, porcentaje_pilotos):
    """
    Distribuye uniformemente las portadoras piloto.

    Args:
        n (int): Número total de subportadoras.
        porcentaje_pilotos (float): Porcentaje de portadoras piloto.

    Returns:
        tuple: (portadoras_piloto, portadoras_datos).
    """
    num_pilotos = int(n * porcentaje_pilotos / 100)
    portadoras_piloto = np.linspace(0, n - 1, num_pilotos, dtype=int)
    portadoras_datos = np.setdiff1d(np.arange(n), portadoras_piloto)
    return portadoras_piloto, portadoras_datos

# --- Función Principal ---

def main():
    ruta_imagen = input("Ingrese la ruta de la imagen: ")
    imagen = cargar_imagen(ruta_imagen)
    if imagen is None:
        return

    imagen_bn = convertir_a_escala_de_grises(imagen)
    bits, ancho, alto = imagen_a_bits(imagen_bn)

    tipo_modulacion = seleccionar_modulacion()
    snr_db = obtener_snr()
    eleccion_cp = obtener_longitud_cp()

    ancho_de_banda, espaciado = obtener_ancho_de_banda_y_espaciado()
    nc = calcular_nc(ancho_de_banda, espaciado)
    n = calcular_tamano_ifft(nc)

    porcentaje_pilotos = float(input("Ingrese el porcentaje de portadoras piloto (0-100): "))
    portadoras_piloto, portadoras_datos = distribuir_pilotos(n, porcentaje_pilotos)
    valor_piloto = -1 - 1j

    simbolos_modulados = modular_con_mapeo(bits, tipo_modulacion)
    if isinstance(simbolos_modulados, str):
        print(simbolos_modulados)
        return

    # Modulación OFDM
    senal_ofdm = modular_ofdm(simbolos_modulados, portadoras_piloto, portadoras_datos, valor_piloto, n)
    longitud_cp = int(n * 0.25) if eleccion_cp == 1 else int(n * 0.5)  # Ejemplo de longitudes de CP
    cp = senal_ofdm[-longitud_cp:]
    senal_transmitida = np.concatenate((cp, senal_ofdm))

    # Simulación del canal
    senal_recibida = canal(senal_transmitida, snr_db)
    senal_sin_cp = senal_recibida[longitud_cp:]  # Eliminar el prefijo cíclico
    simbolos_demodulados = demodular_ofdm(senal_sin_cp, portadoras_piloto, valor_piloto, portadoras_datos, n)
    bits_decodificados = demapear_simbolos(simbolos_demodulados, tipo_modulacion)
    if isinstance(bits_decodificados, str):
        print(bits_decodificados)
        return

    # Reconstrucción de la imagen
    imagen_reconstruida = reconstruir_imagen_desde_bits(bits_decodificados, ancho, alto)

    # Visualización de resultados
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(imagen, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(imagen_bn, cmap='gray')
    plt.title('Imagen en Escala de Grises')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(imagen_reconstruida, cmap='gray')
    plt.title('Imagen Reconstruida')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
