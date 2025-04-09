import sounddevice as sd
import numpy as np
import time

# --- Configuración ---
SAMPLERATE = 16000  # Tasa de muestreo (Hz)
CHANNELS = 1        # Mono
DTYPE = 'int16'     # Tipo de dato
CHUNK_SIZE = 1024   # Tamaño del bloque de audio a analizar (muestras)

def mostrar_sensibilidad(samplerate, channels, chunk_size):
    """Muestra el nivel de sensibilidad del micrófono en tiempo real."""
    print("-" * 20)
    print("Iniciando monitoreo de sensibilidad...")
    print("Habla ahora para ver el nivel de RMS.")
    print("(Presiona Ctrl+C para detener el monitoreo)")
    print("-" * 20)

    try:
        # Usamos InputStream para leer en bloques
        with sd.InputStream(samplerate=samplerate,
                                channels=channels,
                                dtype=DTYPE,
                                blocksize=chunk_size) as stream:
            print("¡Monitoreando!")
            while True:
                # Leer un chunk de audio
                chunk, overflowed = stream.read(chunk_size)
                if overflowed:
                    print("¡Advertencia! Se detectó un overflow (posible pérdida de datos).")

                # Calcular el volumen (RMS) del chunk actual
                # Convertir a float32 para el cálculo para evitar problemas con int16
                rms = np.sqrt(np.mean(np.square(chunk.astype(np.float32))))

                # Imprimir el valor de RMS
                print(f"RMS: {rms:.4f}", end='\r')
                time.sleep(0.1) # Mostrar cada 0.1 segundos

    except KeyboardInterrupt:
        print("\nMonitoreo interrumpido manualmente.")
    except Exception as e:
        print(f"\nERROR durante el monitoreo: {e}")

if __name__ == "__main__":
    mostrar_sensibilidad(SAMPLERATE, CHANNELS, CHUNK_SIZE)
    print("\nProceso finalizado.")