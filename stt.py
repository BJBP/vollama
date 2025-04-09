import sounddevice as sd
import numpy as np
import wave
import whisper
import torch
import time
import os

# --- Configuración de Grabación ---
SAMPLERATE = 16000  # Tasa de muestreo (Hz). 16000 es bueno para voz y compatible con Whisper.
CHANNELS = 1        # Mono
FILENAME_TEMP = "mi_grabacion_temporal.wav" # Nombre del archivo temporal para guardar la grabación
DTYPE = 'int16'     # Tipo de dato para la grabación (común para WAV)

# --- Función para Grabar Audio ---
def grabar_audio(filename, duration, samplerate, channels):
    """Graba audio del micrófono por una duración dada y lo guarda en un archivo WAV."""
    print("-" * 20)
    print(f"Prepárate para grabar durante {duration} segundos...")
    print("3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("¡GRABANDO!")

    try:
        # Grabar audio usando sounddevice
        recording_data = sd.rec(int(duration * samplerate),
                                samplerate=samplerate,
                                channels=channels,
                                dtype=DTYPE,
                                blocking=True) # blocking=True espera hasta que termine

        # sd.wait() # Alternativa a blocking=True si no se usa
        print("Grabación finalizada.")

        # Guardar la grabación en un archivo WAV
        print(f"Guardando grabación en '{filename}'...")
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            # sounddevice usa numpy, necesitamos el tamaño del item para setsampwidth
            # Para int16, el tamaño es 2 bytes
            wf.setsampwidth(np.dtype(DTYPE).itemsize)
            wf.setframerate(samplerate)
            wf.writeframes(recording_data.tobytes())
        print("Grabación guardada exitosamente.")
        print("-" * 20)
        return True
    except Exception as e:
        print(f"\nERROR al grabar o guardar el audio: {e}")
        print("Posibles causas:")
        print("  - ¿Tienes un micrófono conectado y configurado como entrada por defecto?")
        print("  - ¿Permitiste el acceso al micrófono para tu terminal o IDE?")
        print("  - Problemas con las librerías de audio del sistema (PortAudio).")
        print("-" * 20)
        return False

# --- Carga del Modelo Whisper (Solo CPU) ---
def cargar_modelo_whisper(model_name="base"):
    """Carga el modelo Whisper especificado, forzando el uso de CPU."""
    print(f"Cargando modelo Whisper '{model_name}' (forzado a CPU)...")
    if not torch.cuda.is_available():
        print("Confirmado: PyTorch está configurado solo para CPU.")
    else:
        # Aunque tengamos CUDA, forzaremos CPU
        print("CUDA detectado, pero forzando uso de CPU.")

    try:
        model = whisper.load_model(model_name, device='cpu')
        print(f"Modelo '{model_name}' cargado exitosamente en CPU.")
        return model
    except Exception as e:
        print(f"ERROR al cargar el modelo Whisper '{model_name}': {e}")
        print("Asegúrate de tener conexión a internet si es la primera vez que cargas este modelo.")
        return None

# --- Programa Principal ---
if __name__ == "__main__":
    # 1. Preguntar duración de la grabación
    while True:
        try:
            duracion_segundos = int(input("¿Cuántos segundos quieres grabar? (ej. 5): "))
            if duracion_segundos > 0:
                break
            else:
                print("Por favor, introduce un número positivo de segundos.")
        except ValueError:
            print("Entrada inválida. Por favor, introduce un número entero.")

    # 2. Grabar el audio
    grabacion_exitosa = grabar_audio(FILENAME_TEMP, duracion_segundos, SAMPLERATE, CHANNELS)

    # 3. Cargar modelo Whisper (si la grabación fue exitosa)
    modelo_whisper = None
    if grabacion_exitosa:
        # Puedes elegir el modelo aquí: 'tiny', 'base', 'small', 'medium', 'large'
        # Recuerda que modelos más grandes son más precisos pero más lentos en CPU.
        modelo_whisper = cargar_modelo_whisper(model_name="base") # 'base' es un buen compromiso

    # 4. Transcribir (si la grabación y carga del modelo fueron exitosas)
    if grabacion_exitosa and modelo_whisper:
        print("\nIniciando transcripción del audio grabado...")
        start_time = time.time()
        try:
            # Transcribir usando CPU (fp16=False es necesario para CPU)
            result = modelo_whisper.transcribe(FILENAME_TEMP, fp16=False)
            end_time = time.time()

            print(f"\n--- TRANSCRIPCIÓN COMPLETA ({end_time - start_time:.2f} segundos) ---")
            print(result["text"])
            print("-" * 50)

        except Exception as e:
            print(f"ERROR durante la transcripción: {e}")
    elif not grabacion_exitosa:
        print("\nNo se puede transcribir porque la grabación falló.")
    else: # grabacion_exitosa fue True, pero modelo_whisper es None
        print("\nNo se puede transcribir porque el modelo Whisper no se pudo cargar.")

    # 5. Limpieza (Opcional: borrar el archivo temporal)
    if os.path.exists(FILENAME_TEMP):
        try:
            os.remove(FILENAME_TEMP)
            print(f"Archivo temporal '{FILENAME_TEMP}' eliminado.")
        except Exception as e:
            print(f"Advertencia: No se pudo eliminar el archivo temporal '{FILENAME_TEMP}'. Error: {e}")

    print("\nProceso finalizado.")