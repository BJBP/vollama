import sounddevice as sd
import numpy as np
import wave
import whisper
import torch
import time
import os
import math # Para calcular RMS

# --- Configuración ---
SAMPLERATE = 16000  # Tasa de muestreo (Hz)
CHANNELS = 1        # Mono
FILENAME_TEMP = "mi_grabacion_automatica.wav" # Nombre del archivo
DTYPE = 'int16'     # Tipo de dato
CHUNK_SIZE = 1024   # Tamaño del bloque de audio a analizar (muestras)

# --- Parámetros de Detección de Silencio (¡AJUSTA ESTOS VALORES!) ---
# Umbral de volumen para considerar silencio. Valores más bajos = más sensible al ruido.
# Empieza con algo como 0.01 o 0.005 y ajusta según tus pruebas.
# Puedes imprimir el valor RMS en la función para ayudarte a calibrar.
SILENCE_THRESHOLD = 0.008  # -> ¡EXPERIMENTA CON ESTE VALOR! <-

# Duración en segundos de silencio continuo para detener la grabación.
SILENCE_DURATION = 2.5  # -> Puedes ajustar esto (ej. 2, 3)

# --- Variables Globales para Grabación ---
grabacion_completa = [] # Lista para almacenar los chunks de audio
grabando = False
silencio_iniciado = None
speech_detectada = False # Para evitar parar si solo hay silencio al principio

# --- Función de Callback (Alternativa más compleja) o Bucle de Lectura ---
# Usaremos un bucle de lectura, que es más fácil de seguir para este caso.

# --- Función para Grabar con Detección de Silencio ---
def grabar_con_silencio(filename, samplerate, channels, chunk_size, silence_threshold, silence_duration):
    """Graba audio hasta detectar un periodo de silencio."""
    global grabacion_completa, grabando, silencio_iniciado, speech_detectada
    grabacion_completa = [] # Reiniciar por si acaso
    grabando = True
    silencio_iniciado = None
    speech_detectada = False # Asegurarse de que empezamos sin detectar habla

    # Calculamos cuántos chunks de silencio necesitamos
    chunks_de_silencio_necesarios = int((silence_duration * samplerate) / chunk_size)
    chunks_silenciosos_actuales = 0

    print("-" * 20)
    print("Iniciando grabación...")
    print("Habla ahora. La grabación se detendrá automáticamente tras")
    print(f"{silence_duration} segundos de silencio.")
    print("(Presiona Ctrl+C en la terminal si quieres forzar la detención)")
    print("-" * 20)

    stream = None # Inicializar stream a None
    try:
        # Usamos InputStream para leer en bloques
        stream = sd.InputStream(samplerate=samplerate,
                                channels=channels,
                                dtype=DTYPE,
                                blocksize=chunk_size) # Usamos blocksize en lugar de chunk
        stream.start()
        print("¡Grabando!")

        while grabando:
            # Leer un chunk de audio
            # .read devuelve (datos_del_chunk, indicador_de_overflow)
            chunk, overflowed = stream.read(chunk_size)
            if overflowed:
                print("¡Advertencia! Se detectó un overflow (posible pérdida de datos).")

            # Añadir siempre el chunk actual a la grabación
            grabacion_completa.append(chunk)

            # Calcular el volumen (RMS) del chunk actual
            # Convertir a float32 para el cálculo para evitar problemas con int16
            rms = np.sqrt(np.mean(np.square(chunk.astype(np.float32))))

            # --- Lógica de Detección de Silencio ---
            # Descomenta la siguiente línea si quieres calibrar el threshold
            # print(f"RMS: {rms:.4f}")

            if rms > silence_threshold:
                # Se detectó sonido por encima del umbral
                if not speech_detectada:
                    print("-> Sonido detectado, comenzando monitoreo de silencio.")
                    speech_detectada = True # Marcamos que ya hubo sonido
                silencio_iniciado = None # Reiniciar contador de silencio
                chunks_silenciosos_actuales = 0
            elif speech_detectada:
                # Está por debajo del umbral Y ya habíamos detectado sonido antes
                if silencio_iniciado is None:
                    # Empezar a contar el silencio
                    silencio_iniciado = time.time()
                    chunks_silenciosos_actuales = 1
                else:
                    # Incrementar contador de chunks silenciosos
                    chunks_silenciosos_actuales += 1

                # Comprobar si hemos estado en silencio el tiempo suficiente
                # if time.time() - silencio_iniciado >= silence_duration:
                if chunks_silenciosos_actuales >= chunks_de_silencio_necesarios:
                    print(f"\nSilencio detectado durante {silence_duration} segundos.")
                    grabando = False # Señal para detener el bucle
                    # No hacemos break aquí para permitir que el bucle termine limpiamente
            # Si RMS <= threshold PERO speech_detectada es False, no hacemos nada
            # (seguimos esperando a que el usuario empiece a hablar)

    except KeyboardInterrupt:
        print("\nGrabación interrumpida manualmente.")
        grabando = False # Detener el bucle
    except Exception as e:
        print(f"\nERROR durante la grabación: {e}")
        if stream:
            stream.stop()
            stream.close()
        return False
    finally:
        # Asegurarse de detener y cerrar el stream
        if stream and stream.active:
            print("Deteniendo stream de audio...")
            stream.stop()
            stream.close()
            print("Stream cerrado.")

    # Si no hubo error grave y tenemos datos grabados
    if grabacion_completa:
        print("Grabación finalizada.")
        # Concatenar todos los chunks en un único array numpy
        grabacion_final = np.concatenate(grabacion_completa, axis=0)

        # Guardar en archivo WAV
        print(f"Guardando grabación en '{filename}'...")
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(np.dtype(DTYPE).itemsize)
                wf.setframerate(samplerate)
                wf.writeframes(grabacion_final.tobytes())
            print("Grabación guardada exitosamente.")
            return True
        except Exception as e:
            print(f"Error al guardar el archivo WAV: {e}")
            return False
    else:
        print("No se grabó ningún dato.")
        return False

# --- Carga del Modelo Whisper (Solo CPU) ---
# (La función cargar_modelo_whisper es la misma que antes)
def cargar_modelo_whisper(model_name="base"):
    """Carga el modelo Whisper especificado, forzando el uso de CPU."""
    print(f"\nCargando modelo Whisper '{model_name}' (forzado a CPU)...")
    if not torch.cuda.is_available():
        print("Confirmado: PyTorch está configurado solo para CPU.")
    else:
        print("CUDA detectado, pero forzando uso de CPU.")

    try:
        model = whisper.load_model(model_name, device='cpu')
        print(f"Modelo '{model_name}' cargado exitosamente en CPU.")
        return model
    except Exception as e:
        print(f"ERROR al cargar el modelo Whisper '{model_name}': {e}")
        return None

# --- Programa Principal ---
if __name__ == "__main__":
    # 1. Grabar el audio con detección de silencio
    grabacion_exitosa = grabar_con_silencio(FILENAME_TEMP, SAMPLERATE, CHANNELS, CHUNK_SIZE, SILENCE_THRESHOLD, SILENCE_DURATION)

    # 2. Cargar modelo Whisper (si la grabación fue exitosa)
    modelo_whisper = None
    if grabacion_exitosa:
        modelo_whisper = cargar_modelo_whisper(model_name="base") # Elige tu modelo

    # 3. Transcribir (si todo fue exitoso)
    if grabacion_exitosa and modelo_whisper:
        print("\nIniciando transcripción del audio grabado...")
        start_time = time.time()
        try:
            result = modelo_whisper.transcribe(FILENAME_TEMP, fp16=False) # fp16=False para CPU
            end_time = time.time()

            print(f"\n--- TRANSCRIPCIÓN COMPLETA ({end_time - start_time:.2f} segundos) ---")
            print(result["text"].strip()) # .strip() para quitar espacios extra
            print("-" * 50)

        except Exception as e:
            print(f"ERROR durante la transcripción: {e}")

    elif not grabacion_exitosa:
        print("\nNo se puede transcribir porque la grabación falló o no se guardó.")
    else: # grabacion_exitosa=True, pero modelo_whisper=None
        print("\nNo se puede transcribir porque el modelo Whisper no se pudo cargar.")

    # 4. Limpieza (Opcional)
    if os.path.exists(FILENAME_TEMP):
        try:
            # Podrías preguntar al usuario si quiere borrarlo
            # input("Presiona Enter para borrar el archivo temporal o Ctrl+C para salir.")
            os.remove(FILENAME_TEMP)
            print(f"Archivo temporal '{FILENAME_TEMP}' eliminado.")
        except Exception as e:
            print(f"Advertencia: No se pudo eliminar el archivo temporal '{FILENAME_TEMP}'. Error: {e}")

    print("\nProceso finalizado.")