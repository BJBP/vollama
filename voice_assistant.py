# --- Imports Combinados ---
import ollama
import sys
import sounddevice as sd
import numpy as np
import wave
import whisper
import torch
import time
import os
import math # Para calcular RMS
from pathlib import Path
from piper.voice import PiperVoice
from playsound import playsound
import yaml # Para leer config de Piper si es necesario

# --- Configuración General ---
# LLM
DEFAULT_LLM_MODEL = 'phi3:mini' # Asegúrate que este modelo existe en Ollama (ollama list)
SYSTEM_PROMPT_CONVERSATIONAL = "Eres un asistente de IA conversacional y útil. Responde de forma clara y concisa."

# STT (Speech-to-Text)
SAMPLERATE = 16000
CHANNELS = 1
FILENAME_STT_TEMP = "grabacion_stt_temp.wav" # Archivo temporal para la grabación
DTYPE = 'int16'
CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 300  # <-- ¡¡AJUSTA ESTE VALOR!! Experimenta. Más bajo = más sensible.
SILENCE_DURATION = 2.0   # Segundos de silencio para detener grabación
WHISPER_MODEL = "base"   # ("tiny", "base", "small", "medium", "large") - modelos más grandes son más precisos pero más lentos/pesados

# TTS (Text-to-Speech)
# Ajusta la ruta a tu carpeta de modelos Piper
ruta_carpeta_modelos_tts = Path("./modelos/es/es_ES/carlfm/x_low")
nombre_modelo_onnx_tts = "es_ES-carlfm-x_low.onnx"
nombre_modelo_json_tts = "es_ES-carlfm-x_low.onnx.json"
FILENAME_TTS_TEMP = "respuesta_tts_temp.wav" # Archivo temporal para la respuesta hablada

# Construir rutas TTS completas
ruta_modelo_onnx = ruta_carpeta_modelos_tts / nombre_modelo_onnx_tts
ruta_modelo_json = ruta_carpeta_modelos_tts / nombre_modelo_json_tts

# --- Variables Globales STT ---
grabacion_completa_stt = []
grabando_stt = False
silencio_iniciado_stt = None
speech_detectada_stt = False

# --- Funciones STT (Adaptadas de stt.py) ---

def grabar_con_silencio(filename, samplerate, channels, dtype, chunk_size, silence_threshold, silence_duration):
    """Graba audio hasta detectar silencio y guarda en filename. Devuelve True si éxito, False si error."""
    global grabacion_completa_stt, grabando_stt, silencio_iniciado_stt, speech_detectada_stt
    grabacion_completa_stt = []
    grabando_stt = True
    silencio_iniciado_stt = None
    speech_detectada_stt = False

    chunks_de_silencio_necesarios = int((silence_duration * samplerate) / chunk_size)
    chunks_silenciosos_actuales = 0

    print("\n" + "="*10 + " Escuchando... Habla ahora " + "="*10)
    print(f"(La grabación se detendrá tras {silence_duration}s de silencio)")

    stream = None
    try:
        stream = sd.InputStream(samplerate=samplerate,
                                channels=channels,
                                dtype=dtype,
                                blocksize=chunk_size)
        stream.start()

        while grabando_stt:
            chunk, overflowed = stream.read(chunk_size)
            if overflowed:
                print("¡Advertencia STT! Overflow detectado.")

            grabacion_completa_stt.append(chunk)
            rms = np.sqrt(np.mean(np.square(chunk.astype(np.float32))))
            # Descomenta para calibrar:
            # print(f"RMS: {rms:.1f}", end='\r')

            if rms > silence_threshold:
                if not speech_detectada_stt:
                    #print("\n-> Sonido detectado, monitoreando silencio.") # Opcional: puede ser ruidoso
                    speech_detectada_stt = True
                silencio_iniciado_stt = None
                chunks_silenciosos_actuales = 0
            elif speech_detectada_stt:
                if silencio_iniciado_stt is None:
                    silencio_iniciado_stt = time.time()
                    chunks_silenciosos_actuales = 1
                else:
                    chunks_silenciosos_actuales += 1

                if chunks_silenciosos_actuales >= chunks_de_silencio_necesarios:
                    print(f"\n-> Silencio detectado. Grabación finalizada.")
                    grabando_stt = False

    except KeyboardInterrupt:
        print("\nGrabación interrumpida manualmente.")
        grabando_stt = False
        return False # Indicar que no se completó normalmente
    except Exception as e:
        print(f"\nERROR STT durante la grabación: {e}")
        if stream:
            try:
                stream.stop()
                stream.close()
            except Exception as e_close:
                 print(f"Error al cerrar stream STT: {e_close}")
        return False # Indicar error
    finally:
        if stream and stream.active:
            stream.stop()
            stream.close()
        # Limpiar el indicador de RMS de la pantalla
        # print(" " * 20, end='\r')


    if not grabacion_completa_stt or not speech_detectada_stt:
         print("No se detectó suficiente audio o hubo un problema.")
         # Borrar archivo si existe y está vacío o casi vacío
         if os.path.exists(filename):
             try:
                 if os.path.getsize(filename) < 1024: # Si es muy pequeño, probablemente solo ruido/silencio
                     os.remove(filename)
             except OSError:
                 pass # Ignorar si no se puede borrar
         return False


    print(f"Guardando grabación temporal en '{filename}'...")
    try:
        grabacion_final = np.concatenate(grabacion_completa_stt, axis=0)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(np.dtype(dtype).itemsize)
            wf.setframerate(samplerate)
            wf.writeframes(grabacion_final.tobytes())
        print("Grabación guardada.")
        return True
    except Exception as e:
        print(f"Error STT al guardar el archivo WAV: {e}")
        return False

def cargar_modelo_whisper(model_name="base"):
    """Carga el modelo Whisper especificado, forzando CPU si no hay CUDA."""
    print(f"\nCargando modelo Whisper '{model_name}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device.upper()}")
    try:
        model = whisper.load_model(model_name, device=device)
        print(f"Modelo Whisper '{model_name}' cargado exitosamente en {device.upper()}.")
        return model
    except Exception as e:
        print(f"ERROR al cargar el modelo Whisper '{model_name}': {e}")
        return None

def transcribir_audio(modelo_whisper, filename):
    """Transcribe el archivo de audio usando el modelo Whisper cargado."""
    if not modelo_whisper:
        print("Error: Modelo Whisper no está cargado.")
        return None
    if not os.path.exists(filename):
        print(f"Error: Archivo de audio no encontrado: {filename}")
        return None

    print(f"Transcribiendo '{filename}'...")
    try:
        # fp16=False es más seguro para CPU, True puede ser más rápido en GPU
        result = modelo_whisper.transcribe(filename, fp16=torch.cuda.is_available())
        texto_transcrito = result["text"].strip()
        print(f"Transcripción finalizada.")
        return texto_transcrito
    except Exception as e:
        print(f"ERROR durante la transcripción: {e}")
        return None

# --- Funciones LLM (Adaptadas de llm.py) ---

def check_ollama_model(model_name):
    """Verifica si el modelo LLM existe en Ollama."""
    print(f"Verificando modelo Ollama: '{model_name}'...")
    try:
        models_data = ollama.list()
        if 'models' not in models_data or not isinstance(models_data['models'], list):
            print("\n❌ Error: Formato inesperado de 'ollama list'.")
            return False
        
        clean_model_name = model_name.strip()
        found = any(
            # Verifica que 'm' tenga 'name' y sea string antes de comparar
            hasattr(m, 'name') and isinstance(m['name'], str) and 
            (m['name'].strip() == clean_model_name or m['name'].strip().split(':')[0] == clean_model_name)
            for m in models_data['models']
        )

        if found:
             print("Modelo Ollama encontrado.")
        else:
             print(f"Modelo Ollama '{model_name}' NO encontrado.")
             print("Modelos disponibles:")
             for m in models_data['models']:
                  if hasattr(m, 'name') and isinstance(m['name'], str):
                      print(f"- {m['name']}")
        return found
    except Exception as e:
        print(f"\n❌ Error al conectar con Ollama o listar modelos: {e}")
        print("Asegúrate de que Ollama esté instalado y ejecutándose ('ollama serve').")
        return False

def obtener_respuesta_llm(texto_usuario, historial_mensajes, model_name):
    """Obtiene la respuesta del LLM usando Ollama y actualiza el historial."""
    print("Pensando...")
    historial_mensajes.append({'role': 'user', 'content': texto_usuario})
    try:
        response = ollama.chat(model=model_name, messages=historial_mensajes, stream=False)
        respuesta_llm = response['message']['content']
        historial_mensajes.append({'role': 'assistant', 'content': respuesta_llm})
        return respuesta_llm
    except Exception as e:
        print(f"\n❌ Error al comunicarse con el LLM ({model_name}): {e}")
        # Eliminar el último mensaje del usuario si la llamada falla, para no reenviarlo erróneamente
        if historial_mensajes and historial_mensajes[-1]['role'] == 'user':
            historial_mensajes.pop()
        return "Lo siento, tuve un problema para procesar tu solicitud."

# --- Funciones TTS (Adaptadas de tts.py) ---

def cargar_modelo_tts(ruta_onnx, ruta_json):
    """Carga el modelo de voz Piper."""
    print(f"\nCargando modelo TTS desde: {ruta_onnx.parent}...")
    if not ruta_onnx.exists() or not ruta_json.exists():
        print(f"Error TTS: No se encontraron los archivos del modelo:")
        print(f"- {ruta_onnx}")
        print(f"- {ruta_json}")
        return None
    try:
        voz = PiperVoice.load(str(ruta_onnx), str(ruta_json))
        print("Modelo TTS cargado correctamente.")
        return voz
    except Exception as e:
        print(f"Error TTS al cargar el modelo: {e}")
        # Intenta leer la configuración JSON para obtener la tasa de muestreo incluso si falla la carga completa
        try:
             with open(ruta_json, 'r', encoding='utf-8') as f:
                 config = yaml.safe_load(f)
                 if 'audio' in config and 'sample_rate' in config['audio']:
                      print(f"(Tasa de muestreo según JSON: {config['audio']['sample_rate']})")
        except Exception:
             pass # Ignora errores al leer el JSON si la carga principal falló
        return None


def sintetizar_y_reproducir(modelo_tts, texto, filename_out, play_sound=True):
    """Sintetiza el texto a un archivo WAV y lo reproduce."""
    if not modelo_tts:
        print("Error TTS: Modelo no cargado.")
        return False

    print("Sintetizando respuesta...")
    try:
        # Asegurarse de que el modelo tiene la info de sample rate
        if not hasattr(modelo_tts, 'config') or not hasattr(modelo_tts.config, 'sample_rate'):
             print("Error TTS: No se pudo determinar la tasa de muestreo del modelo.")
             # Intenta adivinar una tasa común si falla todo lo demás
             sample_rate = 22050 # Valor común, pero puede ser incorrecto
             print(f"Advertencia: Usando tasa de muestreo predeterminada {sample_rate}Hz.")
        else:
             sample_rate = modelo_tts.config.sample_rate

        with wave.open(filename_out, "wb") as archivo_wav:
            archivo_wav.setnchannels(1)
            # Piper usualmente usa 16-bit audio = 2 bytes
            archivo_wav.setsampwidth(2)
            archivo_wav.setframerate(sample_rate)
            # Pasar el objeto wave para escribir directamente
            modelo_tts.synthesize(texto, archivo_wav)

        print(f"Audio sintetizado guardado como '{filename_out}'")

        if play_sound:
            print("Reproduciendo respuesta...")
            playsound(filename_out)
            print("Reproducción finalizada.")
        return True

    except Exception as e:
        print(f"Error TTS durante la síntesis o reproducción: {e}")
        return False
    finally:
        # Opcional: Borrar el archivo temporal después de reproducirlo
        if play_sound and os.path.exists(filename_out):
             try:
                 # time.sleep(0.5) # Pequeña pausa por si playsound no ha liberado el archivo
                 os.remove(filename_out)
                 # print(f"Archivo temporal TTS '{filename_out}' eliminado.") # Opcional
             except Exception as e_del:
                 print(f"Advertencia: No se pudo eliminar el archivo TTS temporal '{filename_out}'. Error: {e_del}")


# --- Flujo Principal de Conversación ---
if __name__ == "__main__":
    print("\n--- Asistente de Voz con LLM ---")

    # 1. Cargar Modelos al inicio
    modelo_whisper_cargado = cargar_modelo_whisper(WHISPER_MODEL)
    modelo_tts_cargado = cargar_modelo_tts(ruta_modelo_onnx, ruta_modelo_json)

    # Verificar modelo Ollama
    llm_model_name = DEFAULT_LLM_MODEL
    if len(sys.argv) > 1:
        llm_model_name = sys.argv[1].strip()
        print(f"Usando modelo LLM especificado: {llm_model_name}")
    
    if not check_ollama_model(llm_model_name):
         print("El modelo LLM especificado no está disponible. Saliendo.")
         sys.exit(1)


    if not modelo_whisper_cargado:
        print("No se pudo cargar el modelo Whisper. Funcionalidad STT desactivada.")
        # Podrías decidir salir o continuar solo con texto
        # sys.exit(1)
    if not modelo_tts_cargado:
        print("No se pudo cargar el modelo Piper TTS. Funcionalidad TTS desactivada.")
        # sys.exit(1)

    # 2. Inicializar historial de conversación del LLM
    historial_llm = [{'role': 'system', 'content': SYSTEM_PROMPT_CONVERSATIONAL}]

    print("\nListo para conversar. Di 'adiós' o 'salir' para terminar.")
    print("-" * 40)

    try:
        while True:
            # 3. Grabar audio del usuario
            if not modelo_whisper_cargado:
                 print("STT no disponible. Ingresa tu texto manualmente:")
                 texto_usuario = input("👤 Tú: ")
            elif grabar_con_silencio(FILENAME_STT_TEMP, SAMPLERATE, CHANNELS, DTYPE, CHUNK_SIZE, SILENCE_THRESHOLD, SILENCE_DURATION):
                # 4. Transcribir audio a texto
                texto_usuario = transcribir_audio(modelo_whisper_cargado, FILENAME_STT_TEMP)

                # Limpieza del archivo STT temporal
                if os.path.exists(FILENAME_STT_TEMP):
                    try:
                        os.remove(FILENAME_STT_TEMP)
                    except Exception as e_del_stt:
                        print(f"Advertencia: No se pudo eliminar '{FILENAME_STT_TEMP}'. Error: {e_del_stt}")

                if not texto_usuario:
                    print("No pude entender lo que dijiste. Intenta de nuevo.")
                    # Opcionalmente, decir algo con TTS si está disponible
                    if modelo_tts_cargado:
                         sintetizar_y_reproducir(modelo_tts_cargado, "Lo siento, no te entendí bien. ¿Puedes repetirlo?", FILENAME_TTS_TEMP)
                    continue # Volver a escuchar
            else:
                # La grabación falló o fue interrumpida sin habla detectada
                print("No se grabó audio válido. Intenta de nuevo.")
                 # Opcionalmente, decir algo con TTS
                if modelo_tts_cargado:
                    sintetizar_y_reproducir(modelo_tts_cargado, "Parece que hubo un problema con la grabación. ¿Intentamos de nuevo?", FILENAME_TTS_TEMP)
                time.sleep(1) # Pequeña pausa antes de reintentar
                continue # Volver a escuchar

            # Mostrar texto transcrito
            print(f"👤 Tú (detectado): {texto_usuario}")

            # 5. Comprobar si el usuario quiere salir
            if texto_usuario.lower() in ["adiós", "adios", "salir", "terminar", "exit", "quit"]:
                print("🤖 IA: ¡Hasta luego!")
                if modelo_tts_cargado:
                    sintetizar_y_reproducir(modelo_tts_cargado, "¡Hasta luego!", FILENAME_TTS_TEMP)
                break # Salir del bucle principal

            # 6. Obtener respuesta del LLM
            respuesta_texto_llm = obtener_respuesta_llm(texto_usuario, historial_llm, llm_model_name)

            # Mostrar respuesta del LLM
            print(f"🤖 IA: {respuesta_texto_llm}")

            # 7. Sintetizar y reproducir la respuesta del LLM
            if modelo_tts_cargado:
                sintetizar_y_reproducir(modelo_tts_cargado, respuesta_texto_llm, FILENAME_TTS_TEMP)
            else:
                print("(TTS no disponible para leer la respuesta)")

            # Pequeña pausa para evitar que la siguiente escucha empiece inmediatamente
            # time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n👋 Interrupción manual detectada. Saliendo.")
        if modelo_tts_cargado:
             sintetizar_y_reproducir(modelo_tts_cargado, "Entendido. Terminando la sesión.", FILENAME_TTS_TEMP)
    except Exception as e_main:
        print(f"\n❌ Ocurrió un error inesperado en el bucle principal: {e_main}")
    finally:
        # Limpieza final si quedan archivos temporales
        for temp_file in [FILENAME_STT_TEMP, FILENAME_TTS_TEMP]:
             if os.path.exists(temp_file):
                  try:
                      os.remove(temp_file)
                  except Exception:
                      pass # Ignorar errores en la limpieza final
        print("\n--- Fin de la Conversación ---")