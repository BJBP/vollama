# -*- coding: utf-8 -*-

import sounddevice as sd
import numpy as np
import wave
import whisper
import torch
import time
import os
import math
from pathlib import Path
from piper.voice import PiperVoice
import playsound
import ollama
import sys

# --- 1. Configuración General ---

# -- STT (Speech-to-Text) --
SAMPLERATE = 16000  # Tasa de muestreo (Hz) para grabación
CHANNELS = 1        # Mono
FILENAME_REC_TEMP = "grabacion_temporal.wav" # Archivo temporal para grabación
DTYPE_REC = 'int16' # Tipo de dato de grabación
CHUNK_SIZE_REC = 1024 # Tamaño del bloque para análisis de silencio
SILENCE_THRESHOLD = 1000  # Umbral de silencio (¡AJUSTA ESTO SEGÚN TU MICRÓFONO Y AMBIENTE!)
SILENCE_DURATION = 2.0  # Segundos de silencio para detener grabación
WHISPER_MODEL = "base" # Modelo Whisper a usar ("tiny", "base", "small", "medium", "large") - 'base' es un buen compromiso
# Forzar CPU para Whisper (más compatible, aunque más lento si tienes GPU)
WHISPER_DEVICE = "cpu" # Cambia a "cuda" si tienes GPU NVIDIA y PyTorch con soporte CUDA

# -- LLM (Large Language Model) --
# Asegúrate de tener este modelo en Ollama (ollama list)
OLLAMA_MODEL = 'phi4-mini:latest' # Puedes usar 'llama3', 'mistral', 'phi4-mini:latest', etc.
# Prompt del sistema simple para conversación general
SYSTEM_PROMPT_CONVERSACION = "Eres un asistente de IA conversacional útil y amigable. Responde de forma concisa y directa."

# -- TTS (Text-to-Speech) --
# Ruta a la carpeta que contiene los archivos .onnx y .onnx.json del modelo Piper TTS
TTS_MODEL_DIR = Path("./modelos/es/es_ES/carlfm/x_low") # AJUSTA ESTA RUTA
TTS_MODEL_ONNX = "es_ES-carlfm-x_low.onnx"
TTS_MODEL_JSON = "es_ES-carlfm-x_low.onnx.json"
FILENAME_TTS_TEMP = "respuesta_tts_temporal.wav" # Archivo temporal para la respuesta hablada

# -- Control --
EXIT_COMMAND = "adiós" # Palabra o frase para terminar la conversación

# --- 2. Funciones Adaptadas de tus Scripts ---

# == Funciones STT (de stt.py) ==

def grabar_con_silencio(filename, samplerate, channels, dtype, chunk_size, silence_threshold, silence_duration):
    """Graba audio hasta detectar un periodo de silencio."""
    print(f"\n🎙️ Escuchando... (Habla ahora, {silence_duration}s de silencio para parar)")
    print(f"   Di '{EXIT_COMMAND}' para salir.")

    grabacion_completa = []
    grabando = True
    silencio_iniciado = None
    speech_detectada = False
    stream = None

    chunks_de_silencio_necesarios = int((silence_duration * samplerate) / chunk_size)
    chunks_silenciosos_actuales = 0

    try:
        stream = sd.InputStream(samplerate=samplerate, channels=channels, dtype=dtype, blocksize=chunk_size)
        stream.start()

        while grabando:
            chunk, overflowed = stream.read(chunk_size)
            if overflowed:
                print("¡Advertencia! Overflow de audio detectado.")

            grabacion_completa.append(chunk)
            rms = np.sqrt(np.mean(np.square(chunk.astype(np.float32))))
            # Descomenta para calibrar el umbral:
            # print(f"RMS: {rms:.4f}")

            if rms > silence_threshold:
                if not speech_detectada:
                    print("   (Sonido detectado)")
                    speech_detectada = True
                silencio_iniciado = None
                chunks_silenciosos_actuales = 0
            elif speech_detectada:
                if silencio_iniciado is None:
                    silencio_iniciado = time.time()
                    chunks_silenciosos_actuales = 1
                else:
                    chunks_silenciosos_actuales += 1

                if chunks_silenciosos_actuales >= chunks_de_silencio_necesarios:
                    # print(f"\n   (Silencio detectado, deteniendo grabación)") # Opcional: menos verboso
                    grabando = False

    except KeyboardInterrupt:
        print("\nGrabación interrumpida manualmente.")
        grabando = False
    except Exception as e:
        print(f"\n❌ ERROR durante la grabación: {e}")
        return None # Indicar fallo
    finally:
        if stream:
            if stream.active:
                stream.stop()
            stream.close()

    if not grabacion_completa or not speech_detectada:
        print("   (No se detectó habla significativa)")
        return None # No se grabó nada útil

    print("   Grabación finalizada.")
    grabacion_final = np.concatenate(grabacion_completa, axis=0)

    print(f"   Guardando grabación temporal en '{filename}'...")
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(np.dtype(dtype).itemsize)
            wf.setframerate(samplerate)
            wf.writeframes(grabacion_final.tobytes())
        print("   Grabación guardada.")
        return filename # Devolver el nombre del archivo si éxito
    except Exception as e:
        print(f"❌ Error al guardar el archivo WAV: {e}")
        return None

def cargar_modelo_whisper(model_name, device):
    """Carga el modelo Whisper especificado."""
    print(f"\n🔄 Cargando modelo Whisper '{model_name}' en dispositivo '{device}'...")
    try:
        # Verificar si el dispositivo es CPU y si CUDA está disponible pero no se usa
        if device == 'cpu' and torch.cuda.is_available():
            print("   (CUDA disponible, pero forzando uso de CPU)")
        elif device == 'cuda' and not torch.cuda.is_available():
            print("   (Se solicitó CUDA pero no está disponible, usando CPU en su lugar)")
            device = 'cpu' # Cambiar a CPU si CUDA no está

        model = whisper.load_model(model_name, device=device)
        print(f"   ✅ Modelo Whisper '{model_name}' cargado en {device}.")
        return model
    except Exception as e:
        print(f"❌ ERROR al cargar el modelo Whisper '{model_name}': {e}")
        # Intenta con un modelo más pequeño como fallback si falla el original? (opcional)
        if model_name != "tiny":
            print("   Intentando cargar el modelo 'tiny' como alternativa...")
            return cargar_modelo_whisper("tiny", device)
        return None

def transcribir_audio(modelo_whisper, filename):
    """Transcribe el archivo de audio usando el modelo Whisper cargado."""
    if not modelo_whisper or not filename or not os.path.exists(filename):
        print("❌ No se puede transcribir: falta modelo, archivo o el archivo no existe.")
        return None

    print(f"🖊️ Transcribiendo '{filename}'...")
    try:
        # fp16=False es generalmente más seguro para CPU o si no estás seguro de compatibilidad
        result = modelo_whisper.transcribe(filename, fp16=False if WHISPER_DEVICE == 'cpu' else True)
        texto_transcrito = result["text"].strip()
        print(f"   Texto transcrito: '{texto_transcrito}'")
        return texto_transcrito
    except Exception as e:
        print(f"❌ ERROR durante la transcripción: {e}")
        return None

# == Funciones LLM (inspiradas en llm.py) ==

def check_ollama_model(model_name):
    """Verifica si el modelo especificado existe en Ollama."""
    model_name = 'phi4-mini:latest'
    print(f"🔍 Verificando si el modelo Ollama '{model_name}' está disponible...")
    try:
        models_data = ollama.list()
        if 'models' not in models_data or not isinstance(models_data['models'], list):
            print("❌ Error: Respuesta inesperada de 'ollama list'.")
            return False

        clean_model_name = model_name.strip()
        found = any(
            clean_model_name == m.get('name', '').strip() or \
            clean_model_name == m.get('model', '').strip() or \
            clean_model_name == m.get('name', '').split(':')[0].strip() or \
            clean_model_name == m.get('model', '').split(':')[0].strip()
            for m in models_data['models'] if isinstance(m, dict)
        )

        if found:
            print(f"   ✅ Modelo '{model_name}' encontrado en Ollama.")
            return True
        else:
            print(f"   ❌ Modelo '{OLLAMA_MODEL}' NO encontrado. Verifica con 'ollama list'.")
            print(f"   Puedes intentar descargarlo con: ollama pull {OLLAMA_MODEL}")
            return False
    except Exception as e:
        print(f"❌ Error al conectar con Ollama o listar modelos: {e}")
        print("   Asegúrate de que Ollama esté instalado y ejecutándose ('ollama serve').")
        return False

def obtener_respuesta_llm(modelo_llm, historial_mensajes):
    """Obtiene una respuesta del LLM usando Ollama."""
    print("🧠 Pensando...")
    try:
        response = ollama.chat(model='phi4-mini:latest', messages=historial_mensajes, stream=False)
        respuesta = response['message']['content']
        print(f"   Respuesta LLM: '{respuesta}'")
        return respuesta
    except Exception as e:
        print(f"❌ ERROR al obtener respuesta de Ollama ({modelo_llm}): {e}")
        return "Lo siento, tuve un problema al procesar tu solicitud."

# == Funciones TTS (de tts.py) ==

def cargar_modelo_tts(ruta_onnx, ruta_json):
    """Carga el modelo de voz Piper TTS."""
    print(f"\n🔄 Cargando modelo TTS desde '{ruta_onnx.parent}'...")
    if not ruta_onnx.exists() or not ruta_json.exists():
        print(f"❌ Error: No se encontraron los archivos del modelo TTS:")
        print(f"   - {ruta_onnx}")
        print(f"   - {ruta_json}")
        return None, None

    try:
        voz = PiperVoice.load(str(ruta_onnx), str(ruta_json))
        sample_rate = voz.config.sample_rate
        print(f"   ✅ Modelo TTS cargado (Sample rate: {sample_rate} Hz).")
        return voz, sample_rate
    except Exception as e:
        print(f"❌ Error al cargar el modelo TTS: {e}")
        return None, None

def sintetizar_y_reproducir(voz_tts, sample_rate, texto, filename_out):
    """Sintetiza el texto a un archivo WAV y lo reproduce."""
    if not voz_tts:
        print("❌ No se puede sintetizar: modelo TTS no cargado.")
        return False

    print(f"🗣️ Sintetizando respuesta: '{texto}'...")
    try:
        with wave.open(filename_out, "wb") as archivo_wav:
            archivo_wav.setnchannels(1)
            archivo_wav.setsampwidth(2) # Asumiendo 16-bit
            archivo_wav.setframerate(sample_rate)
            # Pasar el objeto wave directamente a synthesize
            voz_tts.synthesize(texto, archivo_wav)
        print(f"   Audio guardado temporalmente en '{filename_out}'.")
        print("   Reproduciendo...")
        playsound.playsound(filename_out)
        return True
    except Exception as e:
        print(f"❌ Error durante la síntesis o reproducción TTS: {e}")
        return False
    finally:
        # Opcional: eliminar el archivo temporal de TTS después de reproducirlo
         if os.path.exists(filename_out):
             try:
                 os.remove(filename_out)
                 # print(f"   Archivo TTS temporal '{filename_out}' eliminado.") # Opcional
             except Exception as e:
                 print(f"   Advertencia: No se pudo eliminar '{filename_out}': {e}")


# --- 3. Bucle Principal de Conversación ---

def main():
    """Función principal que orquesta la conversación."""

    # --- Inicialización ---
    print("--- Asistente de Voz Conversacional ---")

    # Cargar modelo STT (Whisper)
    modelo_whisper = cargar_modelo_whisper(WHISPER_MODEL, WHISPER_DEVICE)
    if not modelo_whisper:
        print("❌ No se pudo cargar el modelo Whisper. Saliendo.")
        return

    # Verificar modelo LLM (Ollama)
    #if not check_ollama_model(OLLAMA_MODEL):
    #    print("❌ El modelo LLM especificado no está disponible. Saliendo.")
    #    return

    # Cargar modelo TTS (Piper)
    ruta_modelo_onnx = TTS_MODEL_DIR / TTS_MODEL_ONNX
    ruta_modelo_json = TTS_MODEL_DIR / TTS_MODEL_JSON
    voz_tts, tts_sample_rate = cargar_modelo_tts(ruta_modelo_onnx, ruta_modelo_json)
    if not voz_tts:
        print("❌ No se pudo cargar el modelo TTS. Saliendo.")
        return

    # Inicializar historial de conversación para el LLM
    historial_conversacion = [{'role': 'system', 'content': SYSTEM_PROMPT_CONVERSACION}]

    print("\n--- Inicio de la Conversación ---")
    print(f"Di '{EXIT_COMMAND}' para terminar.")

    # --- Bucle de Interacción ---
    while True:
        # 1. Grabar audio del usuario con detección de silencio
        ruta_audio_grabado = grabar_con_silencio(
            FILENAME_REC_TEMP,
            SAMPLERATE,
            CHANNELS,
            DTYPE_REC,
            CHUNK_SIZE_REC,
            SILENCE_THRESHOLD,
            SILENCE_DURATION
        )

        # Si la grabación falló o no hubo habla, volver a escuchar
        if not ruta_audio_grabado:
            # Limpiar archivo temporal si existe pero falló la lógica
            if os.path.exists(FILENAME_REC_TEMP):
                 os.remove(FILENAME_REC_TEMP)
            continue # Vuelve al inicio del bucle para escuchar de nuevo

        # 2. Transcribir el audio grabado a texto
        texto_usuario = transcribir_audio(modelo_whisper, ruta_audio_grabado)

        # Limpiar el archivo de grabación temporal después de transcribir
        if os.path.exists(ruta_audio_grabado):
            try:
                os.remove(ruta_audio_grabado)
                # print(f"   Archivo de grabación temporal '{ruta_audio_grabado}' eliminado.") # Opcional
            except Exception as e:
                print(f"   Advertencia: No se pudo eliminar '{ruta_audio_grabado}': {e}")

        # Si la transcripción falló o está vacía, volver a escuchar
        if not texto_usuario:
            print("   No se pudo obtener texto de la grabación.")
            continue # Vuelve al inicio del bucle

        # 3. Comprobar comando de salida
        if EXIT_COMMAND.lower() in texto_usuario.lower():
            print("\n👋 Detectado comando de salida. ¡Hasta luego!")
            mensaje_despedida = "¡Entendido! Que tengas un buen día."
             # Decir adiós antes de salir
            sintetizar_y_reproducir(voz_tts, tts_sample_rate, mensaje_despedida, FILENAME_TTS_TEMP)
            break # Salir del bucle while

        # 4. Añadir mensaje del usuario al historial y obtener respuesta del LLM
        historial_conversacion.append({'role': 'user', 'content': texto_usuario})
        respuesta_llm = obtener_respuesta_llm(OLLAMA_MODEL, historial_conversacion)

        # 5. Añadir respuesta del LLM al historial
        if respuesta_llm:
             historial_conversacion.append({'role': 'assistant', 'content': respuesta_llm})
        else:
             # Si el LLM falla, usar una respuesta genérica y no añadirla (¿o sí?)
             respuesta_llm = "No estoy seguro de cómo responder a eso."
             # Opcional: añadir también el fallo al historial para contexto futuro
             # historial_conversacion.append({'role': 'assistant', 'content': respuesta_llm})


        # 6. Sintetizar la respuesta del LLM a voz y reproducirla
        sintetizar_y_reproducir(voz_tts, tts_sample_rate, respuesta_llm, FILENAME_TTS_TEMP)

        # Fin del ciclo, vuelve a escuchar

    # --- Limpieza Final ---
    print("\n--- Fin de la Conversación ---")
    # Asegurarse de que los archivos temporales se eliminan si aún existen
    for temp_file in [FILENAME_REC_TEMP, FILENAME_TTS_TEMP]:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                # No es crítico si falla la eliminación aquí
                pass

# --- Punto de Entrada ---
if __name__ == "__main__":
    # Verificar dependencias básicas (Ollama es externo, Piper/Whisper se manejan en carga)
     try:
         # Solo para verificar que las librerías principales están instaladas
         import sounddevice
         import whisper
         import ollama
         import piper.voice
         import playsound
     except ImportError as e:
         print(f"❌ Error de importación: {e}")
         print("Asegúrate de haber instalado todas las dependencias necesarias:")
         print("pip install sounddevice numpy wave-fix torch whisper-openai ollama piper-tts playsound")
         # Nota: wave-fix puede ser necesario en algunos sistemas en lugar de solo wave
         # Nota: Para whisper-openai, puede que necesites instalar ffmpeg en tu sistema.
         # Nota: Para piper-tts, puede que necesites dependencias adicionales (onnxruntime).
         # Consulta la documentación de cada librería.
         sys.exit(1)

     main()