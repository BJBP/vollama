# --- Combined AI Interview Coach with STT and TTS ---

import ollama
import sys
import os
import time
import wave
from pathlib import Path

# STT Imports
import sounddevice as sd
import numpy as np
import whisper
import torch

# TTS Imports
from piper.voice import PiperVoice
from playsound import playsound # Ensure playsound is installed (pip install playsound)

# --- LLM Configuration ---
DEFAULT_OLLAMA_MODEL = 'phi3:mini' # Make sure this model exists in Ollama

# --- STT Configuration ---
SAMPLERATE = 16000
CHANNELS = 1
FILENAME_TEMP_RECORDING = "user_response_recording.wav"
DTYPE_REC = 'int16'
CHUNK_SIZE = 1024
# Adjust these based on your microphone and environment!
SILENCE_THRESHOLD = 150 # Lower = more sensitive to noise. Calibrate this!
SILENCE_DURATION = 2.0 # Seconds of silence to stop recording
WHISPER_MODEL = "base" # "tiny", "base", "small", "medium", "large". Base is a good starting point for CPU.

# --- TTS Configuration ---
# Adjust this path to where your Piper model files are located
# Example: './models/es/es_ES/carlfm/x_low'
TTS_MODEL_FOLDER = Path("./models/es/es_ES/carlfm/x_low") # <<<--- CHANGE THIS PATH
# Assuming the model files follow the standard naming convention
# Example: 'es_ES-carlfm-x_low.onnx' and 'es_ES-carlfm-x_low.onnx.json'
# We will derive the filenames based on the folder name structure later if possible,
# or you might need to specify them explicitly if the names don't match the pattern.
# Let's try to derive them:
try:
    # Find the .onnx file in the folder
    onnx_files = list(TTS_MODEL_FOLDER.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx file found in {TTS_MODEL_FOLDER}")
    TTS_MODEL_ONNX_PATH = onnx_files[0]
    # Assume the .json file has the same base name + .json
    TTS_MODEL_JSON_PATH = TTS_MODEL_ONNX_PATH.with_suffix(".onnx.json")
    if not TTS_MODEL_JSON_PATH.exists():
        raise FileNotFoundError(f"Matching .json file not found for {TTS_MODEL_ONNX_PATH}")
    print(f"TTS Model files found: {TTS_MODEL_ONNX_PATH.name}, {TTS_MODEL_JSON_PATH.name}")
except Exception as e:
    print(f"Error automatically finding TTS model files: {e}")
    print("Please ensure TTS_MODEL_FOLDER is correct and contains the .onnx and .onnx.json files.")
    # You might need to manually set TTS_MODEL_ONNX_PATH and TTS_MODEL_JSON_PATH here
    # Example:
    # TTS_MODEL_ONNX_PATH = TTS_MODEL_FOLDER / "your_model_name.onnx"
    # TTS_MODEL_JSON_PATH = TTS_MODEL_FOLDER / "your_model_name.onnx.json"
    sys.exit(1)

FILENAME_TEMP_TTS_OUTPUT = "ai_response_speech.wav"

# --- LLM System Prompt (Same as before) ---
SYSTEM_PROMPT = """
Eres un Simulador de Entrevistas de Trabajo y Coach de IA altamente capacitado. Tu propósito es ayudar al usuario a practicar y mejorar sus respuestas en entrevistas de trabajo.

**Tu Doble Rol:**

1.  **Entrevistador:** Haz preguntas de entrevista de trabajo estándar y relevantes, una a la vez. Comienza con un saludo profesional y una breve introducción. Sigue un flujo lógico (ej: introducción, fortalezas/debilidades, motivación, preguntas situacionales/conductuales (usando STAR), preguntas del candidato).
2.  **Coach:** Después de CADA respuesta del usuario, DEBES evaluar esa respuesta específica y proporcionar feedback CONSTRUCTIVO antes de pasar a la siguiente pregunta.

**Directrices para la Evaluación y Feedback:**

*   **Evalúa CADA respuesta del usuario.** No te saltes ninguna.
*   **Feedback Estructurado:** En tu respuesta DESPUÉS de la respuesta del usuario, primero proporciona el feedback y LUEGO haz la siguiente pregunta.
*   **Contenido del Feedback:**
    *   **Si la respuesta fue buena:** Señala específicamente QUÉ la hizo buena (ej., "Excelente uso del método STAR para estructurar tu ejemplo", "Respuesta muy clara y conectada con los valores que buscamos", "Bien hecho al cuantificar tu logro", "Tu entusiasmo es palpable y positivo"). Sé específico.
    *   **Si la respuesta necesita mejorar:** Identifica CLARAMENTE qué aspecto(s) se pueden mejorar, basándote en las "Cosas a Evitar". Sé directo pero constructivo. (ej., "Intenta ser más conciso aquí", "Recuerda evitar hablar negativamente de empleadores pasados; podrías reformularlo como una experiencia de aprendizaje", "Para esta pregunta conductual, usar el método STAR (Situación, Tarea, Acción, Resultado) haría tu respuesta más impactante", "Asegúrate de responder directamente a la pregunta; pareció que divagabas un poco", "Sería bueno añadir un ejemplo concreto para ilustrar esta fortaleza"). Ofrece sugerencias si es posible.
*   **Cosas a Buscar (Positivo):** Claridad, concisión, relevancia, ejemplos concretos, método STAR (para preguntas conductuales), cuantificación de logros, actitud positiva, entusiasmo, conexión con el puesto/empresa, honestidad (sin ser perjudicial), preparación.
*   **Cosas a Evitar (Negativo):** Hablar mal de empleadores/compañeros anteriores, respuestas vagas o evasivas, respuestas demasiado cortas (monosílabos) o largas (divagaciones), sonar arrogante o poco sincero, no responder la pregunta directamente, mencionar debilidades críticas sin un plan de mejora claro, falta de entusiasmo.

**Flujo de la Interacción:**

1.  Inicia la entrevista con un saludo y la primera pregunta (ej., "Háblame de ti").
2.  Espera la respuesta del usuario.
3.  Recibe la respuesta del usuario (transcrita del audio).
4.  **Tu Turno:** Proporciona feedback sobre la respuesta del usuario Y LUEGO haz la siguiente pregunta de la entrevista.
5.  Repite los pasos 2-4.
6.  Mantén un tono profesional como entrevistador y de apoyo como coach.
7.  Después de 5-7 preguntas, puedes preguntar al usuario si desea continuar o concluir la simulación. Al concluir, ofrece un breve resumen del feedback general.

**Instrucción Inicial:** Comienza ahora. Saluda al usuario profesionalmente e inicia la entrevista con la primera pregunta.
"""

# --- Global Variables ---
ollama_model_name = ""
whisper_model = None
tts_voice = None
tts_sample_rate = None
recording_data = []
is_recording = False
silence_start_time = None
speech_detected = False

# --- STT Functions (from stt.py) ---

def record_until_silence(filename=FILENAME_TEMP_RECORDING,
                         samplerate=SAMPLERATE,
                         channels=CHANNELS,
                         chunk_size=CHUNK_SIZE,
                         silence_threshold=SILENCE_THRESHOLD,
                         silence_duration=SILENCE_DURATION):
    """Records audio until a specified duration of silence is detected."""
    global recording_data, is_recording, silence_start_time, speech_detected
    recording_data = []
    is_recording = True
    silence_start_time = None
    speech_detected = False

    silent_chunks_needed = int((silence_duration * samplerate) / chunk_size)
    silent_chunks_count = 0

    print("-" * 20)
    print("🎤 ¡ESCUCHANDO! Habla ahora...")
    print(f"(La grabación se detendrá tras {silence_duration}s de silencio)")
    print("(Puedes presionar Ctrl+C para forzar la detención)")
    print("-" * 20)

    stream = None
    try:
        stream = sd.InputStream(samplerate=samplerate,
                                channels=channels,
                                dtype=DTYPE_REC,
                                blocksize=chunk_size)
        stream.start()

        while is_recording:
            chunk, overflowed = stream.read(chunk_size)
            if overflowed:
                print("⚠️ Advertencia: Overflow detectado (posible pérdida de datos).")

            recording_data.append(chunk)
            rms = np.sqrt(np.mean(np.square(chunk.astype(np.float32))))

            # --- Silence Detection Logic ---
            # Uncomment to calibrate threshold:
            # print(f"RMS: {rms:.2f}")

            if rms > silence_threshold:
                if not speech_detected:
                    print(" -> Habla detectada. Monitoreando silencio...")
                    speech_detected = True
                silence_start_time = None
                silent_chunks_count = 0
            elif speech_detected: # Only check for silence after speech started
                if silence_start_time is None:
                    silence_start_time = time.time()
                    silent_chunks_count = 1
                else:
                    silent_chunks_count += 1

                if silent_chunks_count >= silent_chunks_needed:
                    print(f"\n -> Silencio detectado durante {silence_duration} segundos. Deteniendo grabación.")
                    is_recording = False # Signal to stop loop

    except KeyboardInterrupt:
        print("\n🛑 Grabación interrumpida manualmente.")
        is_recording = False
    except Exception as e:
        print(f"\n❌ ERROR durante la grabación: {e}")
        if stream:
            stream.stop()
            stream.close()
        return False # Indicate failure
    finally:
        if stream and stream.active:
            print(" -> Deteniendo stream de audio...")
            stream.stop()
            stream.close()
            print(" -> Stream cerrado.")

    if not recording_data:
        print(" -> No se grabó ningún dato.")
        return False # Indicate failure

    # Save the recording to a WAV file
    print(f" -> Guardando grabación en '{filename}'...")
    try:
        final_recording = np.concatenate(recording_data, axis=0)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(np.dtype(DTYPE_REC).itemsize)
            wf.setframerate(samplerate)
            wf.writeframes(final_recording.tobytes())
        print(f" -> Grabación guardada exitosamente en {filename}.")
        return True # Indicate success
    except Exception as e:
        print(f"❌ Error al guardar el archivo WAV: {e}")
        return False

def load_whisper_model(model_name=WHISPER_MODEL):
    """Loads the specified Whisper model, forcing CPU usage."""
    global whisper_model
    if whisper_model:
        print("Whisper model already loaded.")
        return True

    print(f"\n⏳ Cargando modelo Whisper '{model_name}' (forzado a CPU)...")
    # Check if CUDA is available, just to inform the user we're overriding it
    if torch.cuda.is_available():
        print("   (CUDA detectado, pero se forzará el uso de CPU)")
    else:
        print("   (PyTorch configurado solo para CPU)")

    try:
        whisper_model = whisper.load_model(model_name, device='cpu')
        print(f"✅ Modelo Whisper '{model_name}' cargado exitosamente en CPU.")
        return True
    except Exception as e:
        print(f"❌ ERROR al cargar el modelo Whisper '{model_name}': {e}")
        return False

def transcribe_audio(filename=FILENAME_TEMP_RECORDING):
    """Transcribes the specified audio file using the loaded Whisper model."""
    if not whisper_model:
        print("❌ Error: El modelo Whisper no está cargado.")
        return None
    if not os.path.exists(filename):
        print(f"❌ Error: El archivo de audio '{filename}' no existe.")
        return None

    print(f"\n🔄 Transcribiendo audio de '{filename}'...")
    start_time = time.time()
    try:
        # Use fp16=False for CPU inference
        result = whisper_model.transcribe(filename, fp16=False)
        end_time = time.time()
        transcription = result["text"].strip()
        print(f" -> Transcripción completada en {end_time - start_time:.2f} segundos.")
        return transcription
    except Exception as e:
        print(f"❌ ERROR durante la transcripción: {e}")
        return None

# --- TTS Functions (from tts.py) ---

def load_tts_model():
    """Loads the Piper TTS voice model."""
    global tts_voice, tts_sample_rate
    if tts_voice:
        print("TTS model already loaded.")
        return True

    print(f"\n⏳ Cargando modelo TTS desde: {TTS_MODEL_FOLDER}...")
    if not TTS_MODEL_ONNX_PATH.exists() or not TTS_MODEL_JSON_PATH.exists():
        print(f"❌ Error: Archivos del modelo TTS no encontrados.")
        print(f"   - Buscado ONNX: {TTS_MODEL_ONNX_PATH}")
        print(f"   - Buscado JSON: {TTS_MODEL_JSON_PATH}")
        return False

    try:
        tts_voice = PiperVoice.load(str(TTS_MODEL_ONNX_PATH), str(TTS_MODEL_JSON_PATH))
        tts_sample_rate = tts_voice.config.sample_rate
        print(f"✅ Modelo TTS cargado correctamente (Sample Rate: {tts_sample_rate}).")
        return True
    except Exception as e:
        print(f"❌ Error al cargar el modelo TTS: {e}")
        # Attempt to provide more specific feedback for common issues
        if "onnxruntime" in str(e).lower():
             print("   -> Consejo: Asegúrate de que 'onnxruntime' está instalado (`pip install onnxruntime`)")
        return False

def speak_text(text, output_filename=FILENAME_TEMP_TTS_OUTPUT):
    """Synthesizes the given text to speech using Piper and plays it."""
    if not tts_voice:
        print("❌ Error: El modelo TTS no está cargado.")
        return False
    if not text:
        print(" -> No hay texto para decir.")
        return True # Nothing to do, technically success

    print(f"\n🔊 Sintetizando y hablando: \"{text[:60]}...\"") # Print start of text
    try:
        # Synthesize to a WAV file
        with wave.open(output_filename, "wb") as wav_file:
            # Set WAV parameters based on the loaded model's config
            wav_file.setnchannels(1) # Piper models are usually mono
            wav_file.setsampwidth(2) # Standard 16-bit
            wav_file.setframerate(tts_sample_rate)
            # Synthesize directly into the wave file object
            tts_voice.synthesize(text, wav_file)

        print(f" -> Audio guardado temporalmente en '{output_filename}'. Reproduciendo...")
        # Play the generated sound file
        playsound(output_filename)
        return True
    except Exception as e:
        print(f"❌ Error durante la síntesis o reproducción de voz: {e}")
        return False
    finally:
        # Clean up the temporary TTS file
        if os.path.exists(output_filename):
            try:
                os.remove(output_filename)
                # print(f" -> Archivo temporal TTS '{output_filename}' eliminado.")
            except Exception as e_del:
                print(f"⚠️ Advertencia: No se pudo eliminar el archivo TTS temporal '{output_filename}': {e_del}")

# --- LLM Interaction Function ---

def check_ollama_model(model_name_to_check):
    """Verifies if the specified model exists in Ollama."""
    clean_model_name = model_name_to_check.strip()
    print(f"\n🔍 Verificando disponibilidad del modelo Ollama: '{clean_model_name}'...")
    try:
        models_data = ollama.list()
        if 'models' not in models_data or not isinstance(models_data['models'], list):
            print("\n❌ Error: Formato inesperado en la respuesta de 'ollama list'.")
            return False

        # Ollama's list format changed slightly in newer versions
        available_models = [m.get('name', '').strip() for m in models_data['models']]

        # Check for exact match or base name match
        for model_full_name in available_models:
             if clean_model_name == model_full_name or clean_model_name == model_full_name.split(':')[0].strip():
                 print(f"✅ Modelo '{clean_model_name}' encontrado en Ollama.")
                 return True

        print(f"❌ Modelo '{clean_model_name}' NO encontrado en la lista de Ollama.")
        print("   Modelos disponibles:", ", ".join(available_models) or "Ninguno")
        print(f"   Puedes intentar descargarlo con: ollama pull {clean_model_name}")
        return False
    except Exception as e:
        print(f"\n❌ Error al conectar con Ollama o listar modelos: {e}")
        print("   Asegúrate de que Ollama esté instalado y ejecutándose ('ollama serve').")
        return False

# --- Main Simulation Logic ---

def run_interview_simulation():
    global ollama_model_name # Allow modification

    # --- Initial Setup ---
    print("\n--- Simulador de Entrevista Interactivo (Voz) ---")

    # Determine Ollama model to use
    if len(sys.argv) > 1:
        ollama_model_name = sys.argv[1].strip()
        print(f"Intentando usar el modelo Ollama especificado: {ollama_model_name}")
    else:
        ollama_model_name = DEFAULT_OLLAMA_MODEL
        print(f"Usando el modelo Ollama por defecto: {ollama_model_name}")

    # 1. Check Ollama Model Availability
    if not check_ollama_model(ollama_model_name):
        sys.exit(1)

    # 2. Load Whisper STT Model
    if not load_whisper_model():
        print("No se pudo cargar el modelo Whisper. Saliendo.")
        sys.exit(1)

    # 3. Load Piper TTS Model
    if not load_tts_model():
        print("No se pudo cargar el modelo Piper TTS. Saliendo.")
        sys.exit(1)

    print("\n--- Preparación Completa ---")
    print("Instrucciones: Cuando veas '🎤 ¡ESCUCHANDO!', habla tu respuesta.")
    print("La IA escuchará, procesará tu respuesta, te dará feedback y hará la siguiente pregunta.")
    print("Di 'salir' o 'terminar' para finalizar la simulación.")
    print("-" * 40)

    # Initialize conversation history with system prompt
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]

    try:
        # --- Start the Interview ---
        print("\n🤖 Iniciando entrevista...")
        # Get the initial greeting and first question from the LLM
        response = ollama.chat(model=ollama_model_name, messages=messages, stream=False)
        assistant_response = response['message']['content']
        messages.append({'role': 'assistant', 'content': assistant_response})

        print(f"\n🤖 Entrevistador/Coach IA:\n{assistant_response}")
        # Speak the initial message
        if not speak_text(assistant_response):
             print("⚠️ Advertencia: Falló la reproducción de la voz inicial.")


        # --- Main Interview Loop ---
        while True:
            # 1. Record User's Spoken Answer
            if not record_until_silence():
                # Ask user to try again if recording failed
                retry_message = "Hubo un problema con la grabación. ¿Podrías intentar responder de nuevo, por favor?"
                print(f"\n🤖 Entrevistador/Coach IA:\n{retry_message}")
                speak_text(retry_message)
                continue # Skip to the next loop iteration to try recording again

            # 2. Transcribe the Recording
            user_transcription = transcribe_audio(FILENAME_TEMP_RECORDING)

            # Clean up the recording file immediately after transcription attempt
            if os.path.exists(FILENAME_TEMP_RECORDING):
                try:
                    os.remove(FILENAME_TEMP_RECORDING)
                except Exception as e_del_rec:
                     print(f"⚠️ Advertencia: No se pudo eliminar archivo de grabación temporal: {e_del_rec}")


            if user_transcription is None:
                # Ask user to try again if transcription failed
                retry_message = "No pude entender eso claramente. ¿Podrías repetirlo, por favor?"
                print(f"\n🤖 Entrevistador/Coach IA:\n{retry_message}")
                speak_text(retry_message)
                continue # Skip to the next loop iteration

            print(f"\n👤 Tu Respuesta (Transcrita): \"{user_transcription}\"")

            # Check for exit command
            if user_transcription.lower().strip() in ['salir', 'terminar', 'exit', 'quit']:
                farewell_message = "Entendido. Terminando la simulación. ¡Espero que haya sido útil!"
                print(f"\n👋 {farewell_message}")
                speak_text(farewell_message)
                break

            # 3. Send Transcription to LLM
            messages.append({'role': 'user', 'content': user_transcription})

            print("\n🤔 Pensando...") # Give user feedback that processing is happening
            response = ollama.chat(model=ollama_model_name, messages=messages, stream=False)
            assistant_response = response['message']['content']
            messages.append({'role': 'assistant', 'content': assistant_response})

            # 4. Display and Speak LLM's Response (Feedback + Next Question)
            print(f"\n🤖 Entrevistador/Coach IA:\n{assistant_response}")
            if not speak_text(assistant_response):
                 print("⚠️ Advertencia: Falló la reproducción de la respuesta de la IA.")
                 # Optional: Add a fallback or ask the user to read the text


    except Exception as e:
        print(f"\n❌ Ocurrió un error inesperado durante la simulación: {e}")
        print("   Verifica tu conexión con Ollama, los modelos y los permisos.")
    finally:
        # Final cleanup of any lingering temp files (belt and suspenders)
        for temp_file in [FILENAME_TEMP_RECORDING, FILENAME_TEMP_TTS_OUTPUT]:
             if os.path.exists(temp_file):
                 try:
                     os.remove(temp_file)
                 except Exception:
                     pass # Ignore errors during final cleanup
        print("\n--- Simulación Finalizada ---")


# --- Run the Application ---
if __name__ == "__main__":
    run_interview_simulation()