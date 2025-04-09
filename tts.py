# 1. Importa las clases necesarias
from pathlib import Path
from piper.voice import PiperVoice
from playsound import playsound
import wave  # ¡Importa el módulo wave!

# --- Configuración ---
# 2. Define la ruta a la carpeta donde guardaste los archivos del modelo
#    Ajusta esto si tu estructura es diferente (pero la que usaste antes funcionó)
ruta_carpeta_modelos = Path("./modelos/es/es_ES/carlfm/x_low")  # Cambia './modelos' si tu carpeta está en otro lugar


# 3. Especifica los nombres de los archivos del modelo que descargaste
nombre_modelo_onnx = "es_ES-carlfm-x_low.onnx"
nombre_modelo_json = "es_ES-carlfm-x_low.onnx.json"

# 4. Construye las rutas completas a los archivos del modelo
ruta_modelo_onnx = ruta_carpeta_modelos / nombre_modelo_onnx
ruta_modelo_json = ruta_carpeta_modelos / nombre_modelo_json

# 5. El texto que quieres que diga
texto_a_sintetizar = "Hola, ¿cómo estás? Esta es una prueba de voz."

# 6. Nombre del archivo de audio que se generará
nombre_archivo_salida = "hola_salida.wav"
# --- Fin Configuración ---

# 7. Verifica si los archivos del modelo existen antes de continuar
if not ruta_modelo_onnx.exists() or not ruta_modelo_json.exists():
    print(f"Error: No se encontraron los archivos del modelo en la ruta especificada:")
    print(f"- {ruta_modelo_onnx}")
    print(f"- {ruta_modelo_json}")
    print("Por favor, verifica que los archivos existen y la ruta en el script es correcta.")
    exit()  # Termina el script si no se encuentran los modelos

# 8. Carga el modelo de voz Piper
print(f"Cargando el modelo de voz desde: {ruta_carpeta_modelos}...")
try:
    # Crea la instancia de PiperVoice con las rutas a los archivos del modelo
    voz = PiperVoice.load(str(ruta_modelo_onnx), str(ruta_modelo_json))
    print("Modelo cargado correctamente.")
    # --- ¡Importante! Obtén los parámetros de audio del modelo ---
    # Necesitamos saber la tasa de muestreo (framerate), etc., para el archivo WAV
    sample_rate = voz.config.sample_rate
    # -----------------------------------------------------------
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()  # Termina si hay error al cargar

# 9. Sintetiza el texto a un archivo de audio WAV (¡USANDO EL MÓDULO wave!)
print(f"Sintetizando el texto: '{texto_a_sintetizar}'...")
try:
    # Abre el archivo WAV usando el módulo wave en modo escritura ('wb')
    with wave.open(nombre_archivo_salida, "wb") as archivo_wav:
        # --- Establece los parámetros del archivo WAV ---
        archivo_wav.setnchannels(1)  # Número de canales
        archivo_wav.setsampwidth(2)   # Ancho de muestra en bytes (ej. 2 para 16-bit)
        archivo_wav.setframerate(sample_rate)  # Tasa de muestreo (fotogramas/segundo)
        # --------------------------------------------------

        # Llama al método synthesize para generar el audio y escribirlo
        # en el objeto wave (que sabe cómo manejar los frames)
        voz.synthesize(texto_a_sintetizar, archivo_wav)  # Pasar el objeto wave

    print(f"¡Éxito! El audio se ha guardado como '{nombre_archivo_salida}'")
    print("Puedes reproducir este archivo .wav con cualquier reproductor de audio.")
    playsound(nombre_archivo_salida)
except Exception as e:
    print(f"Error durante la síntesis de voz: {e}")

# --- Opcional: Reproducción directa ---
# (El código opcional para voz.say() no necesita cambios, ya que
# maneja la reproducción internamente si tienes las dependencias)
# ... (código opcional sin cambios) ...