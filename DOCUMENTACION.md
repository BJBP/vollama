# Documentación del Proyecto TTS con STT

Este proyecto combina la funcionalidad de Text-to-Speech (TTS) utilizando Piper TTS con la funcionalidad de Speech-to-Text (STT) utilizando OpenAI Whisper.

## Estructura del Proyecto

*   `tts.py`: Script principal para la funcionalidad de Text-to-Speech (TTS).
*   `stt.py`: Script principal para la funcionalidad de Speech-to-Text (STT).
*   `modelos/`: Directorio que contiene los modelos de voz para Piper TTS.
*   `venv-py311/`: Entorno virtual de Python.
*   `README.md`: Archivo README del proyecto.
*   `DOCUMENTACION.md`: Este archivo de documentación.

## Comandos de Instalación

```bash
# Clonar el repositorio
git clone <repository_url>
cd tts

# Crear entorno virtual (opcional)
python3 -m venv venv-py311
source venv-py311/bin/activate  # Linux/macOS
# venv-py311\Scripts\activate.bat  # Windows (CMD)
# venv-py311\Scripts\Activate.ps1  # Windows (PowerShell)

# Instalar dependencias TTS
pip install piper-tts==1.2.0 playsound

# Instalar dependencias STT (CPU)
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -U openai-whisper sounddevice numpy

# Instalar ffmpeg (si no está instalado)
# Linux (Ubuntu/Debian): sudo apt update && sudo apt install ffmpeg
# macOS (con Homebrew): brew install ffmpeg
# Windows: (Ver instrucciones en la guía)
```

## Funcionalidad TTS (Text-to-Speech)

El script `tts.py` utiliza la biblioteca Piper TTS para generar voz a partir de texto.

### Dependencias

*   `piper-tts`
*   `playsound`
*   `wave`
*   `pathlib`

### Configuración

1.  Asegúrate de tener los archivos del modelo de voz en el directorio `modelos/`.
2.  Modifica la ruta a la carpeta de modelos en el script `tts.py` si es necesario.
3.  Especifica el texto que quieres sintetizar en la variable `texto_a_sintetizar`.
4.  Especifica el nombre del archivo de audio de salida en la variable `nombre_archivo_salida`.

### Uso

Ejecuta el script `tts.py` para generar el archivo de audio WAV a partir del texto especificado.

## Funcionalidad STT (Speech-to-Text)

El script `stt.py` utiliza la biblioteca OpenAI Whisper para transcribir audio a texto.

### Dependencias

*   `openai-whisper`
*   `sounddevice`
*   `numpy`
*   `torch` (versión para CPU)
*   `wave`

### Configuración

1.  Asegúrate de tener instalada la versión de PyTorch para CPU.
2.  Especifica el modelo de Whisper a utilizar en la variable `MODEL_NAME`.
3.  Ajusta los parámetros de detección de silencio `SILENCE_THRESHOLD` y `SILENCE_DURATION` para que se adapten a tu entorno.

### Uso

Ejecuta el script `stt.py`. El script grabará audio desde el micrófono hasta que se detecte un período de silencio. Luego, transcribirá el audio y mostrará el texto resultante.

```bash
python stt.py
```

### Ajuste de Sensibilidad

Para ajustar la sensibilidad del micrófono, puedes ejecutar el script `sensibilidad.py`. Este script mostrará el nivel de RMS en tiempo real, lo que te permitirá ajustar el valor de `SILENCE_THRESHOLD` en el script `stt.py` para que se adapte a tu entorno.

```bash
python sensibilidad.py
```

## Instalación

Sigue los pasos en el archivo `README.md` para instalar las dependencias del proyecto.

## Notas Adicionales

*   Asegúrate de tener `ffmpeg` instalado y en el PATH para que Whisper pueda procesar los archivos de audio.
*   Si tienes problemas con `sounddevice`, asegúrate de tener las librerías de audio del sistema instaladas.