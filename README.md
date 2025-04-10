# Vollama

Este proyecto combina la funcionalidad de Text-to-Speech (TTS) utilizando Piper TTS, Speech-to-Text (STT) utilizando OpenAI Whisper, y un modelo de lenguaje local con Ollama para crear un asistente de voz conversacional interactivo.

## Prerequisites

*   Python 3.11
*   pip
*   git

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd tts
    ```

2.  Create a virtual environment:

    ```bash
    python3.11 -m venv venv-py311
    source venv-py311/bin/activate
    ```

3.  Install the dependencies:

    ```bash
    pip install piper-tts==1.2.0
    ```

3. Install playsound:

    ```bash
    pip install playsound
    ```

3. Install playsound:

    ```bash
    pip install playsound
    ```

4.  Download the model files:

    ```bash
    pip install "huggingface-hub[cli]"
    ```

    Then, download the model files:

    ```bash
    cd modelos
    huggingface-cli download rhasspy/piper-voices es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx --local-dir . --local-dir-use-symlinks False
    huggingface-cli download rhasspy/piper-voices es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx.json --local-dir . --local-dir-use-symlinks False
    cd ..
    ```

## Usage

1.  Run the script:

    ```bash
    python tts.py
    ```

The script will automatically play the generated audio file.

## Troubleshooting

*   If you encounter a `404 Not Found` error when downloading the model files, make sure you have the `huggingface-cli` installed and that you are using the correct paths to the model files.
*   If you encounter an error related to `sample_width` or `num_channels`, make sure you are using the correct version of the `tts.py` script.

## Repository Structure

```
.
├── tts.py
├── modelos
│   ├── es
│   │   └── es_ES
│   │       └── carlfm
│   │           └── x_low
│   │               ├── es_ES-carlfm-x_low.onnx
│   │               └── es_ES-carlfm-x_low.onnx.json
├── venv-py311
└── README.md

## Estructura del Proyecto

*   `tts.py`: Script para la funcionalidad de Text-to-Speech (TTS) utilizando Piper TTS.
*   `stt.py`: Script para la funcionalidad de Speech-to-Text (STT) utilizando OpenAI Whisper.
*   `modelos/`: Directorio que contiene los modelos de voz para Piper TTS.
*   `venv-py311/`: Entorno virtual de Python.
*   `sensibilidad.py`: Script para ajustar la sensibilidad del micrófono para la funcionalidad STT.
*   `llm.py`: Script que implementa un simulador de entrevistas de trabajo interactivo por voz.
*   `asistente_voz.py`: Script que integra las funcionalidades de TTS, STT y LLM para crear un asistente de voz conversacional.
*   `decir_hola.py`: Script de ejemplo simple para probar la funcionalidad TTS.
*   `README.md`: Archivo README del proyecto.
*   `documentación/`: Directorio que contiene la documentación del proyecto.

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