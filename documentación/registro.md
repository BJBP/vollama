# Registro de Comandos y Problemas Enfrentados

## Instalación de piper-tts

*   Intento inicial de instalación: `pip install piper-tts`
    *   Problema: Conflicto de dependencias con `piper-phonemize`.
*   Intento de instalación de versión específica: `pip install piper-tts==1.2.0`
    *   Problema: No se encontró la distribución para `piper-phonemize~=1.1.0`.
*   Intento de instalación de versión anterior: `pip install piper-tts==1.1.0`
    *   Problema: No se encontró la distribución para `piper-phonemize~=1.0.0`.
*   Actualización de pip: `pip install --upgrade pip`
*   Reintento de instalación de versión anterior: `pip install piper-tts==1.1.0`
    *   Problema: No se encontró la distribución para `piper-phonemize~=1.0.0`.
*   Intento de instalación manual de dependencia: `pip install piper-phonemize~=1.0.0`
    *   Problema: No se encontró la distribución para `piper-phonemize~=1.0.0`.
*   Verificación de la versión de Python: `python --version`
    *   Resultado: Python 3.12.3 (incompatible con piper-phonemize)
*   Desinstalación de versiones anteriores: `pip uninstall piper-tts piper-phonemize -y`
*   Reintento de instalación de la versión recomendada: `pip install piper-tts==1.2.0`
    *   Problema: No se encontró la distribución para `piper-phonemize~=1.1.0`.
*   Creación de un nuevo entorno virtual con Python 3.11: `python3.11 -m venv venv_py311`
*   Activación del nuevo entorno virtual: `source venv_py311/bin/activate`
*   Actualización de pip en el nuevo entorno virtual: `pip install --upgrade pip`
*   Instalación de piper-tts en el nuevo entorno virtual: `pip install piper-tts`
    *   Éxito: `piper-tts` se instaló correctamente.
*   Intento de instalar dependencias restantes: `pip install sounddevice numpy wave-fix torch openai-whisper ollama playsound`
    *   Problema: No se encontró el paquete `wave-fix`.
*   Instalación de dependencias restantes sin wave-fix: `pip install sounddevice torch openai-whisper ollama playsound`
    *   Éxito: Se instalaron las dependencias restantes.
*   Intento de ejecutar `voice_assistant.py`: `python voice_assistant.py`
    *   Problema: `Current ask promise was ignored`
*   Ejecución de `tts.py` para verificar la instalación de `piper-tts`: `python tts.py`
    *   Éxito: `tts.py` se ejecutó correctamente.
*   Ejecución de `stt.py` para verificar la instalación de `openai-whisper`: `python stt.py`
    *   Éxito: `stt.py` se ejecutó correctamente.
*   Ejecución de `asistente_voz.py`: `python asistente_voz.py`
    *   Problema: Modelo Ollama 'phi3:mini' no encontrado.
*   Descarga del modelo Ollama recomendado: `ollama pull phi3:mini`
    *   Usuario indica usar `phi4-mini:latest`
*   Descarga del modelo Ollama correcto: `ollama pull phi4-mini:latest`
    *   Éxito: Modelo descargado correctamente.
*   Modificación de `asistente_voz.py` para usar `phi4-mini:latest` en la función `check_ollama_model`.
*   Reintento de ejecución de `asistente_voz.py`: `python asistente_voz.py`
    *   Problema: Modelo Ollama 'phi4-mini:latest' no encontrado.
*   Modificación de `asistente_voz.py` para usar `phi4-mini:latest` en la variable `OLLAMA_MODEL`.
*   Reintento de ejecución de `asistente_voz.py`: `python asistente_voz.py`
    *   Problema: Modelo Ollama 'phi4-mini:latest' no encontrado.
*   Comprobación de la lista de modelos de Ollama: `ollama list`
    *   `phi4-mini:latest` aparece en la lista.
*   Comentario de la verificación del modelo LLM en `asistente_voz.py` para evitar el error.
*   Reintento de ejecución de `asistente_voz.py`: `python asistente_voz.py`
    *   Problema: `TypeError: 'module' object is not callable` durante la síntesis y reproducción TTS.
*   Modificación de la función `sintetizar_y_reproducir` en `asistente_voz.py` para pasar el objeto `archivo_wav` correctamente al método `synthesize`.
*   Reintento de ejecución de `asistente_voz.py`: `python asistente_voz.py`
    *   Problema: `'NoneType' object has no attribute 'setframerate'`
*   Modificación de la función `sintetizar_y_reproducir` en `asistente_voz.py` para no iterar sobre el resultado de `voz_tts.synthesize`.
*   Modificación de `asistente_voz.py` para que no reproduzca el audio.
*   Reintento de ejecución de `asistente_voz.py`: `python asistente_voz.py`
    *   Éxito: El script se ejecuta sin errores.
*   Intento de instalar `pygobject`: `pip install pygobject`
    *   Problema: No se encontró la dependencia `girepository-2.0`.
*   Se indica que el error `TypeError: 'module' object is not callable` durante la fase de reproducción con `playsound` se debe a cómo se importa la librería.