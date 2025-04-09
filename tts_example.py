# Importa la biblioteca
import pyttsx3

# Inicializa el motor TTS
try:
    engine = pyttsx3.init()
    print("Motor TTS inicializado correctamente.")
except Exception as e:
    print(f"Error al inicializar el motor TTS: {e}")
    engine = None

if engine:
    # Texto que quieres convertir a voz
    texto_a_decir = "Hola, este es un ejemplo de uso de pyttsx3 en Python."

    print("Preparando para decir:", texto_a_decir)

    # Pasa el texto al motor
    engine.say(texto_a_decir)

    # Ejecuta el motor para que hable y espera a que termine
    engine.runAndWait()

    print("Terminado.")

    # --- Ejemplo un poco más avanzado: Cambiar propiedades ---

    print("\n--- Ejemplo con propiedades modificadas ---")

    # Obtener y mostrar las propiedades actuales
    try:
        rate = engine.getProperty('rate')   # Velocidad de habla (palabras por minuto)
        volume = engine.getProperty('volume') # Volumen (0.0 a 1.0)
        voices = engine.getProperty('voices') # Lista de voces disponibles

        print(f"Velocidad actual: {rate}")
        print(f"Volumen actual: {volume}")
        print("Voces disponibles:")
        for voice in voices:
            print(f"  - Nombre: {voice.name}, ID: {voice.id}")

        # Cambiar la velocidad (más lento)
        engine.setProperty('rate', 150)

        # Cambiar el volumen (más bajo)
        engine.setProperty('volume', 0.8)

        # Cambiar la voz (si hay más de una disponible)
        # Puedes iterar sobre 'voices' y seleccionar una por su ID
        spanish_voice_id = None
        for voice in voices:
            if voice.name == "Spanish (Spain)":
                spanish_voice_id = voice.id
                break

        if spanish_voice_id:
            engine.setProperty('voice', spanish_voice_id)
            print(f"Usando la voz: Spanish (Spain) ({spanish_voice_id})")
        else:
            print("No se encontró la voz Spanish (Spain). Usando la voz predeterminada.")

        texto_modificado = "Ahora estoy hablando un poco más despacio y con otra voz si estaba disponible."
        print("Preparando para decir:", texto_modificado)

        engine.say(texto_modificado)
        engine.runAndWait()

        print("Terminado el segundo ejemplo.")

        # Es buena práctica detener el motor si ya no se va a usar más en el script,
        # aunque runAndWait() a menudo maneja la limpieza necesaria.
        # engine.stop() # Generalmente no es estrictamente necesario al final del script.

    except Exception as e:
        print(f"Error durante la configuración de propiedades: {e}")