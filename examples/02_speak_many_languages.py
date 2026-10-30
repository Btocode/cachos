"""One TTS object, many languages: the voice is picked from the text's language."""
from cachos import TTS

tts = TTS()  # kokoro -> piper -> edge (whatever is installed)
lines = [
    "Hello! Welcome to our store.",
    "¡Hola! Bienvenido a nuestra tienda.",
    "Bonjour ! Bienvenue dans notre magasin.",
    "नमस्ते! हमारी दुकान में आपका स्वागत है।",
    "こんにちは！私たちの店へようこそ。",
    "Willkommen in unserem Geschäft!",
    "আমাদের দোকানে স্বাগতম!",
]
for i, line in enumerate(lines):
    audio = tts.synthesize_sync(line)
    audio.save(f"hello_{i}.wav")
    print(f"hello_{i}.wav  {audio.duration:4.1f}s")

# Explicit voices / filters
print([v.id for v in tts.voices(language="fr", gender="male")][:5])
tts.synthesize_sync("Je suis une voix masculine.", voice=tts.voices(language="fr", gender="male")[0]).save("fr_male.wav")
