"""Zero-shot voice cloning (pip install cachos[chatterbox]; MIT licensed, 23 languages)."""
from cachos import TTS

tts = TTS(["chatterbox"])
me = tts.clone_voice("my-voice", "reference_10s.wav")   # persisted under ~/.cache/cachos/voices
tts.synthesize_sync("Now I can speak Spanish too.", voice=me.id).save("en.wav")
tts.synthesize_sync("Ahora también puedo hablar español.", voice=me.id).save("es.wav")
