"""cachos – one API for multilingual STT, TTS and real-time voice agents.

    from cachos import STT, TTS, VoiceAgent

    text = STT().transcribe_sync("hola.wav").text             # language auto-detected
    TTS().synthesize_sync("Bonjour tout le monde").save("fr.wav")  # voice auto-picked
    VoiceAgent(llm_model="gpt-4.1-mini").run_local()          # mic + speakers, on Pipecat

``VoiceAgent`` needs the ``agent`` extra (Pipecat + sherpa-onnx) and is imported lazily.
"""

from .audio import Audio
from .core import STT, TTS
from .languages import detect_language, normalize_language
from .registry import available_stt, available_tts, create_stt, create_tts, register_stt, register_tts
from .stt.base import STTEngine
from .text import SentenceStream, split_sentences
from .tts.base import TTSEngine
from .types import Segment, Transcript, Voice, Word

__version__ = "0.1.0"

__all__ = [
    "STT", "TTS", "VoiceAgent", "Audio",
    "Transcript", "Segment", "Word", "Voice", "STTEngine", "TTSEngine",
    "SentenceStream", "split_sentences", "detect_language", "normalize_language",
    "create_stt", "create_tts", "register_stt", "register_tts", "available_stt", "available_tts",
]


def __getattr__(name: str):
    if name == "VoiceAgent":
        from .agent import VoiceAgent

        return VoiceAgent
    raise AttributeError(f"module 'cachos' has no attribute {name!r}")
