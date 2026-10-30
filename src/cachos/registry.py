"""Engine registry.

Engines are addressed by spec strings: ``"<engine>"`` or ``"<engine>:<model>"``,
e.g. ``"faster-whisper:large-v3-turbo"``, ``"kokoro"``, ``"openai:gpt-4o-mini-tts"``.
Built-ins are imported lazily so optional dependencies are only needed for the
engines you actually use. Third-party packages can register engines via the
``cachos.engines`` entry-point group (the target is called with no arguments
and should call :func:`register_stt` / :func:`register_tts`).
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from typing import Any, Callable, Union

from .stt.base import STTEngine
from .tts.base import TTSEngine

Factory = Union[str, Callable[..., Any]]  # "module:Class" or a callable

_STT: dict[str, Factory] = {
    "faster-whisper": "cachos.stt.faster_whisper:FasterWhisperSTT",
    "whisper": "cachos.stt.faster_whisper:FasterWhisperSTT",  # alias
    "openai": "cachos.stt.openai_compat:OpenAISTT",  # OpenAI, Groq, any OpenAI-compatible server
    "groq": "cachos.stt.openai_compat:GroqSTT",
}
_TTS: dict[str, Factory] = {
    "kokoro": "cachos.tts.kokoro:KokoroTTS",
    "piper": "cachos.tts.piper:PiperTTS",
    "edge": "cachos.tts.edge:EdgeTTS",
    "xtts": "cachos.tts.xtts:XTTS",
    "chatterbox": "cachos.tts.chatterbox:ChatterboxTTS",
    "mms": "cachos.tts.mms:MMSTTS",
    "openai": "cachos.tts.openai_compat:OpenAITTS",
}
_plugins_loaded = False

# "auto" is accuracy-first: on our 6-language check (en/es/hi/bn/de/ja) large-v3 detected 6/6
# languages with 7% mean error; large-v3-turbo 5/6 (Bengali -> Gujarati/Hindi) at ~25% less
# time. "fast" = turbo, which escalates uncertain utterances to large-v3.
STT_ALIASES = {
    "auto": "faster-whisper:large-v3",
    "default": "faster-whisper:large-v3",
    "accurate": "faster-whisper:large-v3",
    "fast": "faster-whisper:large-v3-turbo",
}


def register_stt(name: str, factory: Factory) -> None:
    _STT[name] = factory


def register_tts(name: str, factory: Factory) -> None:
    _TTS[name] = factory


def available_stt() -> list[str]:
    _load_plugins()
    return sorted(_STT)


def available_tts() -> list[str]:
    _load_plugins()
    return sorted(_TTS)


def parse_spec(spec: str) -> tuple[str, str | None]:
    name, _, model = spec.partition(":")
    return name.strip().lower(), (model.strip() or None)


def _resolve(factory: Factory) -> Callable[..., Any]:
    if isinstance(factory, str):
        module, _, attr = factory.partition(":")
        return getattr(importlib.import_module(module), attr)
    return factory


def _load_plugins() -> None:
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    try:
        for ep in entry_points(group="cachos.engines"):
            try:
                obj = ep.load()
                # an engine class registers under the entry-point name; anything else is a
                # register() hook that calls register_stt/register_tts itself
                if isinstance(obj, type) and issubclass(obj, TTSEngine):
                    _TTS.setdefault(ep.name, obj)
                elif isinstance(obj, type) and issubclass(obj, STTEngine):
                    _STT.setdefault(ep.name, obj)
                else:
                    obj()
            except Exception as e:  # never let a broken plugin break cachos
                import warnings

                warnings.warn(f"cachos: failed to load engine plugin {ep.name!r}: {e}")
    except Exception:
        pass


def create_stt(spec: str | STTEngine, **options: Any) -> STTEngine:
    if isinstance(spec, STTEngine):
        return spec
    _load_plugins()
    spec = STT_ALIASES.get(spec, spec)
    name, model = parse_spec(spec)
    if name not in _STT:
        raise ValueError(f"unknown STT engine {name!r}; available: {', '.join(available_stt())}")
    if model:
        options.setdefault("model", model)
    return _resolve(_STT[name])(**options)


def create_tts(spec: str | TTSEngine, **options: Any) -> TTSEngine:
    if isinstance(spec, TTSEngine):
        return spec
    _load_plugins()
    name, model = parse_spec(spec)
    if name not in _TTS:
        raise ValueError(f"unknown TTS engine {name!r}; available: {', '.join(available_tts())}")
    if model:
        options.setdefault("model", model)
    return _resolve(_TTS[name])(**options)
