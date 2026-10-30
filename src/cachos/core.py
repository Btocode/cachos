"""High-level facades: :class:`STT` and :class:`TTS`.

These are what most users touch::

    stt = STT()                                  # faster-whisper large-v3, auto language
    tts = TTS()                                  # kokoro -> piper -> edge, auto voice per language
    t = await stt.transcribe("call.mp3")
    audio = await tts.synthesize(reply_text)     # language detected, voice picked
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import AsyncIterable, AsyncIterator, Iterable, Optional, Sequence, Union

from .audio import Audio, AudioInput
from .languages import detect_language, normalize_language
from .registry import create_stt, create_tts, parse_spec
from .stt.base import STTEngine
from .text import SentenceStream
from .tts.base import TTSEngine
from .types import Transcript, Voice

log = logging.getLogger("cachos")

EngineSpec = Union[str, STTEngine]
TTSSpec = Union[str, TTSEngine]

#: default engine priority for TTS(): best local quality first, widest coverage last
DEFAULT_TTS_ENGINES = ["kokoro", "piper", "edge"]


class STT:
    """Speech-to-text with automatic language detection and optional fallbacks.

    Args:
        engine: spec like ``"faster-whisper:large-v3-turbo"``, ``"openai:whisper-1"``,
            ``"groq:whisper-large-v3"`` or an :class:`STTEngine` instance.
        fallback: engines tried in order if the primary raises.
        **options: forwarded to the primary engine constructor.
    """

    def __init__(self, engine: EngineSpec = "auto", *, fallback: Sequence[EngineSpec] = (), **options) -> None:
        self.engine = create_stt(engine, **options)
        self._fallback_specs = list(fallback)
        self._fallbacks: list[STTEngine] = []

    @property
    def name(self) -> str:
        return self.engine.name

    def _chain(self) -> Iterable[STTEngine]:
        yield self.engine
        while len(self._fallbacks) < len(self._fallback_specs):
            self._fallbacks.append(create_stt(self._fallback_specs[len(self._fallbacks)]))
        yield from self._fallbacks

    async def transcribe(self, audio: AudioInput, language: Optional[str] = None, **options) -> Transcript:
        """Transcribe a file path, bytes, ndarray or :class:`Audio`. ``language=None`` auto-detects."""
        last: Optional[Exception] = None
        for eng in self._chain():
            try:
                return await eng.transcribe(audio, language, **options)
            except Exception as e:  # try the next engine
                log.warning("STT engine %s failed: %s", eng.name, e)
                last = e
        raise RuntimeError(f"all STT engines failed: {last}") from last

    def transcribe_sync(self, audio: AudioInput, language: Optional[str] = None, **options) -> Transcript:
        return _run(self.transcribe(audio, language, **options))

    async def warmup(self) -> None:
        await self.engine.warmup()


class TTS:
    """Multi-engine, multilingual text-to-speech.

    Voices from every engine are merged into one catalog. Give a ``voice`` id
    (``"kokoro:af_heart"``, ``"edge:hi-IN-SwaraNeural"``, ``"piper:de_DE-thorsten-high"``)
    or let cachos pick: the language is detected from the text and the first
    engine (in priority order) with a voice for it is used.

    Args:
        engines: engine specs in priority order, or ``"auto"`` for
            ``kokoro -> piper -> edge`` (engines whose dependencies are missing
            are skipped with a warning).
        voices: per-language voice overrides, e.g. ``{"en": "kokoro:am_michael", "es": "edge:es-MX-JorgeNeural"}``.
        gender: preferred gender when auto-picking (``"female"``/``"male"``).
        allow_cloud: if False, network engines (edge, openai) are excluded.
    """

    def __init__(
        self,
        engines: Union[str, TTSSpec, Sequence[TTSSpec]] = "auto",
        *,
        voices: Optional[dict[str, str]] = None,
        gender: Optional[str] = None,
        allow_cloud: bool = True,
        engine_options: Optional[dict[str, dict]] = None,
    ) -> None:
        engine_options = engine_options or {}
        strict = engines != "auto"
        if engines == "auto":
            specs: list[TTSSpec] = list(DEFAULT_TTS_ENGINES)
        elif isinstance(engines, (str, TTSEngine)):
            specs = [engines]
        else:
            specs = list(engines)
        self.engines: list[TTSEngine] = []
        for spec in specs:
            name = spec.name if isinstance(spec, TTSEngine) else parse_spec(spec)[0]
            if not allow_cloud and name in ("edge", "openai"):
                continue
            try:
                self.engines.append(create_tts(spec, **engine_options.get(name, {})))
            except ImportError as e:
                if strict:
                    raise
                warnings.warn(f"cachos: skipping TTS engine {name!r} ({e}). Install it with `pip install cachos[{name}]`.")
        if not self.engines:
            raise RuntimeError("no TTS engine available; try `pip install cachos[kokoro,edge]`")
        self.voice_map = {normalize_language(k): v for k, v in (voices or {}).items()}
        self.gender = gender

    # ---- catalog ------------------------------------------------------------
    def engine(self, name: str) -> TTSEngine:
        for e in self.engines:
            if e.name == name:
                return e
        raise KeyError(f"TTS engine {name!r} not loaded (loaded: {[e.name for e in self.engines]})")

    def voices(self, language: Optional[str] = None, gender: Optional[str] = None,
               engine: Optional[str] = None) -> list[Voice]:
        out: list[Voice] = []
        for e in self.engines:
            if engine and e.name != engine:
                continue
            try:
                out.extend(e.voices(language, gender))
            except Exception as ex:
                log.warning("could not list voices for %s: %s", e.name, ex)
        return out

    @property
    def languages(self) -> list[str]:
        langs: set[str] = set()
        for e in self.engines:
            try:
                langs.update(e.languages)
            except Exception:
                pass
        langs.discard("*")
        return sorted(langs)

    def get_voice(self, voice_id: str) -> Voice:
        name, _, _ = voice_id.partition(":")
        v = self.engine(name).get_voice(voice_id)
        if v is None:
            raise ValueError(f"unknown voice {voice_id!r}")
        return v

    def pick_voice(self, language: str, gender: Optional[str] = None) -> Voice:
        """Best voice for ``language``: explicit override, then engine priority order."""
        lang = normalize_language(language) or "en"
        if lang in self.voice_map:
            return self.get_voice(self.voice_map[lang])
        g = gender or self.gender
        for e in self.engines:
            v = e.default_voice(lang, g)
            if v is not None:
                return v
        raise ValueError(f"no voice available for language {lang!r}; loaded engines: "
                         f"{[e.name for e in self.engines]} (edge-tts covers 70+ languages, mms 1000+)")

    def resolve(self, text: str, voice: Optional[Union[str, Voice]] = None, language: Optional[str] = None,
                gender: Optional[str] = None) -> tuple[TTSEngine, Voice, str]:
        if isinstance(voice, str) and voice.startswith("auto"):
            voice = None
        if voice is not None:
            v = voice if isinstance(voice, Voice) else self.get_voice(voice)
            eng = self.engine(v.engine)
            lang = normalize_language(language) if language and language != "auto" else None
            if lang is None:
                supported = [l for l in v.languages if l != "*"]
                lang = (detect_language(text, supported) if len(supported) > 1 else None) or (supported or ["en"])[0]
            return eng, v, lang
        lang = normalize_language(language) if language and language != "auto" else None
        lang = lang or detect_language(text) or "en"
        v = self.pick_voice(lang, gender)
        return self.engine(v.engine), v, lang

    # ---- synthesis ----------------------------------------------------------
    async def synthesize(self, text: str, voice: Optional[Union[str, Voice]] = None, language: Optional[str] = None,
                         *, gender: Optional[str] = None, speed: float = 1.0, **options) -> Audio:
        """Synthesize ``text`` in full. Long text is split into sentences and joined."""
        parts = [a async for a in self.stream(text, voice, language, gender=gender, speed=speed, **options)]
        sr = parts[0].sample_rate if parts else 24000
        return Audio.concat(parts, sr)

    def synthesize_sync(self, text: str, voice: Optional[Union[str, Voice]] = None,
                        language: Optional[str] = None, **kw) -> Audio:
        return _run(self.synthesize(text, voice, language, **kw))

    async def stream(
        self,
        text: Union[str, Iterable[str], AsyncIterable[str]],
        voice: Optional[Union[str, Voice]] = None,
        language: Optional[str] = None,
        *,
        gender: Optional[str] = None,
        speed: float = 1.0,
        per_sentence_language: bool = False,
        prefetch: int = 2,
        **options,
    ) -> AsyncIterator[Audio]:
        """Stream audio for a string or a (async) token stream, e.g. straight from an LLM.

        Sentences are synthesized up to ``prefetch`` ahead of playback. With
        ``per_sentence_language`` each sentence is language-detected on its own
        (for code-switching replies); otherwise the first sentence decides.
        """
        chunks = _chunk_text(text)
        queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, prefetch))
        state: dict = {}

        async def produce() -> None:
            try:
                async for sentence in chunks:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if "voice" not in state or per_sentence_language:
                        eng, v, lang = self.resolve(sentence, voice, language, gender)
                        state.update(engine=eng, voice=v, lang=lang)
                    task = asyncio.ensure_future(
                        state["engine"].synthesize(sentence, state["voice"], state["lang"], speed, **options))
                    await queue.put(task)
            except Exception as e:
                await queue.put(e)
            finally:
                await queue.put(None)

        producer = asyncio.create_task(produce())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield await item
        finally:
            # on early exit (e.g. barge-in) stop pending synthesis instead of wasting compute
            producer.cancel()
            while not queue.empty():
                pending = queue.get_nowait()
                if isinstance(pending, asyncio.Future):
                    pending.cancel()

    def clone_voice(self, name: str, reference: AudioInput, *, engine: Optional[str] = None,
                    language: Optional[str] = None, transcript: Optional[str] = None) -> Voice:
        """Clone a voice from a 5-30 s reference clip with the first engine that supports cloning."""
        for e in self.engines:
            if (engine is None and e.supports_cloning) or e.name == engine:
                return e.clone_voice(name, reference, language=language, transcript=transcript)
        raise RuntimeError("no loaded TTS engine supports cloning; add 'chatterbox' or 'xtts' to engines")

    async def warmup(self) -> None:
        await self.engines[0].warmup()


async def _chunk_text(text: Union[str, Iterable[str], AsyncIterable[str]]) -> AsyncIterator[str]:
    chunker = SentenceStream()
    if isinstance(text, str):
        text = [text]
    if hasattr(text, "__aiter__"):
        async for tok in text:  # type: ignore[union-attr]
            for c in chunker.push(tok):
                yield c
    else:
        for tok in text:  # type: ignore[union-attr]
            for c in chunker.push(tok):
                yield c
    for c in chunker.flush():
        yield c


def _run(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("called a *_sync method inside a running event loop; use the async method instead")
