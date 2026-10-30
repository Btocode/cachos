"""Text-to-speech engine interface."""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from typing import AsyncIterator, ClassVar, Optional

from ..audio import Audio, AudioInput
from ..languages import normalize_language
from ..text import split_sentences
from ..types import Voice


class TTSEngine(ABC):
    """Base class for TTS engines.

    Implement :meth:`list_voices` and :meth:`_synthesize` (blocking; it runs in a
    worker thread). Engines that are natively async (cloud APIs) override
    :meth:`synthesize` instead. Engines that can stream sub-sentence audio may
    override :meth:`stream`; the default streams sentence by sentence.
    """

    name: ClassVar[str] = "base"
    supports_cloning: ClassVar[bool] = False

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._voices: Optional[list[Voice]] = None
        self._cloned: dict[str, Voice] = {}

    # ---- voices -----------------------------------------------------------
    @abstractmethod
    def list_voices(self) -> list[Voice]:
        """All voices this engine offers (called once, then cached)."""

    def voices(self, language: Optional[str] = None, gender: Optional[str] = None) -> list[Voice]:
        if self._voices is None:
            self._voices = self.list_voices()
        out = [*self._voices, *self._cloned.values()]
        if language:
            out = [v for v in out if v.supports(language)]
        if gender:
            out = [v for v in out if v.gender == gender]
        return out

    def get_voice(self, voice_id: str) -> Optional[Voice]:
        if ":" not in voice_id:
            voice_id = f"{self.name}:{voice_id}"
        return next((v for v in self.voices() if v.id == voice_id), None)

    def default_voice(self, language: Optional[str] = None, gender: Optional[str] = None) -> Optional[Voice]:
        candidates = self.voices(language, gender) or (self.voices(language) if gender else [])
        if not candidates:
            return None
        # prefer native speakers of the language over multilingual ("*") voices,
        # then higher quality, then first-listed (engines list their best voice first)
        rank = {"high": 0, "medium": 1, "low": 2, "x_low": 3, None: 1}
        lang = normalize_language(language) if language else None
        return sorted(candidates, key=lambda v: (0 if lang is None or v.languages[0] == lang else 1,
                                                 rank.get(v.quality, 1)))[0]

    @property
    def languages(self) -> list[str]:
        langs: set[str] = set()
        for v in self.voices():
            langs.update(v.languages)
        return sorted(langs)

    # ---- synthesis --------------------------------------------------------
    @abstractmethod
    def _synthesize(self, text: str, voice: Voice, language: str, speed: float, **options) -> Audio:
        """Blocking synthesis of a short text (one or a few sentences)."""

    def resolve(self, voice: Optional[str | Voice], language: Optional[str], gender: Optional[str] = None) -> tuple[Voice, str]:
        """Pick the concrete voice + language for a request."""
        lang = normalize_language(language) if language and language != "auto" else None
        if isinstance(voice, Voice):
            v = voice
        elif voice:
            v = self.get_voice(voice)
            if v is None:
                raise ValueError(f"unknown voice {voice!r} for engine {self.name!r}")
        else:
            v = self.default_voice(lang, gender) or self.default_voice(None, gender)
            if v is None:
                raise ValueError(f"engine {self.name!r} has no voices")
        if lang is None:
            lang = next((l for l in v.languages if l != "*"), "en")
        return v, lang

    async def synthesize(
        self,
        text: str,
        voice: Optional[str | Voice] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        **options,
    ) -> Audio:
        v, lang = self.resolve(voice, language, options.pop("gender", None))

        def run() -> Audio:
            with self._lock:
                return self._synthesize(text, v, lang, speed, **options)

        return await asyncio.to_thread(run)

    def synthesize_sync(self, text: str, voice: Optional[str | Voice] = None, language: Optional[str] = None,
                        speed: float = 1.0, **options) -> Audio:
        v, lang = self.resolve(voice, language, options.pop("gender", None))
        with self._lock:
            return self._synthesize(text, v, lang, speed, **options)

    async def stream(
        self,
        text: str,
        voice: Optional[str | Voice] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        **options,
    ) -> AsyncIterator[Audio]:
        """Yield audio sentence by sentence so playback can start immediately."""
        for sentence in split_sentences(text):
            yield await self.synthesize(sentence, voice, language, speed, **options)

    # ---- cloning ----------------------------------------------------------
    def clone_voice(self, name: str, reference: AudioInput, language: Optional[str] = None,
                    transcript: Optional[str] = None) -> Voice:
        """Register a voice cloned from a short reference clip (engines with ``supports_cloning``)."""
        raise NotImplementedError(f"engine {self.name!r} does not support voice cloning")

    async def warmup(self) -> None:
        v = self.default_voice()
        if v is not None:
            await self.synthesize("Hello.", v)

    def close(self) -> None:  # pragma: no cover - optional
        pass
