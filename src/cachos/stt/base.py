"""Speech-to-text engine interface."""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from ..audio import Audio, AudioInput
from ..types import Transcript


class STTEngine(ABC):
    """Base class for STT engines.

    Implement :meth:`_transcribe` (blocking; it runs in a worker thread). Live
    audio is segmented by Pipecat's Silero VAD (see :mod:`cachos.pipecat`), so
    engines only ever see complete utterances.
    """

    name: ClassVar[str] = "base"
    #: sample rate the engine wants; input audio is resampled to it
    sample_rate: ClassVar[int] = 16000
    #: ISO 639-1 codes, or ["*"] for "anything"
    languages: ClassVar[list[str]] = ["*"]
    #: True if the engine detects the spoken language by itself
    auto_language: ClassVar[bool] = True

    def __init__(self) -> None:
        # Most model runtimes are not re-entrant; serialize calls per instance.
        self._lock = threading.Lock()

    @abstractmethod
    def _transcribe(self, audio: Audio, language: Optional[str], **options) -> Transcript:
        """Blocking transcription of a complete utterance."""

    async def transcribe(self, audio: AudioInput, language: Optional[str] = None, **options) -> Transcript:
        """Transcribe a whole file/buffer. ``language=None`` means auto-detect."""
        pcm = Audio.load(audio, self.sample_rate)

        def run() -> Transcript:
            with self._lock:
                t = self._transcribe(pcm, _lang_or_none(language), **options)
            t.engine = t.engine or self.name
            if t.duration is None:
                t.duration = pcm.duration
            return t

        return await asyncio.to_thread(run)

    def transcribe_sync(self, audio: AudioInput, language: Optional[str] = None, **options) -> Transcript:
        pcm = Audio.load(audio, self.sample_rate)
        with self._lock:
            t = self._transcribe(pcm, _lang_or_none(language), **options)
        t.engine = t.engine or self.name
        t.duration = t.duration if t.duration is not None else pcm.duration
        return t

    async def warmup(self) -> None:
        """Load weights / run a dummy pass so the first real request is fast."""
        await self.transcribe(Audio.silence(0.5, self.sample_rate))

    def close(self) -> None:  # pragma: no cover - optional
        pass


def _lang_or_none(language: Optional[str]) -> Optional[str]:
    from ..languages import normalize_language

    lang = normalize_language(language)
    return None if lang in (None, "auto") else lang
