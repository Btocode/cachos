"""cachos ⟷ Pipecat: services and processors for Pipecat pipelines.

Pipecat (BSD-2) is the real-time runtime: transports (local audio, WebRTC,
telephony WebSockets), Silero VAD, Smart Turn v3 end-of-turn detection,
interruptions, context aggregation. cachos adds what Pipecat lacks:

* :class:`CachosSTTService` – any cachos STT engine with the *detected
  language propagated* on every ``TranscriptionFrame`` (Pipecat's own
  ``WhisperSTTService`` reports only the configured language), Whisper
  anti-hallucination presets, ``language_candidates`` and engine fallbacks.
* :class:`CachosTTSService` – the multi-engine voice catalog with a voice
  picked per reply language (detected from the reply text, falling back to
  the language the user spoke), stable within one reply.
* :class:`DenoiseForVAD` / :class:`DenoiseFilter` – GTCRN/DPDFNet via
  sherpa-onnx. ``DenoiseForVAD`` sits *after* STT so VAD and Smart Turn hear
  clean audio while STT hears the original (evidence: denoising before
  Whisper often raises WER; see docs/RESEARCH.md §8).
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import AsyncGenerator, AsyncIterator, Optional

import numpy as np
from loguru import logger
from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    FilterControlFrame,
    FilterEnableFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
    SystemFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.utils.time import time_now_iso8601

from .audio import Audio
from .core import STT, TTS
from .denoise import StreamingDenoiser
from .languages import detect_language, normalize_language


@dataclass
class LanguageTracker:
    """Shared between STT and TTS services: the language the user last spoke."""

    language: Optional[str] = None
    probability: float = 0.0


def _to_pipecat_language(code: Optional[str]) -> Optional[Language]:
    if not code:
        return None
    try:
        return Language(code)
    except ValueError:
        return None


def _latency_estimate(latencies) -> float:
    """p95 of recent speech-end → transcript times, plus margin (0 when unknown)."""
    return float(np.percentile(latencies, 95)) * 1.25 + 0.1 if latencies else 0.0


class CachosSTTService(SegmentedSTTService):
    """Pipecat STT service backed by :class:`cachos.STT` (VAD-segmented).

    Pipecat's end-of-turn strategies wait up to the STT's ``ttfs_p99_latency``
    for the final transcript; with a fixed guess, a slower device (CPU Whisper)
    ends turns *empty* and the transcript lands in the next turn. Unless a
    value is given, this service measures its real speech-end → transcript
    latency and re-broadcasts the estimate (p95 of recent turns plus margin)
    whenever it grows, so turn detection adapts to the hardware.
    """

    def __init__(self, stt: Optional[STT] = None, *, language: Optional[str] = None,
                 tracker: Optional[LanguageTracker] = None, **kwargs) -> None:
        # Pipecat validates that settings are complete at start; model/language are ours
        kwargs.setdefault("settings", STTSettings(model=None, language=None))
        stt = stt or STT()
        # latency history lives on the (shared) STT, so new sessions start from what was learned
        if not hasattr(stt, "latencies"):
            stt.latencies = deque(maxlen=50)
        self._adaptive_ttfs = kwargs.get("ttfs_p99_latency") is None
        if self._adaptive_ttfs:
            kwargs["ttfs_p99_latency"] = max(1.0, _latency_estimate(stt.latencies))
        super().__init__(**kwargs)
        self.cachos_stt = stt
        self.fixed_language = language
        self.tracker = tracker or LanguageTracker()
        self._latencies = stt.latencies
        self._segment_ended: dict[int, float] = {}  # segment number -> when its speech ended
        self._turn_text: list[str] = []  # transcripts of the current user turn's segments
        self._turn_language: Optional[str] = None

    async def _observe_latency(self, seconds: float) -> None:
        self._latencies.append(seconds)
        if not self._adaptive_ttfs:
            return
        estimate = _latency_estimate(self._latencies)
        if estimate > self._ttfs_p99_latency:
            self._ttfs_p99_latency = estimate
            logger.info(f"{self}: transcript latency p95 {estimate:.2f}s; turn detection now waits for it")
            await self.broadcast_service_metadata()

    @property
    def wants_wav_segments(self) -> bool:
        return False  # we read raw PCM16

    def can_generate_metrics(self) -> bool:
        return True

    #: segments shorter than this reuse the language detected earlier in the same turn
    #: (Whisper's language ID is unreliable on short clips)
    short_segment_secs = 2.5

    async def _segment_task_handler(self):
        """Pipecat's handler transcribes queued VAD segments one by one, so when the STT is
        slower than speech (CPU, long turns) the waits add up and the turn closes before the
        last transcript. We merge every segment already waiting into one clip: one STT call
        instead of N, with the whole sentence as context. Mirrors
        ``SegmentedSTTService._segment_task_handler`` (pipecat 1.15)."""
        running = True
        while running:
            segment = await self._segment_queue.get()
            running = segment is not None
            if not segment:
                break
            audio, number = segment
            while not self._segment_queue.empty():
                nxt = self._segment_queue.get_nowait()
                if nxt is None:
                    running = False
                    break
                audio, number = audio + nxt[0], nxt[1]
            self._transcribing_segment = number  # lets Pipecat mark the transcript finalized
            try:
                await self.process_generator(self.run_stt(audio))
            except Exception as e:
                await self.push_error(f"{self}: transcription failed: {e}", exception=e)
            # latency as Pipecat defines it: end of speech -> transcript, queue wait included
            ended = self._segment_ended.pop(number, None)
            for k in [k for k in self._segment_ended if k < number]:
                del self._segment_ended[k]
            if ended is not None:
                await self._observe_latency(time.monotonic() - ended)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._segment_ended[self._segment_count] = time.monotonic()
        elif isinstance(frame, UserStoppedSpeakingFrame):  # the user's turn is over
            self._turn_text, self._turn_language = [], None

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        # A pause mid-sentence splits one turn into several VAD segments. Transcribing a
        # fragment cold loses context ("…reservieren" -> "rosavieren."), so later segments
        # get the turn's earlier text as Whisper's initial_prompt (the context-carrying
        # technique of whisper_streaming) and, when short, the turn's language.
        turn_text = self._turn_text
        language = self.fixed_language
        options: dict = {}
        if turn_text:
            if self.cachos_stt.name in ("faster-whisper", "whisper"):
                options["initial_prompt"] = " ".join(turn_text)[-400:]
            if language is None and len(audio) / 2 / self.sample_rate < self.short_segment_secs:
                language = self._turn_language
        await self.start_processing_metrics()
        try:
            t = await self.cachos_stt.transcribe(Audio.from_pcm16(audio, self.sample_rate), language, **options)
        except Exception as e:
            yield ErrorFrame(error=f"cachos STT failed: {e}")
            return
        finally:
            await self.stop_processing_metrics()
        text = t.text.strip()
        if not text:
            return
        if t.language:
            self.tracker.language = t.language
            self.tracker.probability = t.language_probability or 0.0
        self._turn_text = [*turn_text, text]
        self._turn_language = self._turn_language or t.language
        logger.debug(f"{self}: [{t.language} {t.language_probability or 0:.2f}] {text}")
        yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), _to_pipecat_language(t.language))


class CachosTTSService(TTSService):
    """Pipecat TTS service backed by the multilingual :class:`cachos.TTS` router.

    Voice selection per reply: an explicit ``voice`` wins; otherwise the reply's
    language is detected from its first sentence (restricted to languages we
    have voices for) and falls back to the user's spoken language for short or
    ambiguous sentences. The voice then stays fixed for the rest of the reply.
    """

    def __init__(self, tts: Optional[TTS] = None, *, voice: Optional[str] = None,
                 tracker: Optional[LanguageTracker] = None, gender: Optional[str] = None, **kwargs) -> None:
        kwargs.setdefault("settings", TTSSettings(model=None, voice=voice, language=None))
        super().__init__(push_start_frame=True, push_stop_frames=True, **kwargs)
        self.cachos_tts = tts or TTS()
        self.voice_id = voice
        self.gender = gender
        self.tracker = tracker or LanguageTracker()
        self._context_voice: dict[str, tuple] = {}

    def can_generate_metrics(self) -> bool:
        return True

    def _pick(self, text: str, context_id: str):
        if context_id in self._context_voice:
            return self._context_voice[context_id]
        user_lang = self.tracker.language
        lang = None
        if not self.voice_id:
            supported = self.cachos_tts.languages or None
            detected = detect_language(text, supported)
            # short sentences ("Okay!", "Sure.") are unreliable: trust what the user spoke
            lang = detected if (detected and (len(text) >= 25 or not user_lang)) else (user_lang or detected)
        eng, voice, lang = self.cachos_tts.resolve(text, self.voice_id, lang, self.gender)
        if len(self._context_voice) > 64:
            self._context_voice.clear()
        self._context_voice[context_id] = (eng, voice, lang)
        logger.debug(f"{self}: reply voice {voice.id} [{lang}] (user spoke {user_lang})")
        return eng, voice, lang

    async def _synthesize_with_fallback(self, text: str, eng, voice, lang: str, context_id: str) -> Audio:
        """Synthesize with the reply's voice; if that engine fails or returns silence
        (e.g. a network engine is down), use the next engine's voice for the language
        and keep it for the rest of the reply. Pipecat otherwise marks the TTS unusable
        after a few empty contexts."""
        tried = []
        candidates = [(eng, voice)] + [(e, e.default_voice(lang, self.gender)) for e in self.cachos_tts.engines
                                       if e is not eng]
        for e, v in candidates:
            if v is None:
                continue
            try:
                audio = await e.synthesize(text, v, lang)
                if len(audio):
                    if e is not eng:
                        logger.warning(f"{self}: {voice.id} failed ({'; '.join(tried)}); using {v.id}")
                        self._context_voice[context_id] = (e, v, lang)
                    return audio
                tried.append(f"{v.id}: no audio")
            except Exception as ex:
                tried.append(f"{v.id}: {ex}")
        raise RuntimeError("; ".join(tried) or f"no voice for {lang!r}")

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            eng, voice, lang = self._pick(text, context_id)
        except Exception as e:
            yield ErrorFrame(error=f"cachos TTS: {e}")
            return
        await self.start_tts_usage_metrics(text)
        try:
            # synthesize first so we know the engine's native rate, then let Pipecat resample
            audio = await self._synthesize_with_fallback(text, eng, voice, lang, context_id)

            async def once() -> AsyncIterator[bytes]:
                yield audio.to_pcm16()

            async for frame in self._stream_audio_frames_from_iterator(
                    once(), in_sample_rate=audio.sample_rate, context_id=context_id):
                await self.stop_ttfb_metrics()
                yield frame
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"cachos TTS failed: {e}")
        finally:
            await self.stop_ttfb_metrics()


# scripts written without spaces between words: count characters, not whitespace tokens
_UNSPACED = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u0e00-\u0e7f"
                       r"\u0e80-\u0eff\u1000-\u109f\u1780-\u17ff]")


def count_words(text: str) -> int:
    """Word count that also works for Chinese/Japanese/Thai/Lao/Burmese/Khmer
    (~1.5 characters per word), where ``str.split`` sees one "word"."""
    unspaced = len(_UNSPACED.findall(text))
    spaced = len(_UNSPACED.sub(" ", text).split())
    return spaced + round(unspaced / 1.5)


class MultilingualMinWordsUserTurnStartStrategy(MinWordsUserTurnStartStrategy):
    """Pipecat's ``MinWordsUserTurnStartStrategy`` with :func:`count_words`.

    While the bot speaks, the user only interrupts after ``min_words`` words,
    so coughs, noise and "mm-hm" backchannels don't cut the bot off.
    """

    async def _handle_transcription(self, frame):
        min_words = self._min_words if self._bot_speaking else 1
        if count_words(frame.text) >= min_words:
            await self.report_user_turn_started()
        else:
            await self.trigger_reset_aggregation()


class PauseOnOverlap(FrameProcessor):
    """Pause the agent while the user makes a sound; resume if it wasn't a real interruption.

    LiveKit Agents' false-interruption handling (``resume_false_interruption``,
    ``false_interruption_timeout=2.0``), which Pipecat lacks. Put it right before
    ``transport.output()`` and pair it with a word-based interruption strategy
    (``VoiceAgent(interrupt_min_words=2)``): a cough or "mm-hm" pauses the agent
    instead of cutting it off, and it carries on where it stopped; real words
    interrupt as usual (Pipecat's ``InterruptionFrame`` clears what was held).

    Pipecat's output transport queues whole sentences ahead of playback, so a
    pause there would only take effect at the next sentence. This gate releases
    audio just ``lead_secs`` ahead of real time instead, so pausing takes effect
    within ~a quarter second. Frame order is preserved.
    """

    def __init__(self, *, resume_after_secs: float = 2.0, lead_secs: float = 0.25, **kwargs) -> None:
        super().__init__(**kwargs)
        self.resume_after_secs = resume_after_secs
        self.lead_secs = lead_secs
        self._queue: deque = deque()
        self._wake = asyncio.Event()
        self._paused = False
        self._resume_task: Optional[asyncio.Task] = None
        self._task: Optional[asyncio.Task] = None
        self._ahead_until = 0.0  # monotonic time at which released audio finishes playing

    @property
    def paused(self) -> bool:
        return self._paused

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._task = self.create_task(self._release_loop())
        elif isinstance(frame, CancelFrame):
            await self._stop_tasks()
        elif isinstance(frame, InterruptionFrame):  # a real interruption: drop what was held
            self._queue.clear()
            self._paused = False
            self._ahead_until = 0.0
            await self._cancel_resume()
        elif isinstance(frame, VADUserStartedSpeakingFrame) and self._speaking():
            self._paused = True
            await self._cancel_resume()
        elif isinstance(frame, VADUserStoppedSpeakingFrame) and self._paused:
            await self._cancel_resume()
            self._resume_task = self.create_task(self._resume_later())

        if direction == FrameDirection.DOWNSTREAM and not isinstance(frame, SystemFrame) \
                and (self._queue or isinstance(frame, OutputAudioRawFrame)):
            self._queue.append(frame)  # keep order behind held audio
            self._wake.set()
        else:
            await self.push_frame(frame, direction)

    def _speaking(self) -> bool:
        return bool(self._queue) or time.monotonic() < self._ahead_until

    async def _resume_later(self):
        await asyncio.sleep(self.resume_after_secs)
        self._paused = False
        self._resume_task = None
        self._wake.set()

    async def _release_loop(self):
        while True:
            if not self._queue or self._paused:
                self._wake.clear()
                await self._wake.wait()
                continue
            frame = self._queue[0]
            if isinstance(frame, OutputAudioRawFrame):
                now = time.monotonic()
                wait = self._ahead_until - self.lead_secs - now
                if wait > 0:
                    await asyncio.sleep(min(wait, 0.05))
                    continue
                secs = len(frame.audio) / (2 * frame.num_channels * frame.sample_rate)
                self._ahead_until = max(self._ahead_until, now) + secs
            self._queue.popleft()
            await self.push_frame(frame)
            if isinstance(frame, EndFrame):  # graceful stop: everything before it was played
                self._task = None
                return

    async def cleanup(self):
        await super().cleanup()
        await self._stop_tasks()

    async def _cancel_resume(self):
        if self._resume_task:
            await self.cancel_task(self._resume_task)
            self._resume_task = None

    async def _stop_tasks(self):
        await self._cancel_resume()
        if self._task:
            await self.cancel_task(self._task)
            self._task = None


class DenoiseForVAD(FrameProcessor):
    """Denoise input audio for everything *downstream* (VAD, Smart Turn, barge-in).

    Place it right after the STT service: STT has already consumed the
    original audio, so recognition accuracy is unaffected while VAD and turn
    detection get a clean signal (fewer false starts / interruptions from noise).
    """

    def __init__(self, model: str = "gtcrn", **kwargs) -> None:
        super().__init__(**kwargs)
        self._model = model
        self._denoiser: Optional[StreamingDenoiser] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._denoiser is None:
                self._denoiser = await asyncio.to_thread(StreamingDenoiser, self._model)
            x = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
            y = await asyncio.to_thread(self._denoiser.process, x, frame.sample_rate)
            if len(y) == 0:
                return  # denoiser still filling its first window
            frame = InputAudioRawFrame(audio=(np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes(),
                                       sample_rate=frame.sample_rate, num_channels=frame.num_channels)
        await self.push_frame(frame, direction)


class DenoiseFilter(BaseAudioFilter):
    """Transport-level filter (``TransportParams(audio_in_filter=DenoiseFilter())``):
    denoises audio for *everything* including STT. Prefer :class:`DenoiseForVAD`
    unless your environment is very noisy and you have A/B-tested WER."""

    def __init__(self, model: str = "gtcrn") -> None:
        self._model = model
        self._denoiser: Optional[StreamingDenoiser] = None
        self._sample_rate = 16000
        self._enabled = True

    async def start(self, sample_rate: int):
        self._sample_rate = sample_rate
        self._denoiser = await asyncio.to_thread(StreamingDenoiser, self._model)

    async def stop(self):
        self._denoiser = None

    async def process_frame(self, frame: FilterControlFrame):
        if isinstance(frame, FilterEnableFrame):
            self._enabled = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        if not self._enabled or self._denoiser is None:
            return audio
        x = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        y = self._denoiser.process(x, self._sample_rate)
        return (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()
