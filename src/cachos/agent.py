"""Real-time voice agent: cachos STT/TTS on the Pipecat runtime.

Pipeline (one per call/session)::

    transport.input()            mic / WebRTC / telephony WebSocket
    CachosSTTService             hears the ORIGINAL audio (best WER), detects language
    DenoiseForVAD                GTCRN via sherpa-onnx: clean audio for everything below
    user aggregator              Silero VAD + Smart Turn v3.2 end-of-turn (Pipecat)
    OpenAILLMService             any OpenAI-compatible endpoint (OpenAI, Groq, Ollama, vLLM…)
    CachosTTSService             voice chosen per reply language
    transport.output()
    assistant aggregator

Usage::

    agent = VoiceAgent(llm_model="gpt-4.1-mini", system_prompt="You are a helpful receptionist.")
    agent.run_local()                       # terminal: mic + speakers
    agent.serve()                           # browser (WebRTC, echo cancelled) / Twilio WebSocket
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

from .core import STT, TTS

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Your replies are spoken aloud, so answer in short, natural "
    "sentences without emojis, markdown, lists or symbols that cannot be spoken. Always reply in the "
    "language the user is speaking."
)


@dataclass
class VoiceAgent:
    """Configuration for a voice agent; each session builds a fresh Pipecat pipeline.

    Args:
        llm_model / llm_base_url / llm_api_key: an OpenAI-compatible chat endpoint
            (defaults: ``gpt-4.1-mini``, ``$OPENAI_BASE_URL``, ``$OPENAI_API_KEY``).
        system_prompt: instructions for the LLM.
        greeting: if set, the agent speaks first by answering this instruction.
        stt / tts: shared :class:`cachos.STT` / :class:`cachos.TTS` (models load once).
        language: fix the STT language (``None`` = detect per utterance).
        voice: fix the TTS voice (``None`` = pick per reply language).
        denoise: ``"gtcrn"`` (default), ``"dpdfnet2"`` … or ``None`` to disable.
        denoise_stt: also denoise what STT hears. Off by default: front-end denoising
            usually raises Whisper's WER (docs/RESEARCH.md §8); A/B-test before enabling.
        smart_turn: use Smart Turn v3.2 for end-of-turn (Pipecat default); False = VAD silence only.
        interrupt_min_words: require this many transcribed words before the user can
            interrupt the bot (robust barge-in in noisy places); ``None`` = interrupt on voice.
        pause_on_overlap: pause the bot while the user makes a sound and resume if no
            real interruption follows within ``resume_after_secs`` (LiveKit's
            false-interruption handling). Default: on when ``interrupt_min_words`` is set.
        filter_incomplete_turns: let the LLM hold its reply when the user is mid-thought
            (Pipecat ``FilterIncompleteUserTurnStrategies``).
        vad_stop_secs: silence before VAD reports the user stopped (Smart Turn then decides).
        allow_client_config: let a connecting client set ``language``, ``voice``,
            ``gender``, ``system_prompt``, ``greeting`` and ``interrupt_min_words`` for its
            session (the ``body`` of Pipecat's ``/start`` request). Off by default: in
            production you rarely want callers rewriting the system prompt.
    """

    llm_model: str = "gpt-4.1-mini"
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    greeting: Optional[str] = None
    stt: Optional[STT] = None
    tts: Optional[TTS] = None
    language: Optional[str] = None
    voice: Optional[str] = None
    gender: Optional[str] = None
    denoise: Optional[str] = "gtcrn"
    denoise_stt: bool = False
    smart_turn: bool = True
    interrupt_min_words: Optional[int] = None
    pause_on_overlap: Optional[bool] = None
    resume_after_secs: float = 2.0
    filter_incomplete_turns: bool = False
    vad_stop_secs: float = 0.2
    allow_client_config: bool = False
    llm_settings: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.stt = self.stt or STT()
        self.tts = self.tts or TTS()
        self._warm = False

    async def warmup(self) -> None:
        """Load model weights once, before any session starts. Warming up inside a
        pipeline would stall its start while caller audio piles up and then floods
        VAD in a burst."""
        if self._warm:
            return
        import asyncio

        from loguru import logger

        for name, part in (("STT", self.stt), ("TTS", self.tts)):
            try:
                await part.warmup()
            except Exception as e:
                logger.warning(f"cachos: {name} warmup failed: {e}")
        try:  # measure real speech-end -> transcript latency so the first turn waits long enough
            import time
            from collections import deque

            speech = await self.tts.synthesize("Hello, I would like to book a table for two people tonight.")
            started = time.monotonic()
            await self.stt.transcribe(speech, self.language)
            if not hasattr(self.stt, "latencies"):
                self.stt.latencies = deque(maxlen=50)
            self.stt.latencies.append(time.monotonic() - started)
            logger.info(f"cachos: STT latency on this machine ~{self.stt.latencies[-1]:.2f}s")
        except Exception as e:
            logger.warning(f"cachos: latency calibration skipped: {e}")
        if self.denoise:
            from .denoise import StreamingDenoiser

            await asyncio.to_thread(StreamingDenoiser, self.denoise)  # downloads the model if needed
        self._warm = True

    # ---- building blocks --------------------------------------------------------
    def build_llm(self):
        from pipecat.services.openai.llm import OpenAILLMService

        kwargs: dict[str, Any] = {"api_key": self.llm_api_key or os.environ.get("OPENAI_API_KEY") or "none"}
        base_url = self.llm_base_url or os.environ.get("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAILLMService(
            **kwargs,
            settings=OpenAILLMService.Settings(model=self.llm_model, system_instruction=self.system_prompt,
                                               **self.llm_settings),
        )

    def user_params(self):
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
        from pipecat.turns.user_turn_strategies import FilterIncompleteUserTurnStrategies, UserTurnStrategies

        strategies: dict[str, Any] = {}
        if self.interrupt_min_words:
            from .pipecat import MultilingualMinWordsUserTurnStartStrategy

            strategies["start"] = [MultilingualMinWordsUserTurnStartStrategy(min_words=self.interrupt_min_words)]
        if not self.smart_turn:
            from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy

            strategies["stop"] = [SpeechTimeoutUserTurnStopStrategy()]
        cls = FilterIncompleteUserTurnStrategies if self.filter_incomplete_turns else UserTurnStrategies
        from .pipecat import _latency_estimate

        # Pipecat ends any open turn after `user_turn_stop_timeout` (default 5 s) even while the
        # transcript is still coming; on slow hardware that yields empty turns, so cover the
        # measured STT latency.
        stt_wait = _latency_estimate(getattr(self.stt, "latencies", ()))
        return LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=self.vad_stop_secs)),
            user_turn_strategies=cls(**strategies),
            user_turn_stop_timeout=max(5.0, stt_wait + 1.0),
        )

    def build_pipeline(self, transport):
        """Return ``(pipeline, context)`` for one session on ``transport``."""
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

        from .pipecat import CachosSTTService, CachosTTSService, DenoiseForVAD, LanguageTracker

        tracker = LanguageTracker(language=self.language)
        stt = CachosSTTService(self.stt, language=self.language, tracker=tracker)
        tts = CachosTTSService(self.tts, voice=self.voice, gender=self.gender, tracker=tracker)
        context = LLMContext()
        user_agg, assistant_agg = LLMContextAggregatorPair(context, user_params=self.user_params())
        processors: list = [transport.input(), stt]
        if self.denoise and not self.denoise_stt:
            processors.append(DenoiseForVAD(self.denoise))
        processors += [user_agg, self.build_llm(), tts]
        pause = self.pause_on_overlap if self.pause_on_overlap is not None else bool(self.interrupt_min_words)
        if pause:
            from .pipecat import PauseOnOverlap

            processors.append(PauseOnOverlap(resume_after_secs=self.resume_after_secs))
        processors += [transport.output(), assistant_agg]
        return Pipeline(processors), context

    def transport_params(self, kind: str = "webrtc"):
        """Pipecat transport params; with ``denoise_stt`` the denoiser runs at the transport."""
        from pipecat.transports.base_transport import TransportParams

        extra: dict[str, Any] = {}
        if self.denoise and self.denoise_stt:
            from .pipecat import DenoiseFilter

            extra["audio_in_filter"] = DenoiseFilter(self.denoise)
        if kind in _WEBSOCKET_TRANSPORTS:
            from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

            return FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True, **extra)
        if kind == "local":
            from pipecat.transports.local.audio import LocalAudioTransportParams

            return LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True, **extra)
        return TransportParams(audio_in_enabled=True, audio_out_enabled=True, **extra)

    # ---- running ------------------------------------------------------------------
    async def run(self, transport, *, handle_sigint: bool = True, idle_timeout_secs: Optional[float] = None,
                  observers: Sequence = ()) -> None:
        """Run one session on any Pipecat transport until the client leaves."""
        from pipecat.frames.frames import LLMRunFrame
        from pipecat.pipeline.worker import PipelineParams, PipelineWorker
        from pipecat.workers.runner import WorkerRunner

        await self.warmup()
        pipeline, context = self.build_pipeline(transport)
        worker_kwargs: dict[str, Any] = {"observers": list(observers)}
        if idle_timeout_secs is not None:
            worker_kwargs["idle_timeout_secs"] = idle_timeout_secs
        worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
                                **worker_kwargs)
        runner = WorkerRunner(handle_sigint=handle_sigint)
        await runner.add_workers(worker)

        async def greet() -> None:
            if self.greeting:
                context.add_message({"role": "developer", "content": self.greeting})
                await worker.queue_frames([LLMRunFrame()])

        if _has_client_events(transport):  # WebRTC / WebSocket: one session per client
            @transport.event_handler("on_client_connected")
            async def _connected(transport, client):
                await greet()

            @transport.event_handler("on_client_disconnected")
            async def _disconnected(transport, client):
                await runner.cancel()
        else:  # local audio: talk right away
            await greet()
        await runner.run()

    def run_local(self) -> None:
        """Talk to the agent with this machine's mic and speakers (use headphones: no AEC)."""
        import asyncio

        from pipecat.transports.local.audio import LocalAudioTransport

        transport = LocalAudioTransport(self.transport_params("local"))
        asyncio.run(self.run(transport))

    #: settings a client may choose per session when ``allow_client_config`` is on
    CLIENT_CONFIG_KEYS = ("language", "voice", "gender", "system_prompt", "greeting", "interrupt_min_words")

    def for_session(self, config: Optional[dict] = None) -> "VoiceAgent":
        """This agent with a client's per-session settings applied (models stay shared)."""
        if not (self.allow_client_config and isinstance(config, dict)):
            return self
        changes = {k: (v if v not in ("", "auto") else None) for k, v in config.items()
                   if k in self.CLIENT_CONFIG_KEYS}
        if changes.get("system_prompt") is None:
            changes.pop("system_prompt", None)
        if changes.get("interrupt_min_words") is not None:
            changes["interrupt_min_words"] = int(changes["interrupt_min_words"]) or None
        session = replace(self, **changes)
        session._warm = self._warm
        return session

    async def bot(self, runner_args) -> None:
        """Entry point for Pipecat's development runner (``pipecat.runner.run``)."""
        from pipecat.runner.utils import create_transport

        params = {k: (lambda k=k: self.transport_params(k)) for k in ("webrtc", *_WEBSOCKET_TRANSPORTS)}
        transport = await create_transport(runner_args, params)
        agent = self.for_session(getattr(runner_args, "body", None))
        await agent.run(transport, handle_sigint=runner_args.handle_sigint,
                        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs)

    def serve(self, host: str = "localhost", port: int = 7860, transport: Optional[str] = "webrtc",
              test_ui: bool = False) -> None:
        """Serve with Pipecat's runner: ``webrtc`` (browser client at /client) or a telephony
        WebSocket (``twilio``, ``telnyx``, ``plivo``, ``exotel``); ``None`` enables all.
        ``test_ui`` adds the cachos test page at ``/test`` (and enables ``allow_client_config``)."""
        from .runner import serve

        if test_ui:
            self.allow_client_config = True
        serve(self, host=host, port=port, transport=transport, test_ui=test_ui)


_WEBSOCKET_TRANSPORTS = ("websocket", "twilio", "telnyx", "plivo", "exotel")


def _has_client_events(transport) -> bool:
    return "on_client_connected" in getattr(transport, "_event_handlers", {})
