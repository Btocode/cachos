"""cachos services inside real Pipecat pipelines (pipecat.tests.utils.run_test)."""

import numpy as np
import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (  # noqa: E402
    ErrorFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402
from pipecat.tests.utils import run_test  # noqa: E402

from cachos import STT, TTS  # noqa: E402
from cachos.pipecat import (  # noqa: E402
    CachosSTTService,
    CachosTTSService,
    LanguageTracker,
    count_words,
)

from .conftest import FakeSTT, FakeTTS, tone  # noqa: E402


async def test_stt_service_propagates_detected_language():
    tracker = LanguageTracker()
    svc = CachosSTTService(STT(FakeSTT("hola amigo", "es")), tracker=tracker, sample_rate=16000)
    frames = [f async for f in _run_stt(svc, tone(0.5))]
    t = [f for f in frames if isinstance(f, TranscriptionFrame)]
    assert len(t) == 1 and t[0].text == "hola amigo" and t[0].language.value == "es"
    assert tracker.language == "es"


async def test_stt_service_drops_empty_and_reports_errors():
    class Broken(FakeSTT):
        def _transcribe(self, audio, language, **o):
            raise RuntimeError("boom")

    assert [f async for f in _run_stt(CachosSTTService(STT(FakeSTT("  ")), sample_rate=16000), tone(0.3))] == []
    out = [f async for f in _run_stt(CachosSTTService(STT(Broken()), sample_rate=16000), tone(0.3))]
    assert len(out) == 1 and isinstance(out[0], ErrorFrame)


async def _run_stt(svc, samples):
    svc._sample_rate = 16000  # normally set from the StartFrame
    async for f in svc.run_stt((samples * 32767).astype(np.int16).tobytes()):
        yield f


async def test_tts_service_speaks_in_reply_language_in_pipeline():
    eng = FakeTTS("f", ["en", "fr", "es"])
    tracker = LanguageTracker(language="es")
    svc = CachosTTSService(TTS([eng]), tracker=tracker)
    down, _ = await run_test(svc, frames_to_send=[
        TTSSpeakFrame("Bonjour, je suis très heureux de vous aider aujourd'hui."),
        TTSSpeakFrame("Sí."),  # too short to detect: falls back to the user's language
    ])
    assert any(isinstance(f, TTSAudioRawFrame) for f in down)
    langs = [c[2] for c in eng.calls]
    assert langs == ["fr", "es"], eng.calls


async def test_tts_service_fixed_voice():
    eng = FakeTTS("f", ["en", "fr"])
    svc = CachosTTSService(TTS([eng]), voice="f:fr_m")
    await run_test(svc, frames_to_send=[TTSSpeakFrame("Bonjour tout le monde, comment allez-vous?")])
    assert eng.calls[0][1] == "f:fr_m"


@pytest.mark.parametrize("text,n", [("stop talking please", 3), ("请停一下好吗", 4), ("ちょっと待って", 5),
                                    ("hey 你好", 2), ("", 0)])
def test_count_words_multilingual(text, n):
    assert count_words(text) == n


def test_agent_builds_pipeline_with_denoise_between_stt_and_vad():
    from cachos.agent import VoiceAgent
    from cachos.pipecat import DenoiseForVAD

    class FakeTransport:
        def input(self):
            return FrameProcessor(name="in")

        def output(self):
            return FrameProcessor(name="out")

    agent = VoiceAgent(stt=STT(FakeSTT()), tts=TTS([FakeTTS()]), llm_api_key="x", interrupt_min_words=2)
    pipeline, _ = agent.build_pipeline(FakeTransport())
    names = [type(p).__name__ for p in pipeline.processors]
    i = names.index("CachosSTTService")
    assert names[i + 1] == "DenoiseForVAD" and "LLMUserAggregator" in names[i + 2]
    assert names.index("CachosTTSService") < names.index("FrameProcessor", i)
    assert isinstance(pipeline.processors[i + 1], DenoiseForVAD)
    assert names[names.index("CachosTTSService") + 1] == "PauseOnOverlap"  # on with interrupt_min_words

    agent.denoise = None
    pipeline, _ = agent.build_pipeline(FakeTransport())
    assert "DenoiseForVAD" not in [type(p).__name__ for p in pipeline.processors]


@pytest.mark.integration
async def test_denoise_for_vad_outputs_audio():
    pytest.importorskip("sherpa_onnx")
    from cachos.pipecat import DenoiseForVAD

    rng = np.random.default_rng(0)
    noisy = tone(1.0) * 0.3 + rng.standard_normal(16000).astype(np.float32) * 0.05
    pcm = (noisy * 32767).astype(np.int16).tobytes()
    frames = [InputAudioRawFrame(audio=pcm[i:i + 640], sample_rate=16000, num_channels=1)
              for i in range(0, len(pcm), 640)]
    down, _ = await run_test(DenoiseForVAD(), frames_to_send=frames)
    out = b"".join(f.audio for f in down if isinstance(f, InputAudioRawFrame))
    assert len(out) > len(pcm) * 0.8


async def test_stt_latency_adapts_and_is_shared_across_sessions():
    stt = STT(FakeSTT())
    svc = CachosSTTService(stt, sample_rate=16000)
    sent = []

    async def capture():
        sent.append(svc._ttfs_p99_latency)

    svc.broadcast_service_metadata = capture
    assert svc._ttfs_p99_latency == 1.0  # nothing learned yet
    for _ in range(3):
        await svc._observe_latency(0.3)
    assert not sent  # 0.3 s * 1.25 + 0.1 < 1.0: no need to wait longer
    await svc._observe_latency(2.0)
    await svc._observe_latency(2.2)
    assert sent and sent[-1] > 2.2  # grew and was re-broadcast
    assert CachosSTTService(stt, sample_rate=16000)._ttfs_p99_latency > 2.2  # new session starts informed
    assert CachosSTTService(stt, ttfs_p99_latency=0.5)._ttfs_p99_latency == 0.5  # explicit value wins


async def test_queued_segments_are_merged_and_latency_measured_from_speech_end():
    import asyncio
    import time as _time

    class Rec(FakeSTT):
        def __init__(self):
            super().__init__()
            self.durations = []

        def _transcribe(self, audio, language, **o):
            self.durations.append(round(audio.duration, 2))
            return super()._transcribe(audio, language, **o)

    eng = Rec()
    svc = CachosSTTService(STT(eng), sample_rate=16000)
    svc._sample_rate = 16000
    pushed = []

    async def process_generator(gen):
        async for f in gen:
            pushed.append(f)

    svc.process_generator = process_generator
    seen = []

    async def observe(sec):
        seen.append(sec)

    svc._observe_latency = observe
    pcm = lambda s: (tone(s) * 32767).astype(np.int16).tobytes()  # noqa: E731
    now = _time.monotonic()
    svc._segment_ended = {1: now - 2.0, 2: now - 1.0, 3: now}
    for n, sec in ((1, 1.0), (2, 0.5), (3, 0.25)):  # three segments already waiting
        svc._segment_queue.put_nowait((pcm(sec), n))
    svc._segment_queue.put_nowait(None)
    await asyncio.wait_for(svc._segment_task_handler(), 5)
    assert eng.durations == [1.75]  # one STT call for all three
    assert svc._transcribing_segment == 3 and len(pushed) == 1
    assert len(seen) == 1 and 0 <= seen[0] < 0.5 and svc._segment_ended == {}


async def test_stt_segments_in_one_turn_share_context_and_language():
    from pipecat.frames.frames import UserStoppedSpeakingFrame

    class Rec(FakeSTT):
        def __init__(self):
            super().__init__("hola", "es")
            self.seen = []

        def _transcribe(self, audio, language, **o):
            self.seen.append((language, o.get("initial_prompt")))
            return super()._transcribe(audio, language, **o)

    eng = Rec()
    eng.name = "faster-whisper"
    svc = CachosSTTService(STT(eng), sample_rate=16000)
    [f async for f in _run_stt(svc, tone(3.0))]  # first segment: open detection, no prompt
    [f async for f in _run_stt(svc, tone(1.0))]  # short follow-up: turn language + context
    [f async for f in _run_stt(svc, tone(3.0))]  # long follow-up: context, own detection
    await svc.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    [f async for f in _run_stt(svc, tone(1.0))]  # new turn: fresh
    assert eng.seen == [(None, None), ("es", "hola"), (None, "hola hola"), (None, None)]


async def test_tts_service_falls_back_to_next_engine():
    class Broken(FakeTTS):
        def _synthesize(self, text, voice, language, speed, **options):
            raise ConnectionError("network down")

    bad, good = Broken("bad", ["en"]), FakeTTS("good", ["en"])
    svc = CachosTTSService(TTS([bad, good]))
    down, _ = await run_test(svc, frames_to_send=[TTSSpeakFrame("Hello there, how are you doing today?"),
                                                  TTSSpeakFrame("Second sentence of the same reply.")])
    assert any(isinstance(f, TTSAudioRawFrame) for f in down)
    assert [c[1] for c in good.calls] == ["good:en_f", "good:en_f"]
    assert not any(isinstance(f, ErrorFrame) for f in down)


def _audio_frames(n, secs=0.2, sr=16000):
    return [TTSAudioRawFrame(audio=bytes(int(sr * secs) * 2), sample_rate=sr, num_channels=1) for _ in range(n)]


class _Arrivals(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.times = []

    async def process_frame(self, frame, direction):
        import time as _time

        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            self.times.append(_time.monotonic())
        await self.push_frame(frame, direction)


async def test_pause_on_overlap_resumes_after_false_interruption():
    from pipecat.frames.frames import VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.tests.utils import SleepFrame

    from cachos.pipecat import PauseOnOverlap

    gate, rec = PauseOnOverlap(resume_after_secs=0.3), _Arrivals()
    await run_test(Pipeline([gate, rec]), frames_to_send=[
        *_audio_frames(10), SleepFrame(sleep=0.4),
        VADUserStartedSpeakingFrame(), SleepFrame(sleep=0.5),     # a cough while the agent talks
        VADUserStoppedSpeakingFrame(), SleepFrame(sleep=2.5),     # no words follow -> resume
    ])
    assert len(rec.times) == 10                                   # nothing lost
    gaps = [b - a for a, b in zip(rec.times, rec.times[1:])]
    assert max(gaps) > 0.7                                        # paused ~0.5 s speech + 0.3 s grace
    assert rec.times[-1] - rec.times[0] < 2.0 + 0.8 + 0.5         # then resumed, not restarted


async def test_pause_on_overlap_real_interruption_drops_held_audio():
    from pipecat.frames.frames import InterruptionFrame, VADUserStartedSpeakingFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.tests.utils import SleepFrame

    from cachos.pipecat import PauseOnOverlap

    gate, rec = PauseOnOverlap(resume_after_secs=5), _Arrivals()
    await run_test(Pipeline([gate, rec]), frames_to_send=[
        *_audio_frames(10), SleepFrame(sleep=0.4),
        VADUserStartedSpeakingFrame(), SleepFrame(sleep=0.3),
        InterruptionFrame(), SleepFrame(sleep=0.3),               # real words -> Pipecat interrupts
    ])
    assert 1 <= len(rec.times) <= 5 and not gate.paused
