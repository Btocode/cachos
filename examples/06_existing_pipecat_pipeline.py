"""Use cachos pieces inside your own Pipecat pipeline (keep your transport, LLM, tools).

cachos adds: multilingual Whisper that passes the detected language on, replies in a
voice matching that language, and GTCRN denoising for VAD / Smart Turn only.
"""
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

from cachos import STT, TTS
from cachos.pipecat import CachosSTTService, CachosTTSService, DenoiseForVAD, LanguageTracker


def build_pipeline(transport, llm):
    tracker = LanguageTracker()  # STT tells TTS which language the user spoke
    user_agg, assistant_agg = LLMContextAggregatorPair(
        LLMContext(), user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()))
    return Pipeline([
        transport.input(),
        CachosSTTService(STT(), tracker=tracker),   # hears the original audio
        DenoiseForVAD("gtcrn"),                     # VAD + Smart Turn hear clean audio
        user_agg,
        llm,                                        # any Pipecat LLM service
        CachosTTSService(TTS(), tracker=tracker),
        transport.output(),
        assistant_agg,
    ])
