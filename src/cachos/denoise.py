"""Speech denoising, built on proven models and runtimes (no custom DSP).

* ``gtcrn`` (default) – GTCRN (ICASSP'24, MIT): 48K params, 16 kHz native,
  32 ms latency, ~0.8 ms CPU per 16 ms hop; beats RNNoise on VoiceBank-DEMAND
  (PESQ 2.87 vs ~2.29) and DNS3. Runs through k2-fsa **sherpa-onnx**.
* ``dpdfnet2`` / ``dpdfnet4`` / ``dpdfnet8`` / ``dpdfnet_baseline`` –
  DPDFNet (Apache-2.0, DeepFilterNet2 + dual-path RNN): higher quality at
  ~4-12x GTCRN's compute, still CPU real-time. Also through sherpa-onnx.

Model choice and evidence: docs/RESEARCH.md §8.

Important: published studies show front-end denoising often *raises* Whisper
WER, so the agent feeds denoised audio to VAD / turn detection / barge-in and
the original audio to STT by default (``VoiceAgent(stt_hears="original")``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .audio import Audio
from .utils import cache_dir, download

_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/"
MODELS = {
    "gtcrn": "gtcrn_simple.onnx",
    "dpdfnet_baseline": "dpdfnet_baseline.onnx",
    "dpdfnet2": "dpdfnet2.onnx",
    "dpdfnet4": "dpdfnet4.onnx",
    "dpdfnet8": "dpdfnet8.onnx",
}


def _model_config(model: str, num_threads: int):
    import sherpa_onnx

    if model not in MODELS:
        raise ValueError(f"unknown denoiser model {model!r}; choose from {list(MODELS)}")
    path = str(download(_RELEASE + MODELS[model], cache_dir("denoise") / MODELS[model]))
    kind = {"gtcrn": sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig}.get(model)
    if kind is not None:
        return sherpa_onnx.OfflineSpeechDenoiserModelConfig(gtcrn=kind(model=path), num_threads=num_threads,
                                                            provider="cpu")
    return sherpa_onnx.OfflineSpeechDenoiserModelConfig(
        dpdfnet=sherpa_onnx.OfflineSpeechDenoiserDpdfNetModelConfig(model=path),
        num_threads=num_threads, provider="cpu")


class StreamingDenoiser:
    """Stateful streaming denoiser (one per audio stream / call)."""

    def __init__(self, model: str = "gtcrn", num_threads: int = 1) -> None:
        import sherpa_onnx

        self.model = model
        self._config = sherpa_onnx.OnlineSpeechDenoiserConfig(model=_model_config(model, num_threads))
        self._sd = sherpa_onnx.OnlineSpeechDenoiser(self._config)

    @property
    def sample_rate(self) -> int:
        return self._sd.sample_rate

    def process(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Feed float32 samples; returns the denoised samples available so far (at ``sample_rate``)."""
        out = np.asarray(self._sd(np.asarray(samples, dtype=np.float32), sample_rate).samples, dtype=np.float32)
        if self._sd.sample_rate != sample_rate and len(out):
            out = Audio(out, self._sd.sample_rate).resample(sample_rate).samples
        return out

    def flush(self, sample_rate: int) -> np.ndarray:
        out = np.asarray(self._sd.flush().samples, dtype=np.float32)
        if self._sd.sample_rate != sample_rate and len(out):
            out = Audio(out, self._sd.sample_rate).resample(sample_rate).samples
        return out

    def reset(self) -> None:
        import sherpa_onnx

        self._sd = sherpa_onnx.OnlineSpeechDenoiser(self._config)


def denoise(audio: Audio, model: str = "gtcrn", num_threads: int = 2) -> Audio:
    """Denoise a complete clip (files, voicemail, offline transcription experiments)."""
    import sherpa_onnx

    off = sherpa_onnx.OfflineSpeechDenoiser(sherpa_onnx.OfflineSpeechDenoiserConfig(
        model=_model_config(model, num_threads)))
    res = off(audio.samples, audio.sample_rate)
    return Audio(np.asarray(res.samples, dtype=np.float32), res.sample_rate).resample(audio.sample_rate)


def available() -> Optional[str]:
    """Why denoising is unavailable, or None if it is."""
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return "sherpa-onnx is not installed (`pip install cachos[agent]`)"
    return None
