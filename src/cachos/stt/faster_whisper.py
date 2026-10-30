"""faster-whisper (CTranslate2 Whisper) – 99 languages with auto-detection.

Default model ``large-v3`` (accuracy first). ``large-v3-turbo`` is ~25-35% faster but
measurably weaker on lower-resource languages (it heard Bengali as Gujarati / Hindi);
with turbo, utterances whose language is detected with low confidence are re-run on
large-v3 (``escalate_model``).
"""

from __future__ import annotations

import warnings
from typing import ClassVar, Optional, Sequence

import numpy as np

from ..audio import Audio
from ..languages import normalize_language
from ..types import Segment, Transcript, Word
from .base import STTEngine


class FasterWhisperSTT(STTEngine):
    name: ClassVar[str] = "faster-whisper"
    languages: ClassVar[list[str]] = ["*"]  # Whisper's 99 languages

    def __init__(
        self,
        model: str = "large-v3",
        *,
        device: str = "auto",
        compute_type: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = False,
        language_candidates: Optional[Sequence[str]] = None,
        hotwords: Optional[str] = None,
        no_speech_prob: float = 0.6,
        escalate_model: Optional[str] = "large-v3",
        escalate_below: float = 0.75,
        cpu_threads: int = 0,
        download_root: Optional[str] = None,
    ) -> None:
        """
        Args:
            language_candidates: restrict auto-detection to these languages (e.g.
                ``["en", "hi", "bn"]`` for a call center). Much more robust than
                open-set detection on short utterances.
            hotwords: domain vocabulary / names to bias decoding toward
                (e.g. ``"Cachos, Kubernetes, Dr. Okafor"``).
            no_speech_prob: segments Whisper itself rates as probably-not-speech
                above this are dropped (the main defence against hallucinated
                "Thank you." on noise; same filter as Pipecat's WhisperSTTService).
            escalate_model: when the language is auto-detected with probability below
                ``escalate_below``, re-transcribe with this (more accurate) model.
                Measured: large-v3-turbo heard Bengali as Gujarati (p=0.34) and as
                Hindi (p=0.68) where large-v3 got it right (p=0.93-0.96); turbo was
                confident (p>=0.87) where it was right. Typical speech stays on the fast model;
                only uncertain utterances pay for the big one. ``None`` disables.
        """
        super().__init__()
        import faster_whisper  # noqa: F401  (fail fast if not installed)

        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.word_timestamps = word_timestamps
        self.language_candidates = [normalize_language(c) for c in language_candidates] if language_candidates else None
        self.hotwords = hotwords
        self.no_speech_prob = no_speech_prob
        self.escalate_model = escalate_model if escalate_model != model else None
        self.escalate_below = escalate_below
        self._escalated = None
        self.cpu_threads = cpu_threads
        self.download_root = download_root
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self, model_name: Optional[str] = None):
        from faster_whisper import WhisperModel

        model_name = model_name or self.model_name

        def build(device: str):
            compute = self.compute_type or ("int8_float16" if device == "cuda" else "int8")
            return WhisperModel(model_name, device=device, compute_type=compute,
                                cpu_threads=self.cpu_threads, download_root=self.download_root)

        if self.device != "auto":
            if self.device == "cuda":
                _preload_cuda_libs()
            return build(self.device)
        try:
            import ctranslate2

            has_gpu = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            has_gpu = False
        if has_gpu:
            _preload_cuda_libs()
            try:
                m = build("cuda")
                m.detect_language(np.zeros(16000, dtype=np.float32))  # cuBLAS/cuDNN load lazily: probe now
                return m
            except Exception as e:  # GPU present but CUDA 12 / cuDNN 9 libraries missing
                warnings.warn(f"cachos: CUDA unusable ({e}); using CPU. For GPU: pip install 'cachos[cuda]'")
        return build("cpu")

    async def warmup(self) -> None:
        """Load the main model and the escalation model up front (loading large-v3
        mid-conversation would stall a turn for tens of seconds)."""
        import asyncio

        await super().warmup()
        if self.escalate_model and self._escalated is None:
            self._escalated = await asyncio.to_thread(self._load, self.escalate_model)

    def _pick_language(self, samples) -> tuple[Optional[str], Optional[float]]:
        """Detect among ``language_candidates`` only."""
        if not self.language_candidates:
            return None, None
        try:
            _lang, _p, all_probs = self.model.detect_language(samples)
        except Exception:
            return None, None
        allowed = [(l, p) for l, p in all_probs if normalize_language(l) in self.language_candidates]
        return max(allowed, key=lambda lp: lp[1]) if allowed else (self.language_candidates[0], None)

    def _transcribe(self, audio: Audio, language: Optional[str], **options) -> Transcript:
        prob: Optional[float] = None
        if language is None and self.language_candidates:
            language, prob = self._pick_language(audio.samples)
        word_ts = options.pop("word_timestamps", self.word_timestamps)
        # Agent-friendly anti-hallucination preset (see docs/RESEARCH.md §9)
        params = dict(
            language=language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            word_timestamps=word_ts,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            hotwords=self.hotwords,
        )
        params.update(options)
        t = self._run(self.model, self.model_name, audio, params, prob)
        if (language is None and self.escalate_model and t.language_probability is not None
                and t.language_probability < self.escalate_below):
            if self._escalated is None:
                self._escalated = self._load(self.escalate_model)
            t2 = self._run(self._escalated, self.escalate_model, audio, params, None)
            if t2.text or not t.text:
                return t2
        return t

    def _run(self, model, model_name: str, audio: Audio, params: dict, prob: Optional[float]) -> Transcript:
        segments_iter, info = model.transcribe(audio.samples, **params)
        segments: list[Segment] = []
        for s in segments_iter:
            if s.no_speech_prob > self.no_speech_prob:
                continue
            words = [Word(w.word, w.start, w.end, w.probability) for w in (s.words or [])]
            segments.append(Segment(s.text.strip(), s.start, s.end, words, info.language,
                                    confidence=float(2.718281828 ** s.avg_logprob) if s.avg_logprob is not None else None))
        text = " ".join(s.text for s in segments if s.text)
        if info.language in ("zh", "ja", "yue", "th", "lo", "my", "km"):
            text = "".join(s.text for s in segments)  # no spaces between segments in these scripts
        return Transcript(
            text=text.strip(),
            language=normalize_language(info.language),
            language_probability=prob if prob is not None else info.language_probability,
            segments=segments,
            duration=info.duration,
            engine=f"{self.name}:{model_name}",
        )


def _preload_cuda_libs() -> None:
    """Load cuBLAS/cuDNN from the ``nvidia-*-cu12`` pip wheels (``cachos[cuda]``), if present,
    so CTranslate2 finds them without LD_LIBRARY_PATH (the setup faster-whisper documents)."""
    import ctypes
    import glob
    import importlib.util
    import os

    for pkg, patterns in (("nvidia.cublas", ("libcublasLt.so.*", "libcublas.so.*")),
                          ("nvidia.cudnn", ("libcudnn*.so.*",))):
        try:
            spec = importlib.util.find_spec(pkg)
        except ModuleNotFoundError:
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        libdir = os.path.join(list(spec.submodule_search_locations)[0], "lib")
        for pat in patterns:
            for lib in sorted(glob.glob(os.path.join(libdir, pat))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
