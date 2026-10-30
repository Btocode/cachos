"""Kokoro-82M via ``kokoro-onnx`` – tiny, Apache-2.0, excellent quality, fast on CPU.

~54 voices across American/British English, Spanish, French, Hindi, Italian,
Brazilian Portuguese, Japanese and Mandarin. Model files (~310 MB fp32, ~88 MB
int8) are downloaded on first use.
"""

from __future__ import annotations

from typing import ClassVar, Optional

from ..audio import Audio
from ..types import Voice
from ..utils import cache_dir, download
from .base import TTSEngine

_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
_MODELS = {"fp32": "kokoro-v1.0.onnx", "fp16": "kokoro-v1.0.fp16.onnx", "int8": "kokoro-v1.0.int8.onnx"}
_VOICES_FILE = "voices-v1.0.bin"

# voice-id prefix letter -> (iso language, espeak/kokoro lang code, accent label)
_PREFIX = {
    "a": ("en", "en-us", "American English"),
    "b": ("en", "en-gb", "British English"),
    "e": ("es", "es", "Spanish"),
    "f": ("fr", "fr-fr", "French"),
    "h": ("hi", "hi", "Hindi"),
    "i": ("it", "it", "Italian"),
    "j": ("ja", "ja", "Japanese"),
    "p": ("pt", "pt-br", "Brazilian Portuguese"),
    "z": ("zh", "cmn", "Mandarin Chinese"),
}
# Best-rated voices first so they become per-language defaults (per the Kokoro VOICES.md grades)
_PREFERRED = ["af_heart", "af_bella", "am_michael", "am_fenrir", "bf_emma", "bm_george", "ef_dora", "em_alex",
              "ff_siwis", "hf_alpha", "hm_omega", "if_sara", "im_nicola", "jf_alpha", "jm_kumo", "pf_dora",
              "pm_alex", "zf_xiaoxiao", "zm_yunxi"]
_HIGH = {"af_heart", "af_bella", "af_nicole", "bf_emma", "ff_siwis"}
# Japanese / Mandarin need Kokoro's own G2P (misaki); espeak phonemes for them are unusable
# (measured: Japanese replies came out as gibberish). Without misaki these voices are hidden
# so the router falls back to Piper/Edge for ja/zh.
_MISAKI = {"j": "ja", "z": "zh"}


def _misaki_g2p(lang: str):
    """misaki G2P for ``ja``/``zh`` (``pip install cachos[kokoro-cjk]``), or None."""
    try:
        if lang == "ja":
            from misaki import ja

            return ja.JAG2P()
        from misaki import zh

        return zh.ZHG2P()
    except Exception:
        return None


class KokoroTTS(TTSEngine):
    name: ClassVar[str] = "kokoro"
    sample_rate: ClassVar[int] = 24000

    def __init__(self, model: str = "int8", *, model_path: Optional[str] = None,
                 voices_path: Optional[str] = None) -> None:
        """
        Args:
            model: ``"int8"`` (default, 88 MB, fastest on CPU), ``"fp16"`` or ``"fp32"``.
        """
        super().__init__()
        import kokoro_onnx  # noqa: F401

        if model not in _MODELS:
            raise ValueError(f"kokoro model must be one of {list(_MODELS)}")
        self.variant = model
        self.model_path = model_path
        self.voices_path = voices_path
        self._kokoro = None
        self._g2p: dict[str, object] = {}

    def _g2p_for(self, prefix: str):
        lang = _MISAKI.get(prefix)
        if lang is None:
            return None
        if lang not in self._g2p:
            self._g2p[lang] = _misaki_g2p(lang)
        return self._g2p[lang]

    @property
    def kokoro(self):
        if self._kokoro is None:
            from kokoro_onnx import Kokoro

            d = cache_dir("kokoro")
            model = self.model_path or str(download(_RELEASE + _MODELS[self.variant], d / _MODELS[self.variant]))
            voices = self.voices_path or str(download(_RELEASE + _VOICES_FILE, d / _VOICES_FILE))
            self._kokoro = Kokoro(model, voices)
        return self._kokoro

    def list_voices(self) -> list[Voice]:
        names = list(self.kokoro.get_voices())
        names.sort(key=lambda n: (_PREFERRED.index(n) if n in _PREFERRED else len(_PREFERRED), n))
        out = []
        for n in names:
            if len(n) < 3 or n[0] not in _PREFIX:
                continue
            if n[0] in _MISAKI and self._g2p_for(n[0]) is None:
                continue
            lang, _code, accent = _PREFIX[n[0]]
            out.append(Voice(
                id=f"kokoro:{n}",
                name=n.split("_", 1)[-1].title(),
                engine=self.name,
                languages=[lang],
                gender="female" if n[1] == "f" else "male" if n[1] == "m" else None,
                description=accent,
                sample_rate=self.sample_rate,
                quality="high" if n in _HIGH else "medium",
            ))
        return out

    def _synthesize(self, text: str, voice: Voice, language: str, speed: float, **options) -> Audio:
        prefix = voice.native_id[0]
        lang_code = _PREFIX.get(prefix, ("en", "en-us", ""))[1]
        g2p = self._g2p_for(prefix)
        if g2p is not None:
            phonemes, _ = g2p(text)
            samples, sr = self.kokoro.create(phonemes, voice=voice.native_id, speed=speed, is_phonemes=True)
        else:
            samples, sr = self.kokoro.create(text, voice=voice.native_id, speed=speed, lang=lang_code)
        return Audio(samples, sr)
