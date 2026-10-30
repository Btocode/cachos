"""Piper (``piper-tts``) – fast local VITS voices, 40+ languages, CPU real-time.

The voice catalog is read from rhasspy/piper-voices on Hugging Face; a voice's
model (15-110 MB) is downloaded the first time it is used. Multi-speaker models
accept ``"piper:<model>#<speaker>"``, e.g. ``"piper:en_US-libritts_r-medium#42"``.

Licensing note: piper-tts >= 1.3 (OHF-Voice/piper1-gpl) is GPL-3.0; each voice
has its own dataset license (see its MODEL_CARD).
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import ClassVar, Optional

from ..audio import Audio
from ..languages import normalize_language
from ..types import Voice
from ..utils import cache_dir, download
from .base import TTSEngine

_HF = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
_QUALITY_RANK = {"high": 0, "medium": 1, "low": 2, "x_low": 3}
# Piper's catalog has no gender field; known single-speaker voices:
_GENDER = {
    "lessac": "female", "amy": "female", "kristin": "female", "kathleen": "female", "ljspeech": "female",
    "hfc_female": "female", "jenny_dioco": "female", "cori": "female", "alba": "female", "southern_english_female": "female",
    "ryan": "male", "joe": "male", "john": "male", "kusal": "male", "norman": "male", "hfc_male": "male",
    "alan": "male", "northern_english_male": "male", "danny": "male", "bryce": "male", "sam": "male",
    "thorsten": "male", "karlsson": "male", "pavoque": "male", "kerstin": "female", "eva_k": "female", "ramona": "female",
    "siwis": "female", "upmc": "male", "gilles": "male", "tom": "male", "mls": None,
    "davefx": "male", "carlfm": "male", "sharvard": "male", "claude": "male", "ald": "male",
    "paola": "female", "riccardo": "male", "faber": "male", "cadu": "male", "jeff": "male",
    "denis": "male", "dmitri": "male", "irina": "female", "ruslan": "male", "lada": "female",
    "dfki": "male", "fettah": "male", "dimitar": "male", "kareem": "male", "amir": "male", "gyro": "male",
    "huayan": "female", "nathalie": "female", "mihai": "male", "fahrettin": "male", "pratham": "male",
    "priyamvada": "female", "rohan": "male", "arjun": "male", "meera": "female", "venkatesh": "male",
    "maya": "female", "padmavathi": "female", "anna": "female", "berta": "female", "imre": "male",
    "nst": None, "talesyntese": "male", "mc_speech": "female", "darkman": "male", "jirka": "male",
}
# Defaults per language when several voices exist (otherwise: best quality first)
_PREFERRED = {"en": "en_US-lessac-high", "de": "de_DE-thorsten-high", "fr": "fr_FR-siwis-medium",
              "es": "es_ES-davefx-medium", "it": "it_IT-paola-medium", "ru": "ru_RU-irina-medium",
              "pt": "pt_BR-faber-medium", "nl": "nl_NL-mls-medium", "zh": "zh_CN-huayan-medium"}


# Piper languages whose phonemizer needs packages beyond espeak (piper/voice.py: JAPANESE,
# THAI and PINYIN phoneme types). Without them synthesis raises at runtime.
_FRONTEND_DEPS = {
    "ja": ("pyopenjtalk",),
    "th": ("tltk", "unicode_rbnf"),
    "zh": ("unicode_rbnf", "transformers", "sentence_stream"),
}


def _frontend_available(lang: str) -> bool:
    import importlib.util

    return all(importlib.util.find_spec(m) is not None for m in _FRONTEND_DEPS.get(lang, ()))


class PiperTTS(TTSEngine):
    name: ClassVar[str] = "piper"

    def __init__(self, model: Optional[str] = None, *, use_cuda: bool = False, catalog_url: str = _HF + "voices.json") -> None:
        super().__init__()
        import piper  # noqa: F401

        self.use_cuda = use_cuda
        self.catalog_url = catalog_url
        self._catalog: Optional[dict] = None
        self._loaded: dict[str, object] = {}

    # ---- catalog ----------------------------------------------------------
    @property
    def catalog(self) -> dict:
        if self._catalog is None:
            path = cache_dir("piper") / "voices.json"
            if not path.exists() or time.time() - path.stat().st_mtime > 7 * 24 * 3600:
                try:
                    with urllib.request.urlopen(self.catalog_url, timeout=20) as r:  # noqa: S310
                        path.write_bytes(r.read())
                except Exception:
                    if not path.exists():
                        raise
            self._catalog = json.loads(path.read_text())
        return self._catalog

    def list_voices(self) -> list[Voice]:
        out = []
        for key, meta in self.catalog.items():
            lang = meta.get("language", {})
            code = lang.get("code", key.split("-")[0])
            if not _frontend_available(lang.get("family") or code.split("_")[0]):
                continue  # phonemizer deps missing: let ja/zh/th route to another engine
            speaker = meta.get("name", key.split("-")[1] if "-" in key else key)
            quality = meta.get("quality")
            sr = {"x_low": 16000, "low": 16000, "medium": 22050, "high": 22050}.get(quality)
            n_spk = meta.get("num_speakers", 1)
            out.append(Voice(
                id=f"piper:{key}",
                name=speaker.replace("_", " ").title(),
                engine=self.name,
                languages=[normalize_language(lang.get("family") or code)],
                gender=_GENDER.get(speaker),
                description=f"{lang.get('name_english', code)} ({lang.get('country_english', '')})"
                            + (f", {n_spk} speakers (use #id)" if n_spk > 1 else ""),
                sample_rate=sr,
                quality=quality,
                tags=[code],
            ))
        preferred = set(_PREFERRED.values())
        out.sort(key=lambda v: (v.languages[0], v.native_id not in preferred, _QUALITY_RANK.get(v.quality or "", 9)))
        return out

    def get_voice(self, voice_id: str) -> Optional[Voice]:
        base, _, speaker = voice_id.partition("#")
        v = super().get_voice(base)
        if v is None or not speaker:
            return v
        return Voice(**{**v.to_dict(), "id": f"{v.id}#{speaker}", "name": f"{v.name} #{speaker}"})

    def default_voice(self, language: Optional[str] = None, gender: Optional[str] = None) -> Optional[Voice]:
        lang = normalize_language(language) if language else None
        if lang in _PREFERRED and not gender:
            v = self.get_voice(_PREFERRED[lang])
            if v:
                return v
        # sorted list already puts preferred/high-quality first; skip base-class re-ranking
        cands = self.voices(lang, gender) or (self.voices(lang) if gender else [])
        return cands[0] if cands else None

    # ---- synthesis ----------------------------------------------------------
    def _load(self, key: str):
        if key not in self._loaded:
            from piper import PiperVoice

            meta = self.catalog[key]
            d = cache_dir("piper")
            onnx_path = None
            for rel in meta["files"]:
                if rel.endswith((".onnx", ".onnx.json")):
                    p = download(_HF + rel, d / rel.rsplit("/", 1)[-1])
                    if rel.endswith(".onnx"):
                        onnx_path = p
            self._loaded[key] = PiperVoice.load(str(onnx_path), use_cuda=self.use_cuda)
        return self._loaded[key]

    def _synthesize(self, text: str, voice: Voice, language: str, speed: float, **options) -> Audio:
        import numpy as np
        from piper import SynthesisConfig

        key, _, speaker = voice.native_id.partition("#")
        pv = self._load(key)
        cfg = SynthesisConfig(speaker_id=int(speaker) if speaker else None, length_scale=1.0 / max(0.25, speed))
        chunks = list(pv.synthesize(text, syn_config=cfg))
        if not chunks:
            return Audio(np.zeros(0, dtype=np.float32), pv.config.sample_rate)
        return Audio(np.concatenate([c.audio_float_array for c in chunks]), chunks[0].sample_rate)
