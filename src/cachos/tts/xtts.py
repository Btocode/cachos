"""Coqui XTTS-v2 (``coqui-tts``, the maintained idiap fork) – 17 languages, 58 studio
speakers, zero-shot cloning from ~6 s of audio, cross-lingual (clone in English,
speak Japanese). GPU recommended (≈2-4 GB VRAM).

License warning: XTTS-v2 weights are under the Coqui Public Model License
(non-commercial). Use Chatterbox (MIT) for commercial cloning.
"""

from __future__ import annotations

from typing import ClassVar, Optional

from ..audio import Audio, AudioInput
from ..languages import normalize_language
from ..types import Voice
from ._cloning import ClonedVoiceStore
from .base import TTSEngine

XTTS_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "hu", "ko", "ja", "hi"]
_FEMALE_HINTS = {"Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema", "Alison Dietlinde", "Ana Florence",
                 "Annmarie Nele", "Asya Anara", "Brenda Stern", "Gitta Nikolina", "Henriette Usha", "Sofia Hellen",
                 "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie", "Nova Hogarth", "Maja Ruoho", "Uta Obando",
                 "Lidiya Szekeres", "Chandra MacFarland", "Szofi Granger", "Camilla Holmström", "Lilya Stainthorpe",
                 "Zofija Kendrick", "Narelle Moon", "Barbora MacLean", "Alexandra Hisakawa", "Alma María",
                 "Rosemary Okafor", "Ige Behringer", "Suad Qasim"}


class XTTS(TTSEngine):
    name: ClassVar[str] = "xtts"
    supports_cloning: ClassVar[bool] = True
    sample_rate: ClassVar[int] = 24000

    def __init__(self, model: str = "tts_models/multilingual/multi-dataset/xtts_v2", *, device: str = "auto") -> None:
        super().__init__()
        import TTS  # noqa: F401  (pip install coqui-tts)

        self.model_name = model
        self.device = device
        self._tts = None
        self._store = ClonedVoiceStore(self.name)
        self._refs = {vid: path for vid, (v, path) in self._store.load_all().items()}
        self._cloned = {vid: v for vid, (v, _p) in self._store.load_all().items()}

    @property
    def tts(self):
        if self._tts is None:
            import torch
            from TTS.api import TTS as CoquiTTS

            device = self.device if self.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            self._tts = CoquiTTS(self.model_name).to(device)
        return self._tts

    def list_voices(self) -> list[Voice]:
        speakers = list(self.tts.speakers or [])
        return [Voice(id=f"xtts:{s}", name=s, engine=self.name, languages=list(XTTS_LANGUAGES),
                      gender="female" if s in _FEMALE_HINTS else "male", description="XTTS-v2 studio speaker",
                      sample_rate=self.sample_rate, quality="high") for s in speakers]

    def clone_voice(self, name: str, reference: AudioInput, language: Optional[str] = None,
                    transcript: Optional[str] = None) -> Voice:
        v, path = self._store.save(name, reference, list(XTTS_LANGUAGES), 24000, transcript)
        self._cloned[v.id] = v
        self._refs[v.id] = path
        return v

    def _synthesize(self, text: str, voice: Voice, language: str, speed: float, **options) -> Audio:
        lang = normalize_language(language) or "en"
        if lang not in XTTS_LANGUAGES:
            raise ValueError(f"XTTS does not support language {lang!r}")
        xlang = "zh-cn" if lang == "zh" else lang
        kw = {"speaker_wav": str(self._refs[voice.id])} if voice.id in self._refs else {"speaker": voice.native_id}
        wav = self.tts.tts(text=text, language=xlang, speed=speed, split_sentences=False, **kw, **options)
        return Audio(wav, self.sample_rate)
