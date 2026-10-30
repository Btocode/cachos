"""Meta MMS-TTS via 🤗 transformers – one VITS voice for each of 1100+ languages.

The long tail fallback: Yoruba, Quechua, Tigrinya, Khmer, Lao… Voices are
addressed by ISO 639-3 code: ``"mms:yor"``, ``"mms:quz"``; ISO 639-1 also works
(``"mms:sw"``). Some non-Latin scripts need ``pip install uroman``.

License warning: MMS weights are CC-BY-NC 4.0 (non-commercial).
"""

from __future__ import annotations

from typing import ClassVar, Optional

from ..audio import Audio
from ..languages import language_name, normalize_language, to_iso3
from ..types import Voice
from .base import TTSEngine


# advertised in the catalog; any other ISO 639-3 code also works
_COMMON = ("en fr de es it pt ru zh ja ko ar hi bn ur tr vi th nl pl uk cs ro hu el he fa id ms sw ta te "
           "mr gu kn ml pa ne si my km lo am yo ig ha zu xh so tl").split()


class MMSTTS(TTSEngine):
    name: ClassVar[str] = "mms"

    def __init__(self, model: Optional[str] = None, *, device: str = "cpu") -> None:
        super().__init__()
        import transformers  # noqa: F401

        self.device = device
        self._models: dict[str, tuple] = {}

    def _voice_for(self, iso3: str) -> Voice:
        iso1 = normalize_language(iso3) or iso3
        return Voice(id=f"mms:{iso3}", name=language_name(iso1), engine=self.name, languages=[iso1],
                     description=f"MMS VITS ({iso3})", sample_rate=16000, quality="low")

    def list_voices(self) -> list[Voice]:
        # Advertise the common languages; any other ISO 639-3 code works via get_voice / language=
        return [self._voice_for(to_iso3(c)) for c in _COMMON]

    def get_voice(self, voice_id: str) -> Optional[Voice]:
        code = voice_id.split(":", 1)[-1]
        return self._voice_for(to_iso3(code))

    def default_voice(self, language: Optional[str] = None, gender: Optional[str] = None) -> Optional[Voice]:
        return self._voice_for(to_iso3(language or "en"))

    def _load(self, iso3: str):
        if iso3 not in self._models:
            from transformers import AutoTokenizer, VitsModel

            repo = f"facebook/mms-tts-{iso3}"
            try:
                model = VitsModel.from_pretrained(repo).to(self.device)
                tok = AutoTokenizer.from_pretrained(repo)
            except OSError as e:
                raise ValueError(f"no MMS voice for language {iso3!r} ({repo} not found)") from e
            self._models[iso3] = (model, tok)
        return self._models[iso3]

    def _synthesize(self, text: str, voice: Voice, language: str, speed: float, **options) -> Audio:
        import torch

        iso3 = voice.native_id
        model, tok = self._load(iso3)
        if getattr(tok, "is_uroman", False):
            try:
                import uroman

                text = uroman.Uroman().romanize_string(text)
            except ImportError as e:
                raise ImportError(f"MMS voice {iso3} needs romanized input: pip install uroman") from e
        model.speaking_rate = speed
        inputs = tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            wav = model(**inputs).waveform
        return Audio(wav.squeeze().cpu().numpy(), model.config.sampling_rate)
