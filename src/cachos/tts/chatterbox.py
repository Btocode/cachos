"""Resemble AI Chatterbox Multilingual (``chatterbox-tts``) – MIT licensed,
23 languages, zero-shot voice cloning, emotion-exaggeration control.
GPU recommended (runs on CPU slowly).
"""

from __future__ import annotations

from typing import ClassVar, Optional

from ..audio import Audio, AudioInput
from ..languages import normalize_language
from ..types import Voice
from ._cloning import ClonedVoiceStore
from .base import TTSEngine

CHATTERBOX_LANGUAGES = ["ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms",
                        "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"]


class ChatterboxTTS(TTSEngine):
    name: ClassVar[str] = "chatterbox"
    supports_cloning: ClassVar[bool] = True

    def __init__(self, model: str = "multilingual", *, device: str = "auto", exaggeration: float = 0.5,
                 cfg_weight: float = 0.5) -> None:
        """
        Args:
            model: ``"multilingual"`` (23 languages) or ``"english"`` (original English model).
            exaggeration: 0.25-2.0, emotional intensity (higher = more dramatic).
            cfg_weight: 0-1, lower gives slower, more deliberate pacing.
        """
        super().__init__()
        import chatterbox  # noqa: F401

        self.variant = model
        self.device = device
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self._model = None
        self._store = ClonedVoiceStore(self.name)
        loaded = self._store.load_all()
        self._refs = {vid: path for vid, (_v, path) in loaded.items()}
        self._cloned = {vid: v for vid, (v, _p) in loaded.items()}

    @property
    def model(self):
        if self._model is None:
            import torch

            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            if self.variant == "english":
                from chatterbox.tts import ChatterboxTTS as Model
            else:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Model
            self._model = Model.from_pretrained(device=device)
        return self._model

    def list_voices(self) -> list[Voice]:
        langs = ["en"] if self.variant == "english" else list(CHATTERBOX_LANGUAGES)
        return [Voice(id="chatterbox:default", name="Default", engine=self.name, languages=langs, gender="female",
                      description="built-in reference voice; clone_voice() for more", sample_rate=24000, quality="high")]

    def clone_voice(self, name: str, reference: AudioInput, language: Optional[str] = None,
                    transcript: Optional[str] = None) -> Voice:
        langs = ["en"] if self.variant == "english" else list(CHATTERBOX_LANGUAGES)
        v, path = self._store.save(name, reference, langs, 24000, transcript)
        self._cloned[v.id] = v
        self._refs[v.id] = path
        return v

    def _synthesize(self, text: str, voice: Voice, language: str, speed: float, **options) -> Audio:
        kw = dict(exaggeration=options.pop("exaggeration", self.exaggeration),
                  cfg_weight=options.pop("cfg_weight", self.cfg_weight), **options)
        if voice.id in self._refs:
            kw["audio_prompt_path"] = str(self._refs[voice.id])
        if self.variant != "english":
            lang = normalize_language(language) or "en"
            if lang not in CHATTERBOX_LANGUAGES:
                raise ValueError(f"Chatterbox does not support language {lang!r}")
            kw["language_id"] = lang
        wav = self.model.generate(text, **kw)
        # Chatterbox has no speed control; pacing is steered with cfg_weight instead
        return Audio(wav.squeeze().detach().cpu().numpy(), self.model.sr)
