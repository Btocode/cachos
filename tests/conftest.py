from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from cachos import Audio, STTEngine, TTSEngine, Transcript, Voice


class FakeSTT(STTEngine):
    name = "fake-stt"

    def __init__(self, text: str = "hello there", language: str = "en") -> None:
        super().__init__()
        self.text, self.language, self.calls = text, language, 0

    def _transcribe(self, audio: Audio, language: Optional[str], **options) -> Transcript:
        self.calls += 1
        return Transcript(text=self.text, language=language or self.language, language_probability=0.99)


class FakeTTS(TTSEngine):
    """Each char -> 10 ms of a tone; voices for the languages given."""

    def __init__(self, name: str = "fake", langs=("en", "fr"), sample_rate: int = 24000) -> None:
        super().__init__()
        self.name = name  # type: ignore[misc]
        self.langs = list(langs)
        self.sr = sample_rate
        self.calls: list[tuple[str, str, str]] = []

    def list_voices(self) -> list[Voice]:
        out = []
        for lang in self.langs:
            out.append(Voice(f"{self.name}:{lang}_f", f"{lang} f", self.name, [lang], "female"))
            out.append(Voice(f"{self.name}:{lang}_m", f"{lang} m", self.name, [lang], "male"))
        return out

    def _synthesize(self, text: str, voice: Voice, language: str, speed: float, **options) -> Audio:
        self.calls.append((text, voice.id, language))
        n = int(self.sr * 0.01 * len(text))
        t = np.arange(n) / self.sr
        return Audio((0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), self.sr)


def tone(seconds: float, sr: int = 16000, amp: float = 0.5, freq: float = 300.0) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    # amplitude-modulated so it looks more like speech energy than a flat tone
    return (amp * np.sin(2 * np.pi * freq * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)


def silence(seconds: float, sr: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(seconds * sr)) * 0.002).astype(np.float32)


@pytest.fixture
def fake_stt():
    return FakeSTT()


@pytest.fixture
def fake_tts():
    return FakeTTS()
