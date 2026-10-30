"""Shared logic for zero-shot voice-cloning engines (reference clip -> voice)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from ..audio import Audio, AudioInput
from ..languages import normalize_language
from ..types import Voice
from ..utils import cache_dir


class ClonedVoiceStore:
    """Persists reference clips under ``~/.cache/cachos/voices/<engine>/`` so cloned voices survive restarts."""

    def __init__(self, engine: str) -> None:
        self.engine = engine
        self.dir = cache_dir("voices", engine)

    def save(self, name: str, reference: AudioInput, languages: list[str], sample_rate: int = 24000,
             transcript: Optional[str] = None) -> tuple[Voice, Path]:
        audio = Audio.load(reference, sample_rate)
        if audio.duration < 3.0:
            raise ValueError("reference clip should be at least 3 s (5-15 s of clean speech is ideal)")
        if audio.duration > 30.0:
            audio = Audio(audio.samples[: 30 * sample_rate], sample_rate)
        slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.lower())
        path = self.dir / f"{slug}.wav"
        audio.save(path)
        meta = {"name": name, "languages": languages, "transcript": transcript,
                "sha1": hashlib.sha1(audio.to_pcm16()).hexdigest()}
        path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False))
        return self._voice(slug, meta), path

    def load_all(self) -> dict[str, tuple[Voice, Path]]:
        out = {}
        for meta_path in self.dir.glob("*.json"):
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            wav = meta_path.with_suffix(".wav")
            if wav.exists():
                v = self._voice(meta_path.stem, meta)
                out[v.id] = (v, wav)
        return out

    def _voice(self, slug: str, meta: dict) -> Voice:
        return Voice(id=f"{self.engine}:{slug}", name=meta.get("name", slug), engine=self.engine,
                     languages=[normalize_language(l) for l in meta.get("languages", ["*"])],
                     description="cloned voice", cloned=True, quality="high")
