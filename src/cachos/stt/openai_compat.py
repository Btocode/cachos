"""OpenAI-compatible transcription API.

Works with OpenAI (``whisper-1``, ``gpt-4o-transcribe``), Groq
(``whisper-large-v3`` by default for accuracy, or ``whisper-large-v3-turbo``; both very fast), and self-hosted servers that speak the
same protocol (speaches, LocalAI, vLLM, another ``cachos serve``).
"""

from __future__ import annotations

import os
from typing import ClassVar, Optional

from ..audio import Audio
from ..languages import normalize_language
from ..types import Segment, Transcript
from .base import STTEngine


class OpenAISTT(STTEngine):
    name: ClassVar[str] = "openai"
    sample_rate: ClassVar[int] = 16000
    default_base_url: ClassVar[str] = "https://api.openai.com/v1"
    api_key_env: ClassVar[str] = "OPENAI_API_KEY"

    def __init__(self, model: str = "whisper-1", *, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, timeout: float = 60.0) -> None:
        super().__init__()
        import httpx

        self.model = model
        self.base_url = (base_url or os.environ.get(f"{self.name.upper()}_BASE_URL") or self.default_base_url).rstrip("/")
        key = api_key or os.environ.get(self.api_key_env, "")
        self._client = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {key}"} if key else {})

    def _transcribe(self, audio: Audio, language: Optional[str], **options) -> Transcript:
        verbose = not self.model.startswith("gpt-4o")  # gpt-4o-*-transcribe only supports json/text
        data = {"model": self.model, "response_format": "verbose_json" if verbose else "json"}
        if language:
            data["language"] = language
        if "initial_prompt" in options:
            data["prompt"] = options.pop("initial_prompt")
        r = self._client.post(f"{self.base_url}/audio/transcriptions", data=data,
                              files={"file": ("audio.wav", audio.to_wav_bytes(), "audio/wav")})
        if r.status_code >= 400:
            raise RuntimeError(f"{self.name} transcription failed ({r.status_code}): {r.text[:300]}")
        j = r.json()
        segs = [Segment(s.get("text", "").strip(), s.get("start", 0.0), s.get("end", 0.0)) for s in j.get("segments") or []]
        return Transcript(text=j.get("text", "").strip(), language=normalize_language(j.get("language")) or language,
                          segments=segs, duration=j.get("duration"), engine=f"{self.name}:{self.model}")


class GroqSTT(OpenAISTT):
    name: ClassVar[str] = "groq"
    default_base_url: ClassVar[str] = "https://api.groq.com/openai/v1"
    api_key_env: ClassVar[str] = "GROQ_API_KEY"

    def __init__(self, model: str = "whisper-large-v3", **kw) -> None:
        super().__init__(model, **kw)
