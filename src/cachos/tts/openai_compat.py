"""OpenAI-compatible speech API (OpenAI ``gpt-4o-mini-tts``/``tts-1``, or any
server that implements ``POST /audio/speech``, e.g. Kokoro-FastAPI, speaches,
another ``cachos serve``)."""

from __future__ import annotations

import os
from typing import ClassVar, Optional

from ..audio import Audio
from ..types import Voice
from .base import TTSEngine

_OPENAI_VOICES = {"alloy": "female", "ash": "male", "ballad": "male", "coral": "female", "echo": "male",
                  "fable": "male", "nova": "female", "onyx": "male", "sage": "female", "shimmer": "female",
                  "verse": "male", "marin": "female", "cedar": "male"}


class OpenAITTS(TTSEngine):
    name: ClassVar[str] = "openai"

    def __init__(self, model: str = "gpt-4o-mini-tts", *, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, voices: Optional[list[str]] = None,
                 instructions: Optional[str] = None, timeout: float = 60.0) -> None:
        super().__init__()
        import httpx

        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.voice_names = voices or list(_OPENAI_VOICES)
        self.instructions = instructions
        self._http = httpx.AsyncClient(timeout=timeout, headers={"Authorization": f"Bearer {key}"} if key else {})
        self._sync = httpx.Client(timeout=timeout, headers=self._http.headers)

    def list_voices(self) -> list[Voice]:
        # OpenAI voices are multilingual: they speak whatever language the text is in
        return [Voice(id=f"openai:{n}", name=n.title(), engine=self.name, languages=["en", "*"],
                      gender=_OPENAI_VOICES.get(n), description=f"{self.model}", sample_rate=24000, quality="high")
                for n in self.voice_names]

    def _body(self, text: str, voice: Voice, speed: float) -> dict:
        body = {"model": self.model, "input": text, "voice": voice.native_id, "response_format": "wav", "speed": speed}
        if self.instructions:
            body["instructions"] = self.instructions
        return body

    async def synthesize(self, text: str, voice=None, language=None, speed: float = 1.0, **options) -> Audio:
        v, _ = self.resolve(voice, language, options.pop("gender", None))
        r = await self._http.post(f"{self.base_url}/audio/speech", json=self._body(text, v, speed))
        if r.status_code >= 400:
            raise RuntimeError(f"openai tts failed ({r.status_code}): {r.text[:300]}")
        return Audio.load(r.content)

    def _synthesize(self, text: str, voice: Voice, language: str, speed: float, **options) -> Audio:
        r = self._sync.post(f"{self.base_url}/audio/speech", json=self._body(text, voice, speed))
        if r.status_code >= 400:
            raise RuntimeError(f"openai tts failed ({r.status_code}): {r.text[:300]}")
        return Audio.load(r.content)
