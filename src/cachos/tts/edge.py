"""Microsoft Edge read-aloud voices via ``edge-tts``.

400+ neural voices across 100+ locales, free, no API key, no GPU, but it
needs network and uses an unofficial endpoint (fine for prototyping and as
the widest-coverage fallback; use a local engine for guaranteed availability).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import ClassVar, Optional

from ..audio import Audio
from ..languages import normalize_language
from ..types import Voice
from ..utils import cache_dir
from .base import TTSEngine

_VOICE_CACHE_TTL = 7 * 24 * 3600


def _run_coro_blocking(coro):
    """Run a coroutine from sync code, even if the caller's thread has a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result: dict = {}

    def target() -> None:
        try:
            result["v"] = asyncio.run(coro)
        except BaseException as e:  # noqa: BLE001
            result["e"] = e

    t = threading.Thread(target=target)
    t.start()
    t.join()
    if "e" in result:
        raise result["e"]
    return result["v"]


class EdgeTTS(TTSEngine):
    name: ClassVar[str] = "edge"

    def __init__(self, model: Optional[str] = None, *, pitch: str = "+0Hz", volume: str = "+0%") -> None:
        super().__init__()
        import edge_tts  # noqa: F401

        self.pitch = pitch
        self.volume = volume

    def list_voices(self) -> list[Voice]:
        path = cache_dir() / "edge_voices.json"
        raw = None
        if path.exists() and time.time() - path.stat().st_mtime < _VOICE_CACHE_TTL:
            raw = json.loads(path.read_text())
        if raw is None:
            import edge_tts

            try:
                raw = _run_coro_blocking(edge_tts.list_voices())
                path.write_text(json.dumps(raw))
            except Exception:
                if not path.exists():
                    raise
                raw = json.loads(path.read_text())  # stale cache beats nothing when offline
        voices = []
        for v in raw:
            locale = v.get("Locale", "")
            tag = v.get("VoiceTag") or {}
            personality = ", ".join(tag.get("VoicePersonalities") or [])
            gender = (v.get("Gender") or "").lower() or None
            multilingual = "Multilingual" in v["ShortName"]
            voices.append(Voice(
                id=f"edge:{v['ShortName']}",
                name=v["ShortName"].split("-", 2)[-1].replace("Neural", ""),
                engine=self.name,
                languages=[normalize_language(locale)] + (["*"] if multilingual else []),
                gender=gender if gender in ("female", "male") else None,
                description=f"{locale} {personality}".strip(),
                sample_rate=24000,
                quality="high",
                tags=[locale, *(tag.get("ContentCategories") or [])],
            ))
        # Put "neutral locale" voices first per language, e.g. en-US before en-IN, so defaults are sensible
        preferred = {"en": "en-US", "es": "es-ES", "fr": "fr-FR", "pt": "pt-BR", "de": "de-DE", "zh": "zh-CN",
                     "ar": "ar-SA", "it": "it-IT", "nl": "nl-NL"}
        voices.sort(key=lambda vo: (vo.languages[0], 0 if vo.tags and vo.tags[0] == preferred.get(vo.languages[0]) else 1,
                                    "Multilingual" in vo.id))
        return voices

    @staticmethod
    def _rate(speed: float) -> str:
        return f"{round((speed - 1.0) * 100):+d}%"

    async def _fetch(self, text: str, voice: Voice, speed: float) -> Audio:
        import edge_tts

        comm = edge_tts.Communicate(text, voice.native_id, rate=self._rate(speed), volume=self.volume, pitch=self.pitch)
        mp3 = bytearray()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                mp3.extend(chunk["data"])
        if not mp3:
            raise RuntimeError(f"edge-tts returned no audio for voice {voice.native_id}")
        return await asyncio.to_thread(Audio.load, bytes(mp3))

    async def synthesize(self, text: str, voice=None, language=None, speed: float = 1.0, **options) -> Audio:
        v, _lang = self.resolve(voice, language, options.pop("gender", None))
        return await self._fetch(text, v, speed)

    def _synthesize(self, text: str, voice: Voice, language: str, speed: float, **options) -> Audio:
        return _run_coro_blocking(self._fetch(text, voice, speed))
