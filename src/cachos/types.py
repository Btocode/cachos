"""Engine-agnostic result and metadata types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

Gender = Literal["female", "male", "neutral"]


@dataclass
class Word:
    text: str
    start: float
    end: float
    confidence: Optional[float] = None


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)
    language: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class Transcript:
    """Result of a transcription. ``language`` is an ISO 639-1 code where possible."""

    text: str
    language: Optional[str] = None
    language_probability: Optional[float] = None
    segments: list[Segment] = field(default_factory=list)
    duration: Optional[float] = None
    engine: Optional[str] = None
    is_final: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Voice:
    """A selectable voice.

    ``id`` is globally unique and always ``"<engine>:<native_id>"``, so any voice
    id can be passed straight to :class:`cachos.TTS` and it routes itself.
    """

    id: str
    name: str
    engine: str
    languages: list[str]
    gender: Optional[Gender] = None
    description: str = ""
    sample_rate: Optional[int] = None
    quality: Optional[str] = None  # e.g. "low" | "medium" | "high"
    cloned: bool = False
    tags: list[str] = field(default_factory=list)

    @property
    def native_id(self) -> str:
        return self.id.split(":", 1)[1] if ":" in self.id else self.id

    def supports(self, language: str) -> bool:
        from .languages import normalize_language

        lang = normalize_language(language)
        return "*" in self.languages or lang in self.languages

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
