"""Sentence segmentation for streaming TTS.

Uses Microsoft's ``blingfire`` sentence breaker (multilingual, ~µs per
sentence), with the streaming strategy of LiveKit Agents'
``tokenize.blingfire.SentenceTokenizer``: buffer LLM tokens, and once the
buffer holds two or more sentences, emit all but the last (the last may still
be growing). Inside Pipecat pipelines, Pipecat's own TTS text aggregation is
used instead.
"""

from __future__ import annotations

from typing import AsyncIterable, AsyncIterator, Iterable, Iterator


def split_sentences(text: str, min_chars: int = 20) -> list[str]:
    """Split complete text into sentences, merging fragments shorter than ``min_chars``."""
    import blingfire

    parts = [s.strip() for s in blingfire.text_to_sentences(text.strip()).split("\n") if s.strip()]
    out: list[str] = []
    for p in parts:
        if out and len(out[-1]) < min_chars:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


class SentenceStream:
    """Incremental sentence splitter for a token stream."""

    def __init__(self, min_chars: int = 20) -> None:
        self.min_chars = min_chars
        self._buf = ""

    def push(self, token: str) -> list[str]:
        self._buf += token
        sentences = split_sentences(self._buf, self.min_chars)
        if len(sentences) < 2:
            return []
        done, last = sentences[:-1], sentences[-1]
        # keep the (possibly incomplete) last sentence; drop what we emitted from the buffer
        idx = self._buf.rfind(last[: max(1, min(len(last), 12))])
        self._buf = self._buf[idx:] if idx >= 0 else last
        return done

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []


def iter_sentences(tokens: Iterable[str], min_chars: int = 20) -> Iterator[str]:
    s = SentenceStream(min_chars)
    for t in tokens:
        yield from s.push(t)
    yield from s.flush()


async def aiter_sentences(tokens: AsyncIterable[str], min_chars: int = 20) -> AsyncIterator[str]:
    s = SentenceStream(min_chars)
    async for t in tokens:
        for c in s.push(t):
            yield c
    for c in s.flush():
        yield c
