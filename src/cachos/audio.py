"""Audio primitives: a single in-memory format (mono float32 numpy) plus I/O.

Every engine in cachos consumes and produces :class:`Audio`. Conversions to and
from PCM16 bytes, WAV, arbitrary files (via soundfile or ffmpeg) and resampling
live here so engines never have to care.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

AudioInput = Union["Audio", str, os.PathLike, bytes, np.ndarray]


@dataclass
class Audio:
    """Mono float32 samples in [-1, 1] at ``sample_rate`` Hz."""

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        s = np.asarray(self.samples)
        if s.ndim == 2:  # (n, channels) or (channels, n) -> mono
            s = s.mean(axis=1 if s.shape[0] > s.shape[1] else 0)
        if s.dtype == np.int16:
            s = s.astype(np.float32) / 32768.0
        elif s.dtype != np.float32:
            s = s.astype(np.float32)
        self.samples = s

    # ---- properties -------------------------------------------------------
    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate) if self.sample_rate else 0.0

    def __len__(self) -> int:
        return len(self.samples)

    # ---- constructors -----------------------------------------------------
    @classmethod
    def silence(cls, seconds: float, sample_rate: int = 16000) -> "Audio":
        return cls(np.zeros(int(seconds * sample_rate), dtype=np.float32), sample_rate)

    @classmethod
    def from_pcm16(cls, data: bytes, sample_rate: int) -> "Audio":
        return cls(np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0, sample_rate)

    @classmethod
    def from_wav_bytes(cls, data: bytes) -> "Audio":
        with wave.open(io.BytesIO(data), "rb") as w:
            sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width == 2:
            arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 4:
            arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        elif width == 1:
            arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"unsupported WAV sample width: {width}")
        if ch > 1:
            arr = arr.reshape(-1, ch).mean(axis=1)
        return cls(arr, sr)

    @classmethod
    def load(cls, source: AudioInput, sample_rate: int | None = None) -> "Audio":
        """Load from an Audio, path, encoded bytes (wav/mp3/ogg/...), or ndarray.

        A bare ndarray is assumed to be 16 kHz unless ``sample_rate`` is given.
        If ``sample_rate`` is given the result is resampled to it.
        """
        if isinstance(source, Audio):
            audio = source
        elif isinstance(source, np.ndarray):
            audio = cls(source, sample_rate or 16000)
        elif isinstance(source, (bytes, bytearray)):
            audio = _decode_bytes(bytes(source))
        else:
            audio = _decode_path(Path(source))
        return audio.resample(sample_rate) if sample_rate else audio

    # ---- transforms -------------------------------------------------------
    def resample(self, target_rate: int) -> "Audio":
        if target_rate == self.sample_rate:
            return self
        if len(self.samples) == 0:
            return Audio(self.samples, target_rate)
        try:
            import soxr

            out = soxr.resample(self.samples, self.sample_rate, target_rate)
        except ImportError:
            n_out = int(round(len(self.samples) * target_rate / self.sample_rate))
            x_old = np.linspace(0.0, 1.0, num=len(self.samples), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            out = np.interp(x_new, x_old, self.samples)
        return Audio(np.asarray(out, dtype=np.float32), target_rate)

    def __add__(self, other: "Audio") -> "Audio":
        other = other.resample(self.sample_rate)
        return Audio(np.concatenate([self.samples, other.samples]), self.sample_rate)

    @staticmethod
    def concat(parts: list["Audio"], sample_rate: int | None = None) -> "Audio":
        if not parts:
            return Audio(np.zeros(0, dtype=np.float32), sample_rate or 16000)
        sr = sample_rate or parts[0].sample_rate
        return Audio(np.concatenate([p.resample(sr).samples for p in parts]), sr)

    # ---- encoders ---------------------------------------------------------
    def to_pcm16(self) -> bytes:
        return (np.clip(self.samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    def to_wav_bytes(self) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(self.to_pcm16())
        return buf.getvalue()

    def encode(self, fmt: str = "wav") -> bytes:
        """Encode to ``wav`` | ``pcm`` (raw s16le) | ``mp3`` | ``opus`` | ``flac`` | ``aac``."""
        fmt = fmt.lower()
        if fmt == "wav":
            return self.to_wav_bytes()
        if fmt in ("pcm", "raw", "s16le"):
            return self.to_pcm16()
        return _ffmpeg_encode(self, fmt)

    def save(self, path: str | os.PathLike) -> Path:
        p = Path(path)
        p.write_bytes(self.encode(p.suffix.lstrip(".") or "wav"))
        return p


# --------------------------------------------------------------------------
# decoding helpers
# --------------------------------------------------------------------------
def _decode_path(path: Path) -> Audio:
    if path.suffix.lower() == ".wav":
        try:
            return Audio.from_wav_bytes(path.read_bytes())
        except (wave.Error, ValueError):
            pass  # e.g. float WAV: fall through
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return Audio(data, sr)
    except Exception:
        return _ffmpeg_decode(path.read_bytes())


def _decode_bytes(data: bytes) -> Audio:
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        try:
            return Audio.from_wav_bytes(data)
        except (wave.Error, ValueError):
            pass
    try:
        import soundfile as sf

        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return Audio(arr, sr)
    except Exception:
        return _ffmpeg_decode(data)


def _require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg or `pip install cachos[audio]` to decode this format."
        )
    return exe


def _ffmpeg_decode(data: bytes, sample_rate: int = 16000) -> Audio:
    proc = subprocess.run(
        [_require_ffmpeg(), "-nostdin", "-loglevel", "error", "-i", "pipe:0",
         "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "pipe:1"],
        input=data, capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise ValueError(f"ffmpeg could not decode audio: {proc.stderr.decode(errors='ignore')[:300]}")
    return Audio.from_pcm16(proc.stdout, sample_rate)


_FFMPEG_FORMATS = {
    "mp3": ["-f", "mp3", "-c:a", "libmp3lame", "-b:a", "128k"],
    "opus": ["-f", "ogg", "-c:a", "libopus", "-b:a", "48k"],
    "ogg": ["-f", "ogg", "-c:a", "libopus", "-b:a", "48k"],
    "flac": ["-f", "flac"],
    "aac": ["-f", "adts", "-c:a", "aac", "-b:a", "128k"],
}


def _ffmpeg_encode(audio: Audio, fmt: str) -> bytes:
    if fmt not in _FFMPEG_FORMATS:
        raise ValueError(f"unsupported output format {fmt!r}; use one of wav, pcm, {', '.join(_FFMPEG_FORMATS)}")
    proc = subprocess.run(
        [_require_ffmpeg(), "-nostdin", "-loglevel", "error", "-f", "s16le", "-ac", "1",
         "-ar", str(audio.sample_rate), "-i", "pipe:0", *_FFMPEG_FORMATS[fmt], "pipe:1"],
        input=audio.to_pcm16(), capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed: {proc.stderr.decode(errors='ignore')[:300]}")
    return proc.stdout
