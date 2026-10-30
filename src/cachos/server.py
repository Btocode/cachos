"""HTTP + WebSocket server.

* OpenAI-compatible: ``POST /v1/audio/transcriptions``, ``POST /v1/audio/speech``
  (existing OpenAI SDK clients work by changing ``base_url``).
* ``GET /v1/voices``, ``GET /v1/languages`` – the merged voice catalog.

Real-time voice agents are served by Pipecat's runner instead (WebRTC browser
client, telephony WebSockets): ``cachos agent`` / :meth:`cachos.VoiceAgent.serve`.

Run: ``cachos serve`` or ``uvicorn`` on :func:`create_app`.
"""


import logging
from typing import Optional

from pydantic import BaseModel

from .audio import Audio
from .core import STT, TTS
from .languages import language_name

log = logging.getLogger("cachos.server")

# OpenAI voice names -> gender hint so OpenAI-SDK clients get something sensible
_OPENAI_VOICES = {"alloy": "female", "ash": "male", "ballad": "male", "coral": "female", "echo": "male",
                  "fable": "male", "nova": "female", "onyx": "male", "sage": "female", "shimmer": "female",
                  "verse": "male"}


class SpeechRequest(BaseModel):
    """OpenAI ``/v1/audio/speech`` body plus cachos extras (language, gender, stream)."""

    input: str
    model: Optional[str] = None
    voice: Optional[str] = None
    language: Optional[str] = None
    gender: Optional[str] = None
    response_format: str = "mp3"
    speed: float = 1.0
    stream: bool = False


_MEDIA = {"mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac", "flac": "audio/flac",
          "wav": "audio/wav", "pcm": "audio/L16"}


def create_app(
    stt: Optional[STT] = None,
    tts: Optional[TTS] = None,
    *,
    output_sample_rate: int = 24000,
    api_key: Optional[str] = None,
):
    """Build the FastAPI app."""
    from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    
    app = FastAPI(title="cachos", version="0.1.0", description="Multilingual STT/TTS and voice agents")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    state: dict = {"stt": stt, "tts": tts}
    bearer = HTTPBearer(auto_error=False)

    def get_stt() -> STT:
        if state["stt"] is None:
            state["stt"] = STT()
        return state["stt"]

    def get_tts() -> TTS:
        if state["tts"] is None:
            state["tts"] = TTS()
        return state["tts"]

    def auth(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> None:
        if api_key and (creds is None or creds.credentials != api_key):
            raise HTTPException(401, "invalid api key")

    # ---- meta -------------------------------------------------------------------
    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/v1/voices", dependencies=[Depends(auth)])
    def voices(language: Optional[str] = None, gender: Optional[str] = None, engine: Optional[str] = None) -> dict:
        vs = get_tts().voices(language, gender, engine)
        return {"object": "list", "data": [v.to_dict() for v in vs]}

    @app.get("/v1/languages", dependencies=[Depends(auth)])
    def languages() -> dict:
        return {"tts": [{"code": c, "name": language_name(c)} for c in get_tts().languages]}

    # ---- STT ----------------------------------------------------------------------
    @app.post("/v1/audio/transcriptions", dependencies=[Depends(auth)])
    async def transcriptions(
        file: UploadFile = File(...),
        model: Optional[str] = Form(None),
        language: Optional[str] = Form(None),
        response_format: str = Form("json"),
        prompt: Optional[str] = Form(None),
    ):
        data = await file.read()
        try:
            audio = Audio.load(data)
        except Exception as e:
            raise HTTPException(400, f"could not decode audio: {e}")
        opts = {"initial_prompt": prompt} if prompt else {}
        t = await get_stt().transcribe(audio, language, **opts)
        if response_format == "text":
            return PlainTextResponse(t.text)
        if response_format in ("srt", "vtt"):
            return PlainTextResponse(_subtitles(t, response_format))
        if response_format == "verbose_json":
            return JSONResponse({"task": "transcribe", "language": t.language, "duration": t.duration,
                                 "text": t.text, "language_probability": t.language_probability,
                                 "segments": [s.__dict__ | {"words": [w.__dict__ for w in s.words]}
                                              for s in t.segments]})
        return {"text": t.text, "language": t.language}

    # ---- TTS ----------------------------------------------------------------------
    @app.post("/v1/audio/speech", dependencies=[Depends(auth)])
    async def speech(req: SpeechRequest):
        tts_ = get_tts()
        voice, gender = req.voice, req.gender
        if voice and ":" not in voice:  # OpenAI-style voice name, or "auto"
            gender = gender or _OPENAI_VOICES.get(voice.lower())
            voice = None
        fmt = req.response_format.lower()
        if fmt not in _MEDIA:
            raise HTTPException(400, f"response_format must be one of {list(_MEDIA)}")
        try:
            if req.stream:
                async def gen():
                    if fmt == "wav":  # streaming WAV: one header with "unknown" length, then raw PCM
                        yield _streaming_wav_header(output_sample_rate)
                    async for chunk in tts_.stream(req.input, voice, req.language, gender=gender, speed=req.speed):
                        a = chunk.resample(output_sample_rate)
                        yield a.to_pcm16() if fmt in ("pcm", "wav") else a.encode(fmt)

                headers = {"X-Sample-Rate": str(output_sample_rate)}
                return StreamingResponse(gen(), media_type=_MEDIA[fmt], headers=headers)
            audio = await tts_.synthesize(req.input, voice, req.language, gender=gender, speed=req.speed)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if fmt == "pcm":
            audio = audio.resample(output_sample_rate)
        return Response(audio.encode(fmt), media_type=_MEDIA[fmt],
                        headers={"X-Sample-Rate": str(audio.sample_rate)})

    return app


def _streaming_wav_header(sample_rate: int) -> bytes:
    import struct

    size = 0xFFFFFFFF
    return (b"RIFF" + struct.pack("<I", size) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
            sample_rate * 2, 2, 16) + b"data" + struct.pack("<I", size))


def _subtitles(t, fmt: str) -> str:
    def ts(s: float) -> str:
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        sep = "," if fmt == "srt" else "."
        return f"{int(h):02d}:{int(m):02d}:{int(sec):02d}{sep}{int((sec % 1) * 1000):03d}"

    lines = ["WEBVTT", ""] if fmt == "vtt" else []
    for i, s in enumerate(t.segments, 1):
        if fmt == "srt":
            lines.append(str(i))
        lines += [f"{ts(s.start)} --> {ts(s.end)}", s.text.strip(), ""]
    return "\n".join(lines)
