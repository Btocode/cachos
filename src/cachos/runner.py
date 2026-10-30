"""Serve a :class:`cachos.VoiceAgent` with Pipecat's development runner.

Pipecat's runner (FastAPI + uvicorn) provides the SmallWebRTC browser client
at ``/client`` and telephony WebSocket endpoints (Twilio, Telnyx, Plivo,
Exotel). It discovers a ``bot(runner_args)`` coroutine on ``__main__`` and
reads its options from ``sys.argv``, so we provide both.
"""

from __future__ import annotations

import sys
from typing import Optional


def serve(agent, *, host: str = "localhost", port: int = 7860, transport: Optional[str] = "webrtc",
          proxy: Optional[str] = None, test_ui: bool = False) -> None:
    from pipecat.runner.run import main

    if test_ui:
        add_test_ui(agent)
    sys.modules["__main__"].bot = agent.bot  # type: ignore[attr-defined]
    argv = [sys.argv[0], "--host", host, "--port", str(port)]
    if transport:
        argv += ["-t", transport]
    if proxy:
        argv += ["-x", proxy]
    sys.argv = argv
    main()


def add_test_ui(agent) -> None:
    """Mount the cachos test page (``/test``) and its catalog endpoint on Pipecat's runner app."""
    from importlib.resources import files

    from fastapi.responses import HTMLResponse
    from pipecat.runner.run import app

    page = files("cachos").joinpath("web/test.html").read_text(encoding="utf-8")

    @app.get("/test", response_class=HTMLResponse, include_in_schema=False)
    async def test_page() -> str:
        return page

    @app.get("/cachos/config")
    async def cachos_config() -> dict:
        voices = agent.tts.voices()
        return {
            "defaults": {"language": agent.language, "voice": agent.voice, "gender": agent.gender,
                         "system_prompt": agent.system_prompt, "greeting": agent.greeting,
                         "interrupt_min_words": agent.interrupt_min_words},
            "languages": agent.tts.languages,
            "voices": [{"id": v.id, "name": v.name, "languages": v.languages, "gender": v.gender,
                        "engine": v.engine} for v in voices],
            "stt": getattr(agent.stt.engine, "model_name", agent.stt.name),
            "llm": agent.llm_model,
        }
