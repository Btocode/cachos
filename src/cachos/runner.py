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
          proxy: Optional[str] = None) -> None:
    from pipecat.runner.run import main

    sys.modules["__main__"].bot = agent.bot  # type: ignore[attr-defined]
    argv = [sys.argv[0], "--host", host, "--port", str(port)]
    if transport:
        argv += ["-t", transport]
    if proxy:
        argv += ["-x", proxy]
    sys.argv = argv
    main()
