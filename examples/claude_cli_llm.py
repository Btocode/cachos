"""OpenAI-compatible chat endpoint backed by the local Claude Code CLI (for testing).

Lets a cachos VoiceAgent use your logged-in ``claude`` CLI instead of an API key::

    python examples/claude_cli_llm.py --port 8765 &
    cachos chat --llm-base-url http://127.0.0.1:8765/v1 --llm-model haiku

Each request runs ``claude -p`` with streaming JSON output; MCP servers, tools and
local settings are disabled so replies start fast (~2-4 s, mostly CLI startup).
For a full-featured wrapper see github.com/RichardAtCT/claude-code-openai-wrapper.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if isinstance(p, dict))


def _prompt(messages: list[dict]) -> tuple[str, str]:
    """System prompt, and the conversation so far as one prompt for ``claude -p``."""
    system = "\n".join(_text(m["content"]) for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    *history, last = turns or [{"role": "user", "content": ""}]
    lines = [f"{m['role']}: {_text(m['content'])}" for m in history]
    convo = ("Conversation so far:\n" + "\n".join(lines) + "\n\n") if lines else ""
    instruction = " (instruction)" if last["role"] == "developer" else ""
    return system, f"{convo}Reply to this{instruction}:\n{_text(last['content'])}"


async def _claude(model: str, system: str, prompt: str):
    proc = await asyncio.create_subprocess_exec(
        "claude", "-p", "--model", model, "--output-format", "stream-json", "--include-partial-messages",
        "--verbose", "--no-session-persistence", "--tools", "", "--strict-mcp-config",
        "--mcp-config", '{"mcpServers":{}}', "--setting-sources", "", "--system-prompt", system, prompt,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        async for line in proc.stdout:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "stream_event":
                delta = ev["event"].get("delta", {})
                if delta.get("type") == "text_delta":
                    yield delta["text"]
    finally:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    model = body.get("model") or "haiku"
    system, prompt = _prompt(body.get("messages", []))
    rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())

    def chunk(delta: dict, finish=None) -> str:
        return "data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created,
                                      "model": model, "choices": [{"index": 0, "delta": delta,
                                                                   "finish_reason": finish}]}) + "\n\n"

    if not body.get("stream"):
        text = "".join([t async for t in _claude(model, system, prompt)])
        return {"id": rid, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}]}

    async def gen():
        yield chunk({"role": "assistant", "content": ""})
        async for t in _claude(model, system, prompt):
            yield chunk({"content": t})
        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
