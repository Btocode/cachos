"""Serve the voice agent to browsers (WebRTC) or phone calls, via Pipecat's runner.

    OPENAI_API_KEY=... python examples/04_serve_agent.py            # open http://localhost:7860/client
    OPENAI_API_KEY=... python examples/04_serve_agent.py twilio     # Twilio Media Streams WebSocket

The browser client uses WebRTC, so echo cancellation and jitter handling come from the
browser. For your own web page use `@cachos/client/agent` (clients/ts).
"""
import sys

from cachos import STT, TTS, VoiceAgent

agent = VoiceAgent(
    llm_model="gpt-4.1-mini",
    system_prompt="You are the front desk of Hotel Lumen. Answer briefly and politely.",
    greeting="Welcome the guest.",
    stt=STT(),                    # shared by all sessions: models load once
    tts=TTS(gender="female"),     # voice picked per reply language
    interrupt_min_words=2,        # noisy lobbies: coughs and "mm-hm" don't cut the agent off
)
agent.serve(host="0.0.0.0", port=7860, transport=sys.argv[1] if len(sys.argv) > 1 else "webrtc")
