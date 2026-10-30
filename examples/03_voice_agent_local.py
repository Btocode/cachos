"""Talk to a multilingual voice agent with this machine's mic and speakers.

    pip install "cachos[recommended,mic]"      # PyAudio needs: apt install portaudio19-dev
    OPENAI_API_KEY=... python examples/03_voice_agent_local.py

Use headphones: local audio has no echo cancellation (the browser transport does).
Any OpenAI-compatible LLM works, e.g. Ollama:
    VoiceAgent(llm_model="llama3.2", llm_base_url="http://localhost:11434/v1")
"""
from cachos import VoiceAgent

agent = VoiceAgent(
    llm_model="gpt-4.1-mini",
    system_prompt="You are a friendly restaurant host. Help callers book a table. Keep replies short.",
    greeting="Greet the caller and ask how you can help.",
    # language_candidates narrows Whisper's language ID to what your callers speak:
    # stt=STT(language_candidates=["en", "hi", "bn"]),
)
agent.run_local()
