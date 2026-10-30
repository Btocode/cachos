"""`cachos serve` speaks the OpenAI audio API, so the official SDKs work unchanged.

    cachos serve --port 8000            # in another terminal
    pip install openai && python examples/07_openai_sdk_client.py
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

speech = client.audio.speech.create(model="auto", voice="nova", input="Hola, ¿en qué puedo ayudarte hoy?",
                                    response_format="wav")
speech.write_to_file("hola.wav")  # Spanish detected -> Spanish voice; "nova" = female

with open("hola.wav", "rb") as f:
    t = client.audio.transcriptions.create(model="auto", file=f, response_format="verbose_json")
print(t.language, t.text)
