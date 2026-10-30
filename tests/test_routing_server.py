import asyncio
import io

import pytest

from cachos import STT, TTS, Audio

from .conftest import FakeSTT, FakeTTS, tone


# ---- TTS routing -------------------------------------------------------------
async def test_tts_routes_by_detected_language():
    a, b = FakeTTS("a", ["en"]), FakeTTS("b", ["en", "fr", "ja"])
    tts = TTS([a, b])
    await tts.synthesize("Hello there, friend.")
    await tts.synthesize("Bonjour, je suis très content de vous voir.")
    await tts.synthesize("こんにちは、お元気ですか。")
    assert a.calls[0][1] == "a:en_f"  # priority engine wins for en
    assert b.calls[0][1:] == ("b:fr_f", "fr")
    assert b.calls[1][1:] == ("b:ja_f", "ja")
    assert set(tts.languages) == {"en", "fr", "ja"}


async def test_explicit_voice_gender_and_overrides():
    a, b = FakeTTS("a", ["en"]), FakeTTS("b", ["en", "fr"])
    tts = TTS([a, b], voices={"fr": "b:fr_m"}, gender="male")
    await tts.synthesize("Hello there, how are you?")
    await tts.synthesize("Bonjour à tous, je suis là.")
    await tts.synthesize("Hello.", voice="b:en_f")
    assert a.calls[0][1] == "a:en_m"
    assert b.calls[0][1] == "b:fr_m"
    assert b.calls[1][1] == "b:en_f"
    with pytest.raises(ValueError):
        tts.pick_voice("de")


async def test_tts_stream_from_async_tokens_in_order():
    eng = FakeTTS("a", ["en"])
    tts = TTS([eng])

    async def tokens():
        for t in "The first sentence is here. Another one follows now! And a third one?".split(" "):
            await asyncio.sleep(0)
            yield t + " "

    chunks = [c async for c in tts.stream(tokens(), language="en")]
    assert [c[0] for c in eng.calls] == ["The first sentence is here.", "Another one follows now!",
                                         "And a third one?"]
    assert len(chunks) == 3 and all(isinstance(c, Audio) for c in chunks)


async def test_auto_engines_skip_missing(monkeypatch):
    import cachos.core as core

    def fake_create(spec, **kw):
        if spec == "kokoro":
            raise ImportError("no kokoro")
        return FakeTTS(spec, ["en"])

    monkeypatch.setattr(core, "create_tts", fake_create)
    with pytest.warns(UserWarning, match="kokoro"):
        tts = TTS("auto")
    assert [e.name for e in tts.engines] == ["piper", "edge"]
    assert [e.name for e in TTS("auto", allow_cloud=False).engines] == ["piper"]


async def test_stt_fallback_chain():
    class Broken(FakeSTT):
        def _transcribe(self, audio, language, **o):
            raise RuntimeError("boom")

    stt = STT(Broken(), fallback=[FakeSTT("fallback works")])
    t = await stt.transcribe(Audio(tone(0.5), 16000))
    assert t.text == "fallback works"


# ---- REST server ----------------------------------------------------------------
@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from cachos.server import create_app

    return TestClient(create_app(STT(FakeSTT("bonjour", "fr")), TTS([FakeTTS("t", ["en", "fr"])])))


def test_server_voices_and_languages(client):
    r = client.get("/v1/voices", params={"language": "fr"})
    assert r.status_code == 200 and {v["id"] for v in r.json()["data"]} == {"t:fr_f", "t:fr_m"}
    assert {l["code"] for l in client.get("/v1/languages").json()["tts"]} == {"en", "fr"}


def test_server_openai_compatible_speech_and_transcription(client):
    r = client.post("/v1/audio/speech", json={"input": "Bonjour à tous, comment ça va?", "voice": "onyx",
                                              "response_format": "wav"})
    assert r.status_code == 200 and r.content[:4] == b"RIFF"
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", io.BytesIO(r.content), "audio/wav")},
                    data={"response_format": "verbose_json"})
    assert r.status_code == 200 and r.json()["text"] == "bonjour" and r.json()["language"] == "fr"
    r = client.post("/v1/audio/speech", json={"input": "Hello there.", "response_format": "pcm", "stream": True})
    assert r.status_code == 200 and len(r.content) > 0 and r.headers["x-sample-rate"] == "24000"


def test_server_auth():
    from fastapi.testclient import TestClient

    from cachos.server import create_app

    c = TestClient(create_app(STT(FakeSTT()), TTS([FakeTTS()]), api_key="sekret"))
    assert c.get("/v1/voices").status_code == 401
    assert c.get("/v1/voices", headers={"Authorization": "Bearer sekret"}).status_code == 200


def test_voice_listing_puts_native_voices_before_multilingual():
    from cachos.types import Voice

    eng = FakeTTS("m", ["fr"])
    eng._voices = [Voice("m:multi", "Multi", "m", ["de", "*"], "female"), *eng.list_voices()]
    ids = [v.id for v in eng.voices(language="fr")]
    assert ids[-1] == "m:multi" and ids[0].startswith("m:fr")
