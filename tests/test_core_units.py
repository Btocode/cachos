import numpy as np
import pytest

from cachos import Audio, SentenceStream, split_sentences
from cachos import languages as L
from cachos.text import iter_sentences

from .conftest import tone


# ---- audio -------------------------------------------------------------------
def test_wav_roundtrip_and_pcm():
    a = Audio(tone(0.5), 16000)
    b = Audio.load(a.to_wav_bytes())
    assert b.sample_rate == 16000 and len(b) == len(a)
    assert np.allclose(a.samples, b.samples, atol=1e-3)
    c = Audio.from_pcm16(a.to_pcm16(), 16000)
    assert np.allclose(a.samples, c.samples, atol=1e-3)


def test_resample_and_concat():
    a = Audio(tone(1.0, 24000), 24000)
    r = a.resample(16000)
    assert abs(len(r) - 16000) <= 2 and r.sample_rate == 16000
    joined = Audio.concat([a, Audio(tone(0.5), 16000)])
    assert joined.sample_rate == 24000 and abs(joined.duration - 1.5) < 0.01
    assert Audio(np.zeros(0, np.float32), 8000).resample(16000).sample_rate == 16000


def test_stereo_and_int16_input():
    st = np.stack([tone(0.1), tone(0.1)], axis=1)
    assert Audio(st, 16000).samples.ndim == 1
    i16 = (tone(0.1) * 32767).astype(np.int16)
    assert Audio(i16, 16000).samples.dtype == np.float32


# ---- languages (langcodes + lingua) ----------------------------------------------
@pytest.mark.parametrize("inp,out", [("en-US", "en"), ("English", "en"), ("eng", "en"), ("pt_BR", "pt"),
                                     ("zh-HK", "yue"), ("iw", "he"), ("auto", "auto"), (None, None)])
def test_normalize(inp, out):
    assert L.normalize_language(inp) == out


@pytest.mark.parametrize("text,lang", [
    ("Hello, how are you doing today?", "en"),
    ("Hola, ¿cómo estás? Quiero una mesa para dos.", "es"),
    ("Bonjour, je voudrais réserver une table pour ce soir.", "fr"),
    ("Guten Tag, ich möchte einen Tisch für zwei Personen.", "de"),
    ("こんにちは、お元気ですか？", "ja"), ("你好，今天天气怎么样？", "zh"), ("안녕하세요, 반갑습니다", "ko"),
    ("नमस्ते, आप कैसे हैं?", "hi"), ("مرحبا، كيف حالك اليوم؟", "ar"), ("Привет, как у тебя дела?", "ru"),
    ("আপনি কেমন আছেন? আমি ভালো আছি।", "bn"),
])
def test_detect_language(text, lang):
    assert L.detect_language(text) == lang


def test_detect_language_candidates():
    # Portuguese vs Spanish is hard; restricting candidates must be honored
    assert L.detect_language("Olá, tudo bem com você?", ["es", "pt"]) == "pt"
    assert L.detect_language("Hello there my friend", ["fr", "de"]) in ("fr", "de")


# ---- sentences (blingfire) ---------------------------------------------------------
def test_split_sentences_abbrev_and_decimals():
    parts = split_sentences("Dr. Smith paid $3.50 for coffee today. Then he left for the office!")
    assert parts == ["Dr. Smith paid $3.50 for coffee today.", "Then he left for the office!"]


def test_split_sentences_merges_short_fragments():
    assert split_sentences("Yes. I can absolutely help you with that request.") == \
        ["Yes. I can absolutely help you with that request."]


def test_cjk_boundaries():
    parts = split_sentences("今日はとても良い天気ですね。明日も晴れるといいですね。", min_chars=0)
    assert len(parts) == 2


def test_sentence_stream_from_tokens():
    text = "The first sentence is right here. The second one comes next! And the last?"
    out = list(iter_sentences(t + " " for t in text.split(" ")))
    assert out == ["The first sentence is right here.", "The second one comes next!", "And the last?"]
    s = SentenceStream()
    assert s.push("Not finished yet") == []
    assert s.flush() == ["Not finished yet"]


def test_every_module_compiles():
    """Engines load lazily, so a syntax error in one would otherwise go unnoticed."""
    import pathlib
    import py_compile

    import cachos

    for f in pathlib.Path(cachos.__file__).parent.rglob("*.py"):
        py_compile.compile(str(f), doraise=True)
