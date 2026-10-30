# Open-Source Voice Agent Landscape: Research Report (late October 2026)

Scope: building blocks for a Python library that unifies (1) multilingual STT with automatic
language detection, (2) multilingual TTS with many voices, (3) VAD and turn-taking for real-time
voice agents.

**How this was compiled (2026-10-30):** Stars, licenses and last-push dates come from `gh api repos/...`.
Model-weight licenses come from Hugging Face model tags (`/api/models/<id>`). PyPI versions come from
`pypi.org/pypi/<pkg>/json`. **Every Python API snippet in this report was checked against the
repo README, its example files or its source** (function signatures were grepped from source where
it mattered). Where an item is approximate or unverified, it says so.

> **Licensing matters more than usual here.** Code license and **model-weight license often differ**.
> Several popular models are **non-commercial** (marked **NC** below). A library meant for broad
> use should either avoid NC backends by default or gate them behind explicit opt-in extras with
> a warning.

---

## 0. TL;DR recommendations for the unified library

| Role | Default (permissive, CPU-friendly) | Quality / GPU tier | Avoid as default (license) |
|---|---|---|---|
| STT + language ID | `faster-whisper` (MIT, 99 langs, built-in LID) | NeMo Parakeet-TDT-0.6B-v3 (CC-BY-4.0, 25 EU langs, auto LID), Qwen3-ASR (Apache, 52 langs/dialects + LID) | Meta MMS-1b-all (CC-BY-NC), legacy non-English Moonshine models (NC) |
| TTS, many voices, offline | `kokoro-onnx` (MIT code + Apache-2.0 weights, 54 voices, 9 langs), Piper (`piper-tts`, **GPL-3.0**, ~178 voices / ~50 langs) | Chatterbox Multilingual V3 (MIT, 23 langs, cloning), Qwen3-TTS (Apache, 10 langs, 97 ms streaming), VoxCPM2 (Apache, 30 langs) | XTTS-v2 (**CPML, NC**), F5-TTS weights (**CC-BY-NC**), MMS-TTS (**CC-BY-NC**), Fish Audio (revenue-capped), IndexTTS (custom bilibili license) |
| TTS, cloud, zero setup | `edge-tts` (LGPL-3.0, 325 voices / 143 locales; unofficial use of a Microsoft service) | - | - |
| VAD | `silero-vad` (MIT, <1 ms per 32 ms chunk on CPU) | TEN-VAD (Apache-2.0 + Agora non-compete terms) | - |
| Turn detection | Pipecat Smart Turn v3 (BSD-2, audio-native, 23 langs, ONNX on CPU) | - | LiveKit turn detector (LiveKit Model License: **only usable inside LiveKit Agents**) |

Architectural take-aways (details in section 5):
1. Treat **interruption as a first-class, out-of-band signal** (Pipecat: `InterruptionFrame` is a
   SystemFrame that bypasses the data queue and cancels each processor's work task; LiveKit: a
   `SpeechHandle` is interrupted and the TTS/playout is cancelled, with "false interruption" resume).
2. **Stream TTS by sentence** from LLM tokens through a pluggable sentence tokenizer (Pipecat
   `TextAggregationMode.SENTENCE`; LiveKit `StreamAdapter` + blingfire tokenizer; RealtimeTTS
   `stream2sentence` with `fast_sentence_fragment`).
3. Define **capability-flagged interfaces** like LiveKit (`STTCapabilities.streaming`,
   `interim_results`; `TTSCapabilities.streaming`) and use **adapters** to make non-streaming
   engines usable in streaming pipelines (VAD-segmented STT; sentence-chunked TTS).
4. Layer turn-taking: **VAD (fast, acoustic) -> end-of-turn model (semantic/prosodic) -> endpointing
   delays (min/max)**.

---

## 1. Speech-to-Text (multilingual, language detection)

### 1.1 Summary table

| Project | Stars | Code lic. | Weights lic. | Languages | LID | Streaming | Hardware | pip |
|---|---|---|---|---|---|---|---|---|
| openai/whisper | 111.6k | MIT | MIT | 99 | Yes | No (30 s windows) | GPU preferred; CPU slow | `openai-whisper` (20250625) |
| SYSTRAN/faster-whisper | 26.3k | MIT | MIT (Whisper) | 99 | Yes (`info.language`) | No native streaming; segments are a lazy generator; build streaming via VAD chunking | **CPU OK** (int8), GPU fp16 | `faster-whisper` (1.2.1) |
| ggml-org/whisper.cpp (formerly ggerganov) | 54.8k | MIT | MIT | 99 | Yes | `stream` example (sliding window) | **CPU-first**, Metal/CUDA/Vulkan/CoreML | bindings: `pywhispercpp` (350 stars, third party) |
| m-bain/whisperX | 24.9k | BSD-2 | MIT + wav2vec2 aligners + pyannote | 99 (alignment per language model) | Yes | No (batch) | GPU recommended | `whisperx` (3.8.6) |
| NVIDIA-NeMo/Speech (Parakeet, Canary; repo renamed from NVIDIA/NeMo) | 18.7k | Apache-2.0 | **CC-BY-4.0** | Parakeet-TDT-0.6B-v3: 25 EU langs; Canary-1B-v2: 25 EU langs + translation | Parakeet v3: **automatic**; Canary: `source_lang` required | Yes (Nemotron-Speech-Streaming, Parakeet-unified-en, cache-aware models) | GPU preferred; Parakeet runs on CPU (also via sherpa-onnx int8) | `nemo_toolkit[asr]` (3.0.0) |
| FunAudioLLM/SenseVoice (now QwenAudio/SenseVoice) | 9.6k | MIT | FunASR "model-license" (other, attribution) | zh, yue, en, ja, ko (+ more claimed) | Yes (`language="auto"`) + emotion and audio events | Non-autoregressive, very fast; not token streaming | **CPU OK** | `funasr` (1.4.17) |
| moonshine-ai/moonshine (formerly usefulsensors) | 11.3k | MIT | MIT for streaming and English models; **legacy non-English non-streaming = NC Moonshine Community License** | en + several (check docs) | Per-language models | **Yes, designed for live streaming** | **CPU / edge**, down to ~1 MB models | `moonshine-voice` (0.1.5) |
| kyutai-labs/delayed-streams-modeling (Kyutai STT) | 3.0k | Apache-2.0 (Py) / MIT | **CC-BY-4.0** | `stt-1b-en_fr` (en, fr), `stt-2.6b-en` | en/fr model only | **Yes**, 0.5 s delay, semantic VAD | GPU (PyTorch/Rust), MLX on Apple | `moshi` (0.2.13), `moshi-mlx` |
| facebookresearch/omnilingual-asr | 3.0k | Apache-2.0 | **Apache-2.0** | **1,600+** | LLM variants: optional `lang` conditioning (no separate LID head) | No (clips < 40 s) | GPU (300M ~5 GiB VRAM, 7B ~17 GiB) | `omnilingual-asr` (0.2.0, py<=3.12) |
| Meta MMS (fairseq, archived) / `facebook/mms-1b-all` | (fairseq 32.2k, archived) | MIT | **CC-BY-NC-4.0 (NC)** | 1,162 (ASR), 4,000+ (LID) | Separate `mms-lid-*` models | No | GPU | `transformers` |
| QwenLM/Qwen3-ASR (2026 standout) | 3.8k | Apache-2.0 | Apache-2.0 | 30 langs + 22 Chinese dialects (52 total) | **Yes** (`language=None`) | Yes (streaming/offline unified, vLLM) | GPU | `qwen-asr` (0.0.6) |
| microsoft/VibeVoice (VibeVoice-ASR) | 55.0k | MIT | MIT | 50+ | Yes | Streaming variant | GPU 7B; BitNet CPU variant via VibeASR.cpp | repo install |
| k2-fsa/sherpa-onnx (runtime) | 15.6k | Apache-2.0 | per model | Runs Whisper, Parakeet, SenseVoice, Moonshine, Kokoro, Piper, Silero | per model | **Yes** (streaming zipformer/Nemotron) | **CPU / mobile / embedded** | `sherpa-onnx` |

Other 2025-2026 STT notes: **Nemotron-Speech-Streaming** (NVIDIA, 2026-01/03) and
**Parakeet-unified-en-0.6b** (2026-04, 160 ms minimum latency) for English streaming;
**Canary-Qwen-2.5B** (speech-LLM); **FireRedASR** (Apache-2.0, Mandarin/dialects/English);
**ufal/whisper_streaming** (3.7k, MIT) and **collabora/WhisperLive** (4.3k, MIT) for pseudo-streaming
Whisper.

### 1.2 Minimal APIs (verified)

**faster-whisper** (`pip install faster-whisper`, import `faster_whisper`)
```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cpu", compute_type="int8")   # or device="cuda", compute_type="float16"
segments, info = model.transcribe("audio.mp3", beam_size=5)          # audio: str | BinaryIO | np.ndarray (16 kHz float32)
print(info.language, info.language_probability)                     # auto language detection when language=None
for s in segments:                                                   # generator: transcription runs while you iterate
    print(s.start, s.end, s.text)
```
- `WhisperModel(model_size_or_path, device="auto", device_index=0, compute_type="default", cpu_threads=0, download_root=None, ...)`
- `transcribe(audio, language=None, task="transcribe", beam_size=5, ..., initial_prompt=None, word_timestamps=False, multilingual=False, vad_filter=False, vad_parameters=None, hotwords=None, language_detection_threshold=0.5, language_detection_segments=1, ...)`
  returns `(Iterable[Segment], TranscriptionInfo)`.
- `Segment` fields: `id, seek, start, end, text, tokens, avg_logprob, compression_ratio, no_speech_prob, words, temperature`.
- `TranscriptionInfo`: `language, language_probability, duration, duration_after_vad, all_language_probs, ...`.
- `model.detect_language(audio=np.ndarray, vad_filter=False, ..., language_detection_segments=1, language_detection_threshold=0.5) -> (lang, prob, [(lang, prob), ...])`.
- `BatchedInferencePipeline(model=model).transcribe(audio, batch_size=16)` for throughput (its `vad_filter` defaults to `True`).
- `vad_filter=True` uses a bundled Silero VAD. Model names include `tiny ... large-v3`, `turbo`, `distil-large-v3`.
- `multilingual=True` re-detects language per segment (code-switching).

**openai-whisper** (`pip install openai-whisper`, import `whisper`)
```python
import whisper
model = whisper.load_model("turbo")
result = model.transcribe("audio.mp3")      # dict: text, segments, language
# low level: mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(whisper.load_audio(p)), n_mels=model.dims.n_mels).to(model.device)
# _, probs = model.detect_language(mel)
```

**whisperX** (`pip install whisperx`)
```python
import whisperx
model = whisperx.load_model("large-v2", "cuda", compute_type="float16")
audio = whisperx.load_audio("audio.mp3")
result = model.transcribe(audio, batch_size=16)            # result["language"]
model_a, meta = whisperx.load_align_model(language_code=result["language"], device="cuda")
result = whisperx.align(result["segments"], model_a, meta, audio, "cuda", return_char_alignments=False)
# diarization: from whisperx.diarize import DiarizationPipeline; DiarizationPipeline(token=HF_TOKEN, device="cuda")
```

**whisper.cpp via pywhispercpp** (`pip install pywhispercpp`)
```python
from pywhispercpp.model import Model
model = Model("base.en", print_realtime=False, print_progress=False)
for seg in model.transcribe("file.wav"):
    print(seg.text)
```

**NeMo Parakeet / Canary** (`pip install -U "nemo_toolkit[asr]"`)
```python
import nemo.collections.asr as nemo_asr
asr = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
out = asr.transcribe(["audio.wav"])           # language auto-detected (25 EU languages)
print(out[0].text)                            # timestamps=True for word/segment stamps

from nemo.collections.asr.models import ASRModel
canary = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")
out = canary.transcribe(["audio.wav"], source_lang="en", target_lang="en")   # target_lang != source -> translation
```

**SenseVoice** (`pip install funasr`)
```python
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
model = AutoModel(model="iic/SenseVoiceSmall", trust_remote_code=True, remote_code="./model.py",
                  vad_model="fsmn-vad", vad_kwargs={"max_single_segment_time": 30000}, device="cuda:0")
res = model.generate(input="a.mp3", cache={}, language="auto", use_itn=True, batch_size_s=60,
                     merge_vad=True, merge_length_s=15)
text = rich_transcription_postprocess(res[0]["text"])   # raw text contains <|lang|><|emotion|><|event|> tags
```
(`remote_code="./model.py"` refers to the file in the SenseVoice repo; `hf` hub id `FunAudioLLM/SenseVoiceSmall` also works.)

**Moonshine Voice** (`pip install moonshine-voice`, import `moonshine_voice`). Pattern: construct, then chainable setters, then `load()`, then `start()`.
```python
from moonshine_voice import MicTranscriber
mic = (MicTranscriber().language("en")
       .on_text(lambda t: print(t, end="\r"))      # partial hypothesis
       .on_line(lambda line: print(line.text)))     # finalized segment
mic.load(); mic.start()
# Transcriber = feed PCM yourself; TextToSpeech and AgentFlow also exist in the same package.
```

**Kyutai STT** (`pip install moshi`): CLI-first. `python -m moshi.run_inference --hf-repo kyutai/stt-1b-en_fr audio.mp3`.
Production use is the Rust `moshi-server` (websocket streaming). Python streaming means stepping the
`moshi` LM/Mimi models (see `scripts/stt_from_file_pytorch.py`), so an adapter should wrap the server
protocol rather than the internal API.

**Omnilingual ASR** (`pip install omnilingual-asr`)
```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
pipe = ASRInferencePipeline(model_card="omniASR_LLM_1B_v2")
texts = pipe.transcribe(["a.flac"], lang=["eng_Latn"], batch_size=1)   # lang codes: {iso639-3}_{Script}
```

**MMS ASR** (`transformers`, NC weights): `Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")` +
`processor.tokenizer.set_target_lang("fra"); model.load_adapter("fra")`. LID through
`facebook/mms-lid-126` and similar (`Wav2Vec2ForSequenceClassification`).

**Qwen3-ASR** (`pip install -U qwen-asr`)
```python
import torch
from qwen_asr import Qwen3ASRModel
model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0",
                                      max_inference_batch_size=32, max_new_tokens=256)
r = model.transcribe(audio="a.wav", language=None)[0]    # None = auto LID
print(r.language, r.text)
```

---

## 2. Text-to-Speech (multilingual, many voices)

### 2.1 Summary table

| Project | Stars | Code lic. | Weights lic. | Languages | Voices / cloning | Streaming | Hardware | pip |
|---|---|---|---|---|---|---|---|---|
| hexgrad/kokoro (Kokoro-82M) | 9.8k | Apache-2.0 | Apache-2.0 | 9 (en-US, en-GB, es, fr, hi, it, ja, pt-BR, zh) | **54 preset voices**, no cloning (voice blending possible) | Per-sentence generator | **CPU OK** (82M) | `kokoro` (0.9.4, py<3.13) + system `espeak-ng` |
| thewh1teagle/kokoro-onnx | 2.8k | MIT | Apache-2.0 | same as Kokoro | 54 voices, style-vector blending | `create_stream` async (sentence batches) | **CPU OK**, ~300 MB (80 MB quantized), onnxruntime | `kokoro-onnx` (0.6.1) |
| rhasspy/piper | 11.3k | MIT | per voice (mostly permissive; check each model card) | - | - | - | - | **archived 2025**; superseded |
| OHF-Voice/piper1-gpl (current Piper) | 6.1k | **GPL-3.0** | per voice | ~50 language families | **~178 voices** (`rhasspy/piper-voices`) | Yes (`synthesize()` yields chunks per sentence) | **CPU, very fast** (Raspberry Pi) | `piper-tts` (1.8.0) |
| idiap/coqui-ai-TTS (maintained Coqui fork) | 2.3k (orig. coqui-ai/TTS 46k, unmaintained) | MPL-2.0 | **XTTS-v2: CPML = NC**; other models vary; MMS/fairseq VITS = NC | XTTS-v2: 17; 1,100+ through fairseq VITS | XTTS: 58 built-in speakers + 3 s cloning | XTTS `inference_stream` (<200 ms) | XTTS GPU recommended | `coqui-tts` (0.28.0), import `TTS` |
| resemble-ai/chatterbox | 27.4k | MIT | MIT | Multilingual V3: 23 (ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh); Turbo/Nano: en | Zero-shot cloning (reference clip); paralinguistic tags (Turbo) | Not in the official package (community forks) | Multilingual 0.5B GPU / MPS; **Nano 110M CPU 3x realtime** | `chatterbox-tts` (0.1.7) |
| SWivid/F5-TTS | 15.5k | MIT | **CC-BY-NC-4.0 (NC)** (base weights) | en, zh (+ community finetunes) | Zero-shot cloning (needs ref text) | Chunked socket server | GPU | `f5-tts` (1.1.22) |
| FunAudioLLM/CosyVoice (now QwenAudio/CosyVoice) | 24.2k | Apache-2.0 | Apache-2.0 (Fun-CosyVoice3-0.5B) | 9 langs + 18 Chinese dialects | Zero-shot + cross-lingual cloning, instruct | **Bi-streaming** (text in and audio out), ~150 ms | GPU | No PyPI; repo + `requirements.txt` |
| fishaudio/fish-speech (Drama 3, S2-Pro) | 33.5k | **Fish Audio Community License** | same / research license (S2-Pro) | Multilingual | Cloning, NL tags | Yes (server) | Big GPU (4B / 35B-A3B) | repo / SGLang-Omni |
| myshell-ai/MeloTTS | 7.7k | MIT | MIT | en (US/BR/IN/AU), es, fr, zh (mixed en), ja, ko | ~10 accent speakers | No | **CPU real-time** | `melotts` (0.1.1, stale; git install recommended) |
| myshell-ai/OpenVoice | 38.1k | MIT | MIT (v2) | via MeloTTS base | Tone-color cloning (voice conversion) | No | GPU preferred | repo |
| nari-labs/dia (+ dia2) | 19.4k | Apache-2.0 | Apache-2.0 | **English only** | Dialogue `[S1]/[S2]`, audio-prompt cloning | Dia2: streaming | GPU (1.6B) | via `transformers` (`DiaForConditionalGeneration`) |
| canopyai/Orpheus-TTS | 6.4k | Apache-2.0 | Apache-2.0 | en (+ 7-language research-preview family) | 8 English voices (`tara`, `leah`, ...), zero-shot cloning | Yes (token stream) | GPU (3B, vLLM) | `orpheus-speech` (0.1.0) |
| Kyutai TTS (delayed-streams-modeling) | 3.0k | Apache/MIT | CC-BY-4.0 | en, fr | Voices from `kyutai/tts-voices` (cloning by voice embeddings) | **Yes, text-in streaming** | GPU / Rust server / MLX | `moshi` |
| kyutai-labs/pocket-tts (2026 standout) | 10.1k | MIT | CC-BY-4.0 | en, fr, de, pt, it, es, nl | Preset voices + cloning from wav | Yes | **CPU** (100M; English models down to 1.9M) | `pocket-tts` (4.0.1) |
| rany2/edge-tts | 12.4k | **LGPL-3.0** (one file MIT) | n/a (Microsoft cloud) | 78 languages / **143 locales** | **325 neural voices** (counted 2026-10-30), no cloning | Yes (`stream()` async/sync) | Any, needs internet | `edge-tts` (7.3.1) |
| Meta MMS-TTS (`facebook/mms-tts-*`) | - | Apache (transformers) | **CC-BY-NC-4.0 (NC)** | **~1,100** (one checkpoint per language) | 1 voice per language | No | **CPU OK** (VITS) | `transformers` (+ `uroman` for non-Latin scripts) |
| microsoft/VibeVoice | 55.0k | MIT | MIT | Realtime-0.5B: en + experimental de, fr, it, ja, ko, nl, pl, pt, es voices; 1.5B: en, zh | Preset voices; long-form 4-speaker | Realtime-0.5B: streaming text in | GPU | repo (TTS-1.5B code was **removed** 2025-09) |
| index-tts/index-tts (IndexTTS-2 / 2.5) | 24.8k | **bilibili Model Use License** | same (custom) | 2.5: zh, en, ja, es, ar | Zero-shot cloning, emotion control, duration control | No | GPU | repo (uv) |
| QwenLM/Qwen3-TTS (2026 standout) | 14.1k | Apache-2.0 | Apache-2.0 | 10 (zh, en, ja, ko, de, fr, ru, pt, es, it) | 9 premium voices (CustomVoice), 3 s cloning (Base), VoiceDesign | **Yes, 97 ms first packet** | GPU | `qwen-tts` (0.1.1) |
| OpenBMB/VoxCPM (VoxCPM2) | 40.8k | Apache-2.0 | Apache-2.0 | 30 | Voice design + cloning, 48 kHz | Yes (RTF ~0.3 on a 4090) | GPU (2B) | `voxcpm` (2.0.3) |
| KittenML/KittenTTS | 16.7k | Apache-2.0 | "other" on HF (check) | English | Named preset voices | Long text / streaming | **CPU** (15M-80M; 1.7B 1-bit model) | `kittenml` (0.9.7) |
| supertone Supertonic 3 | 13.8k | MIT | **OpenRAIL** (use restrictions) | 31 (+ `lang="na"`) | 10 voices (M1-M5, F1-F5) | Chunked | **CPU, ONNX** | `supertonic` (1.3.1) |
| neuphonic/neutts | 6.3k | NeuTTS Open License (free under USD 5M revenue) | Air/Nano: Apache-2.0 on HF | en, fr, de, es (Nano family) | Instant cloning | Yes | **CPU** (GGUF) | `neutts` (1.4.1) |
| 2noise/ChatTTS | 40.0k | **AGPL-3.0** | NC | zh, en | random speaker sampling | - | GPU | avoid for a permissive library |

### 2.2 Minimal APIs (verified)

**kokoro-onnx** (`pip install -U kokoro-onnx soundfile`). Download model files from the GitHub release
`model-files-v1.1`: `kokoro-v1.0.onnx` and `voices-v1.0.bin`.
```python
import soundfile as sf
from kokoro_onnx import Kokoro

kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
samples, sample_rate = kokoro.create("Hello world", voice="af_sarah", speed=1.0, lang="en-us")
sf.write("audio.wav", samples, sample_rate)           # float32 numpy, 24000 Hz
print(kokoro.get_voices())                            # list[str], 54 voices

# streaming (async generator, yields per sentence batch):
async for samples, sr in kokoro.create_stream(text, voice="af_nicole", speed=1.0, lang="en-us"):
    ...
```
Verified signatures (`src/kokoro_onnx/__init__.py`):
- `Kokoro(model_path: str, voices_path: str, espeak_config: EspeakConfig | None = None, vocab_config: dict | str | None = None)`; also `Kokoro.from_session(...)` for a custom onnxruntime session (CUDA providers).
- `create(text, voice: str | NDArray[float32], speed=1.0, lang="en-us", is_phonemes=False, trim=True, sentence_pause=0.25, clause_pause=0.1, continuous=False) -> (NDArray[float32], int)`
- `async create_stream(text, voice, speed=1.0, lang="en-us", is_phonemes=False, trim=True, sentence_pause=0.25, clause_pause=0.1) -> AsyncGenerator[(NDArray, int)]`
- `create_timed(...)` (adds timestamps), `get_voice_style(name) -> NDArray` (for blending), `get_voices() -> list[str]`.
- Voice prefix gives language and gender: `af_`/`am_` American, `bf_`/`bm_` British, `jf_` Japanese, `zf_` Mandarin, `ef_` Spanish, `ff_` French, `hf_` Hindi, `if_` Italian, `pf_` Portuguese-BR.
- `lang` is an espeak-style code (`en-us`, `en-gb`, `es`, `fr-fr`, `hi`, `it`, `pt-br`, `ja`, `cmn`). For ja/zh, the README recommends the `misaki` G2P (see `examples/japanese.py`, `chinese.py`).

**kokoro (PyTorch)** (`pip install kokoro>=0.9.4 soundfile`, plus `apt install espeak-ng`)
```python
from kokoro import KPipeline
pipeline = KPipeline(lang_code="a")      # a=en-US b=en-GB e=es f=fr h=hi i=it j=ja p=pt-BR z=zh
for gs, ps, audio in pipeline(text, voice="af_heart"):   # yields (graphemes, phonemes, audio @24 kHz) per chunk
    ...
```

**Piper (current = `piper-tts` from OHF-Voice/piper1-gpl; GPL-3.0)**
```bash
pip install piper-tts
python3 -m piper.download_voices en_US-lessac-medium
```
```python
import wave
from piper import PiperVoice, SynthesisConfig

voice = PiperVoice.load("/path/to/en_US-lessac-medium.onnx")     # use_cuda=True needs onnxruntime-gpu
with wave.open("test.wav", "wb") as wav_file:
    voice.synthesize_wav("Welcome to the world of speech synthesis!", wav_file)

# streaming: one AudioChunk per sentence
for chunk in voice.synthesize("Some text. More text."):
    chunk.sample_rate, chunk.sample_width, chunk.sample_channels
    chunk.audio_int16_bytes          # also .audio_float_array, .audio_int16_array

cfg = SynthesisConfig(volume=0.5, length_scale=2.0, noise_scale=1.0, noise_w_scale=1.0, normalize_audio=False)
voice.synthesize_wav(text, wav_file, syn_config=cfg)
```
Exports: `AudioChunk, PhonemeType, PiperConfig, PiperVoice, SynthesisConfig`. Each voice is an `.onnx`
file plus an `.onnx.json` file. The voice name encodes language, region, name and quality.
**The old `piper-phonemize`/`piper` MIT packages are dead.** If GPL is unacceptable, run Piper voices
through **sherpa-onnx** (Apache-2.0). The voices are plain VITS ONNX files, but each model card has its own dataset license.

**edge-tts** (`pip install edge-tts`; LGPL-3.0; needs internet; unofficial use of the Edge Read-Aloud endpoint, which can break or be rate limited)
```python
import edge_tts
from edge_tts import VoicesManager

async def main():
    communicate = edge_tts.Communicate("Hola", "es-ES-AlvaroNeural", rate="+0%", volume="+0%", pitch="+0Hz")
    await communicate.save("out.mp3")                     # MP3 output (24 kHz 48 kbit/s mono)
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":            chunk["data"]   # mp3 bytes
        elif chunk["type"] in ("WordBoundary", "SentenceBoundary"): ...   # offset/duration/text
    voices = await VoicesManager.create()
    v = voices.find(Gender="Female", Language="es")       # or Locale="es-AR"; each item has ["Name"]
    all_voices = await edge_tts.list_voices()
# sync variants: communicate.stream_sync(), communicate.save_sync(path)
```
`Communicate(text, voice=DEFAULT_VOICE, *, rate="+0%", volume="+0%", pitch="+0Hz", boundary="SentenceBoundary", connector=None, proxy=None, connect_timeout=10, receive_timeout=60)`.
Exports: `Communicate, SubMaker, VoicesManager, list_voices, exceptions`. Custom SSML is **not** supported.
Output is MP3, so decode it (for example with PyAV or ffmpeg) to get PCM for a real-time pipeline.

**Coqui TTS, idiap fork** (`pip install coqui-tts`; import name is still **`TTS`**; py 3.10-3.14)
```python
import torch
from TTS.api import TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print(tts.speakers)                                                   # built-in XTTS speakers
wav = tts.tts(text="Hello world!", speaker_wav="ref.wav", language="en")           # list of floats, 24 kHz
tts.tts_to_file(text="Hello world!", speaker="Craig Gutsy", language="en", file_path="out.wav")
# ~1100 languages through fairseq MMS VITS (NC):  TTS("tts_models/<iso>/fairseq/vits")
```
XTTS streaming (low-level API):
```python
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
config = XttsConfig(); config.load_json("/path/xtts/config.json")
model = Xtts(config); model.load_checkpoint(config, checkpoint_dir="/path/xtts/"); model.cuda()
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["ref.wav"])
for chunk in model.inference_stream(text, "en", gpt_cond_latent, speaker_embedding):   # torch tensors @24 kHz
    ...
```
XTTS-v2 languages: ar, cs, de, en, es, fr, hi, hu, it, ja, ko, nl, pl, pt, ru, tr, zh-cn.
**XTTS-v2 weights are under the Coqui Public Model License (non-commercial only)**, and Coqui the company no longer exists to sell licenses.

**Chatterbox Multilingual** (`pip install chatterbox-tts`; developed on Python 3.11; pinned deps)
```python
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")   # "cpu"/"mps" OK; omit/"v2" = legacy
wav = model.generate("Bonjour, comment ça va?", language_id="fr")                # torch tensor (1, N)
wav = model.generate(text, language_id="fr", audio_prompt_path="ref.wav")        # zero-shot cloning
ta.save("out.wav", wav, model.sr)                                                # model.sr == 24000
```
Verified signatures (`src/chatterbox/mtl_tts.py`):
- `ChatterboxMultilingualTTS.from_pretrained(device, t3_model: str | None = None)`
- `generate(text, language_id, audio_prompt_path=None, exaggeration=0.5, cfg_weight=0.5, temperature=0.8, repetition_penalty=1.2, min_p=0.05, top_p=1.0)`. It raises `ValueError` on an unsupported `language_id`.
- Other classes: `chatterbox.tts.ChatterboxTTS` (English), `chatterbox.tts_turbo.ChatterboxTurboTTS.from_pretrained(device=..., nano=False)` (English, 350M; `nano=True` gives the 110M CPU model).
- Output carries the **Perth watermark**. Use `cfg_weight=0` to reduce accent bleed when the language of the reference clip differs from `language_id`.
- Single-language finetunes on HF: `ResembleAI/Chatterbox-Multilingual-{zh-cmn, es-mx-latam, pt-br, es-es, pt-pt, hi}`.

**Transformers MMS-TTS** (`pip install transformers` (5.x needs py>=3.11) plus `pip install uroman` for non-Latin scripts; **weights CC-BY-NC-4.0**)
```python
import torch, scipy
from transformers import AutoTokenizer, VitsModel, set_seed
tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")       # one checkpoint per ISO-639-3 code: mms-tts-fra, mms-tts-hin, ...
model = VitsModel.from_pretrained("facebook/mms-tts-eng")
inputs = tok("Hello, my dog is cute", return_tensors="pt")
set_seed(555)                                                     # VITS is stochastic
with torch.no_grad():
    waveform = model(**inputs).waveform[0]                         # float tensor
scipy.io.wavfile.write("hello.wav", rate=model.config.sampling_rate, data=waveform.numpy())   # 16 kHz
# or: pipeline("text-to-speech", model="facebook/mms-tts-eng")(text) -> {"audio", "sampling_rate"}
# tokenizer.is_uroman tells whether romanization is applied (automatic if uroman is installed)
```

**F5-TTS** (`pip install f5-tts`; NC weights)
```python
from f5_tts.api import F5TTS
f5 = F5TTS()                                  # model="F5TTS_v1_Base", device=None
wav, sr, spec = f5.infer(ref_file="ref.wav", ref_text="transcript of ref", gen_text="text to speak")
```

**CosyVoice** (repo clone; `sys.path` must include `third_party/Matcha-TTS`)
```python
from cosyvoice.cli.cosyvoice import AutoModel
cv = AutoModel(model_dir="pretrained_models/Fun-CosyVoice3-0.5B")   # or CosyVoice2-0.5B
for out in cv.inference_zero_shot(text, prompt_text, "prompt.wav", stream=True):
    out["tts_speech"]                         # torch tensor @ cv.sample_rate
# also inference_sft(text, spk_id), inference_cross_lingual(text, wav), inference_instruct2(text, instruct, wav);
# text may be a generator for text-in streaming
```

**MeloTTS** (`pip install git+https://github.com/myshell-ai/MeloTTS.git`; `python -m unidic download` for ja)
```python
from melo.api import TTS
model = TTS(language="EN", device="auto")     # EN, ES, FR, ZH, JP, KR
spk = model.hps.data.spk2id                   # e.g. 'EN-US', 'EN-BR', 'EN_INDIA', 'EN-AU', 'EN-Default'
model.tts_to_file("text", spk["EN-US"], "out.wav", speed=1.0)
```

**Orpheus** (`pip install orpheus-speech`; vLLM)
```python
from orpheus_tts import OrpheusModel
model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod", max_model_len=2048)
for pcm16_chunk in model.generate_speech(prompt=text, voice="tara"):   # bytes, 24 kHz mono int16
    ...
```

**Dia** (through `transformers`): `AutoProcessor` + `DiaForConditionalGeneration.from_pretrained("nari-labs/Dia-1.6B-0626")`,
`model.generate(**inputs, max_new_tokens=3072, guidance_scale=3.0, temperature=1.8, top_p=0.90, top_k=45)`,
then `processor.batch_decode(...)` and `processor.save_audio(...)`. English only. Script format is `[S1] ... [S2] ...`.

**Qwen3-TTS** (`pip install -U qwen-tts`)
```python
import torch
from qwen_tts import Qwen3TTSModel
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0", dtype=torch.bfloat16)
wavs, sr = m.generate_custom_voice(text="...", language="English", speaker="Ryan", instruct="")  # language="Auto" allowed
m.get_supported_speakers(); m.get_supported_languages()
# -Base checkpoints: 3-second voice cloning; -VoiceDesign: voices described in natural language
```

**Pocket TTS** (`pip install pocket-tts`; CPU)
```python
from pocket_tts import TTSModel
tts = TTSModel.load_model()
state = tts.get_state_for_audio_prompt("alba")          # preset voice, local wav, or "hf://kyutai/tts-voices/..."
audio = tts.generate_audio(state, "Hello world")         # 1-D torch tensor; tts.sample_rate
```

**IndexTTS-2.5** (repo, `uv`): `from indextts.infer_v2_5 import IndexTTS2; tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_bf16=True); tts.infer(spk_audio_prompt="ref.wav", text=..., lang="EN", output_path="gen.wav")`.

**VoxCPM2** (`pip install voxcpm`): `VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False).generate(text=..., cfg_value=2.0, inference_timesteps=10)`; the sample rate is `model.tts_model.sample_rate`.

**Supertonic** (`pip install supertonic`): `TTS(auto_download=True)`, `style = tts.get_voice_style(voice_name="M1")`, `wav, dur = tts.synthesize(text, voice_style=style, lang="en", total_steps=8, speed=1.05)`.

**KittenTTS** (`pip install kittenml`): `KittenTTS("KittenML/kitten-tts-2").generate(text, voice="Matthew")`; the sample rate is `m.sample_rate`.

**NeuTTS** (`pip install neutts`): `NeuTTS(backbone_repo="neuphonic/neutts-nano", backbone_device="cpu", codec_repo="neuphonic/neucodec", codec_device="cpu")`, `ref_codes = tts.encode_reference(wav)`, `wav = tts.infer(text, ref_codes, ref_text)` (24 kHz).

---

## 3. VAD and turn detection

| Project | Stars | License | What it is | Languages | Hardware | pip |
|---|---|---|---|---|---|---|
| snakers4/silero-vad (v6) | 10.6k | **MIT** (code and model) | Frame-level speech probability | Language-agnostic (trained on 6,000+ languages) | CPU, <1 ms per 30+ ms chunk on one thread; ONNX or JIT | `silero-vad` (6.2.3); extras `[onnx-cpu]`, `[onnx-gpu]` |
| TEN-framework/ten-vad | 2.3k | Apache-2.0 **plus Agora conditions** (no deployment that competes with Agora; only for your own app and its end users) | Frame-level VAD, lower latency than Silero on speech-to-silence transitions | Language-agnostic | CPU, C library + ONNX | `ten-vad` (1.0.6.9) / git |
| pipecat-ai/smart-turn v3.2 | 1.6k | **BSD-2** (code, data, weights) | **Audio-native** end-of-turn classifier (Whisper-encoder features to sigmoid) | 23 languages | CPU int8 8 MB (~10-65 ms), GPU fp32 32 MB | no package; `inference.py`/`model.py` from the repo, or Pipecat's `LocalSmartTurnAnalyzerV3` |
| LiveKit turn detector | (in livekit/agents) | Plugin code Apache-2.0; **model under the LiveKit Model License: may only be used with LiveKit Agents** | Was a text-based (transcript plus chat context) EOU LM, English and multilingual. **Now deprecated** in favor of `livekit.agents.inference.TurnDetector`, an audio EOT detector with cloud-to-local (`v1-mini`) fallback | Multilingual | CPU, <500 MB RAM | `livekit-plugins-turn-detector` (deprecated) / built into `livekit-agents` |
| TEN-framework/ten-turn-detection | 0.6k | Apache + Agora conditions | Text classifier (finished / unfinished / wait) built on Qwen2.5-7B | en, zh | GPU | repo |
| Kyutai STT `stt-1b-en_fr` | - | CC-BY-4.0 | STT with a built-in **semantic VAD** head | en, fr | GPU | `moshi` |

### 3.1 silero-vad API (verified against `src/silero_vad/`)
```python
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, VADIterator

model = load_silero_vad()                      # load_silero_vad(onnx=False, opset_version=16, sequence=False, sampling_rate=16000, max_frames=512)
wav = read_audio("file.wav")                   # 16 kHz torch tensor
ts = get_speech_timestamps(wav, model, return_seconds=True)
# get_speech_timestamps(audio, model, threshold=0.5, sampling_rate=16000, min_speech_duration_ms=250,
#                       max_speech_duration_s=inf, min_silence_duration_ms=100, speech_pad_ms=30, return_seconds=False, ...)

# Streaming:
vad = VADIterator(model, threshold=0.5, sampling_rate=16000, min_silence_duration_ms=100, speech_pad_ms=30)
for chunk in chunks_of_512_samples:            # EXACTLY 512 samples @16 kHz (256 @8 kHz)
    ev = vad(chunk, return_seconds=True)       # None | {'start': t} | {'end': t}
vad.reset_states()

# Raw probability per chunk:
prob = model(chunk_tensor, 16000).item()       # also requires 512 samples @16 kHz
```
Supported sample rates: 8000 and 16000 only. Exports: `load_silero_vad, get_speech_timestamps,
get_speech_timestamps_from_probs, save_audio, read_audio, VADIterator, collect_chunks, drop_chunks`,
plus `SileroVADSequence` and `get_speech_timestamps_sequence` (loaded lazily; need the onnxruntime extra).

### 3.2 TEN-VAD
```python
from ten_vad import TenVad
vad = TenVad(hop_size=256, threshold=0.5)      # 16 kHz only; hop 160 (10 ms) or 256 (16 ms)
prob, flag = vad.process(int16_frame_of_hop_size)
```

### 3.3 Smart Turn v3
```python
# from the smart-turn repo: inference.py (needs onnxruntime + transformers' WhisperFeatureExtractor)
from inference import predict_endpoint
r = predict_endpoint(audio_float32_16k)        # {"prediction": 1|0, "probability": float}; uses the last 8 s
```
Intended use: run Silero for speech/silence. When Silero reports silence, run Smart Turn on the whole
current user turn (truncated to the last 8 s). If the result is "incomplete", keep listening, with
a fallback timeout.

---

## 4. Frameworks (overview)

| Framework | Stars | License | Last push | Lang | pip | Positioning |
|---|---|---|---|---|---|---|
| pipecat-ai/pipecat | 16.9k | BSD-2 | 2026-10-29 | Python | `pipecat-ai` (1.15.0, py>=3.11) | Frame/processor pipelines, very many service integrations (local: Whisper, Kokoro, Piper, XTTS, Moonshine, Pocket TTS, FunASR...) |
| livekit/agents | 15.0k | Apache-2.0 (turn models: LiveKit Model License) | 2026-10-30 | Python (+ Node) | `livekit-agents` (1.8.10) | WebRTC-native agent sessions with STT/LLM/TTS/VAD plugin interfaces |
| KoljaB/RealtimeSTT | 10.2k | MIT | 2026-10-28 | Python | `RealtimeSTT` (1.1.2, py 3.11-3.12) | Mic/feed-in VAD + ASR recorder with realtime partials, wake words, pluggable engines |
| KoljaB/RealtimeTTS | 4.0k | MIT | 2026-10-28 | Python | `realtimetts` (0.8.10) | Text-stream to sentence splitting to engine to playback, 30+ engines |
| TEN-framework/ten-framework | 11.2k | Apache-2.0 + Agora conditions | 2026-10-30 | C/C++ core, Go/Python/Node extensions | - | Graph of extensions with a multi-language runtime; Agora RTC centric |
| vocodedev/vocode-core | 3.8k | MIT | **2024-11 (dormant, seeking maintainers)** | Python | `vocode` | Legacy `StreamingConversation`; do not build on it |
| huggingface/speech-to-speech | 13.6k | Apache-2.0 | 2026-10-29 | Python | repo | Simple VAD, STT, LLM, TTS cascade with local models |

---

## 5. Architecture methodology analysis

### 5.1 Pipecat: frames and processors

**Core abstraction.** Everything is a `Frame` flowing through a linear `Pipeline([...])` of
`FrameProcessor`s. Frames move `FrameDirection.DOWNSTREAM` (input to output) or `UPSTREAM` (for example
errors and control). A processor overrides `async def process_frame(self, frame, direction)` and
calls `await self.push_frame(frame, direction)` to pass frames on.

**Three frame classes with different scheduling** (`pipecat/frames/frames.py`):
- `SystemFrame`: high priority, processed **immediately on the input task**. Examples: `StartFrame`,
  `CancelFrame`, `InterruptionFrame`, `UserStartedSpeakingFrame`/`UserStoppedSpeakingFrame`,
  `VADUserStartedSpeakingFrame`, `BotStartedSpeakingFrame`, `InputAudioRawFrame`, `STTMuteFrame`, `MetricsFrame`.
- `DataFrame`: ordered content. Examples: `TextFrame`, `LLMTextFrame`, `TranscriptionFrame`,
  `InterimTranscriptionFrame`, `TTSAudioRawFrame`, `OutputAudioRawFrame`, `TTSSpeakFrame`, `LLMRunFrame`, `LLMMessagesAppendFrame`.
- `ControlFrame`: ordered control. Examples: `EndFrame`, `LLMThoughtStartFrame`, and others.
- An `UninterruptibleFrame` mixin marks frames that must survive interruptions.

**Two queues per processor** (`processors/frame_processor.py`). `queue_frame()` puts everything on an
`__input_queue`. The input task processes `SystemFrame`s inline and forwards all other frames to an
`__process_queue`, which is drained by a separate **process task**. This is the key to barge-in: a
system frame never waits behind slow data work.

**Interruption / barge-in.** When the user turn controller decides the user has started speaking
(VAD and/or transcription start strategies), it broadcasts an `InterruptionFrame`
(`broadcast_interruption()`). Each processor's `_start_interruption()`:
- cancels and recreates its process task, which drops queued data frames (frames marked
  uninterruptible are kept). If the frame currently being processed is uninterruptible, it only
  flushes the queue.
- The TTS service's `_handle_interruption` drops its audio contexts. The output transport stops
  playback and clears its buffers. The assistant context aggregator commits only what was actually
  spoken (word timestamps through `TTSTextFrame`).

**Turn-taking is strategy-based** (`pipecat/turns/`). `UserTurnStrategies(start=[...], stop=[...])`. The defaults are:
- start: `[VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy]`
- stop: `[TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3)]`. **Smart Turn v3 is the default end-of-turn detector.**

Alternatives include `MinWordsUserTurnStartStrategy` (prevents a backchannel like "uh-huh" from
interrupting the bot), `WakePhraseUserTurnStartStrategy`, `SpeechTimeoutUserTurnStopStrategy`,
`ClassifierUserTurnStopStrategy`, LLM-based turn completion, `EagerUserTurnStrategies` (speculative
early responses with cancel frames), user-mute strategies, and user-idle detection.
The VAD analyzer is attached to the user aggregator:
`LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()))`.

**Services.**
- `STTService.run_stt(audio: bytes) -> AsyncGenerator[Frame]`. `SegmentedSTTService` buffers audio
  between VAD start/stop and transcribes each segment, which is how Whisper-style batch models fit a
  streaming pipeline. `WebsocketSTTService` covers streaming cloud STT.
- `TTSService.run_tts(text, context_id) -> AsyncGenerator[Frame]` is the only method a TTS adapter
  must implement.

**Streaming TTS by sentence.** `TTSService` aggregates incoming `LLMTextFrame` tokens.
`TextAggregationMode.SENTENCE` (the default) buffers until a sentence boundary is detected, using a
language-aware sentence tokenizer through `SimpleTextAggregator`. `TOKEN` passes tokens straight to
providers that accept streaming input. `PatternPairAggregator` and `SkipTagsAggregator` handle inline
tags, and text filters and transforms (for example markdown stripping) run before synthesis. The
source comment puts the sentence-mode cost at about 200-300 ms per sentence. Audio contexts keyed by
`context_id` keep ordering and allow per-utterance cancellation.

**Runtime (1.x).** `PipelineWorker(pipeline, params=PipelineParams(...))` and `WorkerRunner().add_workers(worker)`
replace the older `PipelineTask`/`PipelineRunner`. Kick-off is `await worker.queue_frames([LLMRunFrame()])`.
Transports (`LocalAudioTransport`, Daily, LiveKit, FastAPI websocket, SmallWebRTC) supply
`transport.input()` and `transport.output()` processors.

Canonical pipeline:
```python
Pipeline([transport.input(), stt, user_aggregator, llm, tts, transport.output(), assistant_aggregator])
```

### 5.2 LiveKit Agents: plugin interfaces, sessions, turn handling

**Shape.** `AgentSession(vad=..., stt=..., llm=..., tts=..., turn_handling=...)` runs an `Agent`
(instructions plus tools) on a room: `await session.start(agent=agent, room=ctx.room)`. Workers are
registered through `AgentServer()` and `@server.rtc_session()`, then started with `cli.run_app(server)`.
Each provider is a plugin package (`livekit-plugins-<name>`, about 90 exist) implementing abstract base classes:

- **STT** (`livekit/agents/stt/stt.py`): `STT(capabilities=STTCapabilities(streaming, interim_results, diarization=False, aligned_transcript=False, offline_recognize=True, keyterms=False, ...))`.
  - Batch: `async _recognize_impl(buffer: AudioBuffer, *, language, conn_options) -> SpeechEvent`.
  - Streaming: `stream(*, language, conn_options) -> RecognizeStream`. A `RecognizeStream` has `push_frame(rtc.AudioFrame)`, `flush()` and `end_input()`, and is consumed as an async iterator of `SpeechEvent`.
  - `SpeechEventType`: `START_OF_SPEECH, INTERIM_TRANSCRIPT, PREFLIGHT_TRANSCRIPT, FINAL_TRANSCRIPT, RECOGNITION_USAGE, END_OF_SPEECH`.
  - `SpeechData`: `language, text, start_time, end_time, confidence, speaker_id, words, source_languages, ...`. **Language detection results are first-class.**
  - A non-streaming STT is wrapped with `stt.StreamAdapter(stt, vad)`, which uses the VAD to segment audio and then calls `recognize()`.
- **TTS** (`tts/tts.py`): `TTS(capabilities=TTSCapabilities(streaming, aligned_transcript=False), sample_rate, num_channels)`.
  - `synthesize(text) -> ChunkedStream`, implemented through `async _run(self, output_emitter: AudioEmitter)`.
  - `stream() -> SynthesizeStream`, which has `push_text(token)`, `flush()` (segment boundary) and `end_input()`.
  - `AudioEmitter.initialize(request_id, sample_rate, num_channels, mime_type, frame_size_ms=200, stream=False)`, then `push(bytes)`, `flush()`, `end_input()`. It decodes encoded audio (for example mp3) when `mime_type` is not raw PCM. **This is a good pattern for normalizing heterogeneous engine outputs.**
  - A non-streaming TTS becomes streaming through `tts.StreamAdapter(tts=..., sentence_tokenizer=tokenize.blingfire.SentenceTokenizer())`, which buffers pushed tokens into sentences and synthesizes each one. `FallbackAdapter` provides failover across providers.
- **VAD** (`vad.py`): `VAD.stream() -> VADStream` with `push_frame()`. It emits `VADEvent` with type `START_OF_SPEECH | INFERENCE_DONE | END_OF_SPEECH`, plus `speech_duration`, `silence_duration` and `probability`. Built-in `inference.VAD(model="silero")`; plugin `livekit.plugins.silero.VAD.load()`.

**Turn handling** (`voice/turn.py`, consolidated into `TurnHandlingOptions`):
- `turn_detection`: `"stt" | "vad" | "realtime_llm" | "manual"` or a turn-detector object (the protocol
  `predict_end_of_turn(chat_ctx) -> probability` plus per-language `unlikely_threshold`). It can also
  be a streaming audio turn detector, which is now the default `inference.TurnDetector()`.
- `endpointing`: `mode "fixed"|"dynamic"`, `min_delay` 0.5 s, `max_delay` 3.0 s (streaming defaults 0.3/2.5).
  When the EOT model says "unlikely done", the session waits up to `max_delay`; otherwise it waits `min_delay`.
- `interruption`: `enabled`, `mode "adaptive"|"vad"` ("adaptive" is ML-based and tells backchannels
  apart from real interruptions), `min_duration` 0.5 s, `min_words`, `discard_audio_if_uninterruptible`,
  **`resume_false_interruption` (default on) with `false_interruption_timeout` 2.0 s**: if the user
  makes a noise and then stays silent, the agent resumes the speech it paused. `backchannel_boundary`
  is also available.
- `preemptive_generation`: starts the LLM (and optionally TTS) on a preflight or interim transcript
  before end of turn, and discards the result if the transcript changes.

**Barge-in mechanics.** Each agent reply is a `SpeechHandle`. When VAD or adaptive detection fires
during agent speech for at least `min_duration`, the handle is interrupted: LLM and TTS tasks are
cancelled, audio playout stops, and the chat context is truncated to the text that was actually
played (synced transcription). Tools and speech can be marked `allow_interruptions=False`.

### 5.3 RealtimeSTT / RealtimeTTS (KoljaB)

**RealtimeSTT**: one recorder-centered state machine (`AudioToTextRecorder`):
```
mic (PyAudio) or feed_audio()  ->  audio queue (16 kHz mono PCM16)
  -> wake word gate (Porcupine / OpenWakeWord, optional)
  -> VAD: WebRTC VAD (cheap, start detection) + Silero VAD (confirmation; optionally end detection)
  -> pre-roll buffer (keeps audio from before the VAD trigger)
  -> recording until post_speech_silence_duration (0.6 s) of silence
  -> optional realtime lane: a small model re-transcribes the growing buffer every
     realtime_processing_pause (0.2 s) -> text stabilizer -> on_realtime_transcription_update/stabilized
  -> final ASR engine (separate process) -> recorder.text() return value or callback
```
- API: `AudioToTextRecorder(model="tiny", transcription_engine="faster_whisper", language="", enable_realtime_transcription=False, realtime_model_type="tiny", silero_sensitivity=0.4, webrtc_sensitivity=3, post_speech_silence_duration=0.6, early_transcription_on_silence=0, ...)`.
  `recorder.text()` blocks for one utterance, `recorder.text(callback)` delivers it to a callback,
  and `feed_audio(bytes, original_sample_rate=16000)` is for `use_microphone=False`.
  Use `with AudioToTextRecorder() as r:` or call `shutdown()`.
- An empty `language` means engine auto-detection.
- Pluggable engines (`transcription_engines/base.py`: `BaseTranscriptionEngine`, `StreamingTranscriptionSession`, `TranscriptionResult`) include faster-whisper, whisper.cpp, openai-whisper, Moonshine, sherpa-onnx, Parakeet NeMo, Omnilingual, FunASR, Granite/Qwen.
- The recommended CPU profile pairs **streaming Nemotron (fast partials) with a Parakeet-v3 final
  pass**. That two-lane "fast provisional, accurate final" pattern is worth copying.
- `early_transcription_on_silence` starts the final pass speculatively and discards it if speech resumes.
- Barge-in is the application's job: it watches recorder callbacks (`on_vad_start`, `on_recording_start`) and calls `RealtimeTTS.stop()`.

**RealtimeTTS**: `TextToAudioStream(engine)`. `feed()` accepts a string or an **iterator/generator
of LLM tokens**. Then `play()` (blocking) or `play_async()` (background thread), with `pause()`,
`resume()`, `stop()` and `is_playing()`.
- Text is split by **stream2sentence** (default tokenizer `"nltk+rule-based"` consensus, optional Stanza).
- Latency knobs: `fast_sentence_fragment=True` (speak the first clause before the sentence is complete),
  `minimum_first_fragment_length=10`, `minimum_sentence_length=10`, `buffer_threshold_seconds`
  (adaptive: only request more text when buffered audio falls below the threshold),
  `on_audio_chunk` (raw PCM out for websocket use), and `output_wavfile`/`muted`.
- Engine contract (`engines/base_engine.py`): subclass `BaseEngine`, implement
  `get_stream_info() -> (pyaudio_format, channels, sample_rate)`, `synthesize(text) -> bool` (put
  audio chunks on `self.queue`) and `get_voices()`. `stop_synthesis_event` handles cancellation,
  `can_consume_generators` marks engines that take text streams natively, and
  `provides_word_timings`/`timings` carry word timing.
- Utilities: fade-in/out and silence trimming. Engines include Kokoro, Piper, Coqui, Edge, Chatterbox,
  Orpheus, Pocket TTS, NeuTTS, Qwen and others.
- Barge-in: `stream.stop()` sets the stop event, clears queues and stops PyAudio playback.

### 5.4 TEN framework and vocode (brief)

**TEN**: a C/C++ runtime that runs a declarative **graph** (JSON) of *extensions* written in C++, Go,
Python or Node. Extensions exchange typed messages: `cmd`, `data`, `audio_frame` and `video_frame`.
The ecosystem is built around Agora RTC (needs `AGORA_APP_ID`). It provides TEN-VAD and TEN Turn
Detection (text-based finished/unfinished/wait states). Barge-in is handled with a flush command
propagated through the graph. **License: Apache-2.0 plus Agora non-compete conditions**, which is
risky for a redistributable library.

**vocode-core**: `StreamingConversation(output_device, transcriber, agent, synthesizer)` with
`interrupt_sensitivity`. **No meaningful activity since 2024-11.** Use it for ideas only.

### 5.5 Cross-cutting design lessons for the unified library

1. **Interfaces with capability flags** (from LiveKit):
   - `STT.capabilities = {streaming, interim_results, word_timestamps, language_detection}`; `transcribe(audio) -> Result(text, language, language_prob, segments)`; `stream() -> session(push_audio, flush, end_input, async-iter events)`.
   - `TTS.capabilities = {streaming_input, streaming_output, voice_cloning, languages, voices}`; `synthesize(text, voice, lang) -> AudioChunks`; `stream() -> session(push_text, flush, end_input)`.
   - `VAD.stream()` emits `START/END/PROB` events.
   - `TurnDetector.predict(audio_turn | chat_ctx) -> p_end`.
2. **Adapters as the universal fix.** A VAD-segmented STT adapter makes any batch STT streaming (Pipecat
   `SegmentedSTTService`, LiveKit `stt.StreamAdapter`). A sentence-tokenizing TTS adapter makes any
   batch TTS streaming (LiveKit `tts.StreamAdapter`, RealtimeTTS). Make the sentence tokenizer
   pluggable and language-aware: blingfire, nltk, or regex that handles CJK punctuation `。！？`.
3. **Normalize audio** at the boundary: PCM16 or float32, explicit `sample_rate` (engines output 16k,
   22.05k, 24k or 48k; edge-tts outputs MP3). Use a LiveKit-style `AudioEmitter` that decodes and
   resamples, plus fades and silence trimming (from RealtimeTTS).
4. **Interruption = cancellation token + out-of-band priority channel.** Never queue an interrupt
   behind audio (Pipecat SystemFrame). Cancel in-flight synthesis, flush the playout buffer, and
   truncate the assistant transcript to the words actually played.
5. **False-interruption and backchannel handling**: `min_duration`/`min_words` before interrupting;
   pause instead of stopping; resume after about 2 s of silence (LiveKit).
6. **Layered endpointing**: VAD silence (about 200-600 ms), then the Smart Turn probability, then
   `min_delay`/`max_delay`. Optionally start speculative or preemptive LLM generation on a stable
   interim transcript.
7. **Language routing.** Whisper and Qwen3-ASR return a detected language, which can be fed straight
   into TTS voice selection (for example the kokoro voice prefix, Piper voice, Chatterbox
   `language_id`, or an edge-tts `VoicesManager.find(Language=...)`). Keep a
   `lang -> {engine: voice}` registry, and use ISO-639-1 with BCP-47 mapping, because the engines
   disagree: `en-us` (kokoro-onnx), `a` (kokoro), `en_US` (piper), `en-US-...Neural` (edge),
   `eng` (MMS), `eng_Latn` (omnilingual), `"English"` (Qwen).
8. **License-tiered extras**: `cachos[permissive]` by default. Make `[gpl]` (piper-tts), `[nc]`
   (xtts, f5, mms) and `[cloud]` (edge-tts) explicit opt-ins, and warn at load time.

---

## 6. License red-flag checklist

| Component | Issue |
|---|---|
| XTTS-v2 (`coqui/XTTS-v2`) | **CPML: non-commercial only** (model and outputs) |
| F5-TTS pretrained weights | **CC-BY-NC-4.0** |
| Meta MMS-TTS / MMS-1b-all ASR | **CC-BY-NC-4.0** |
| Moonshine legacy non-English non-streaming models | Moonshine Community License (**NC**) |
| ChatTTS | **AGPL-3.0** code, NC weights |
| Piper (`piper-tts`, piper1-gpl) | **GPL-3.0**: linking/importing affects your license; consider subprocess isolation or sherpa-onnx |
| edge-tts | **LGPL-3.0**; relies on an undocumented Microsoft endpoint (ToS and availability risk) |
| Fish Audio (fish-speech, S2-Pro, Drama 3) | Community license: commercial use only under USD 500k revenue, **no hosted TTS services**; S2-Pro weights are research license |
| IndexTTS-2/2.5 | bilibili license: written license needed above 100M MAU or RMB 1B revenue; cannot be used to improve other models |
| NeuTTS (code) | Free for commercial use only under USD 5M revenue (Air/Nano weights tagged Apache-2.0 on HF) |
| Supertonic weights | OpenRAIL (use-based restrictions) |
| LiveKit turn-detector models | Usable **only with LiveKit Agents** |
| TEN framework / TEN-VAD / TEN turn detection | Apache-2.0 **plus Agora non-compete clauses** |
| SenseVoice weights | FunASR model license (permissive with attribution; read it) |
| NeMo Parakeet/Canary, Kyutai STT/TTS, Pocket TTS weights | CC-BY-4.0 (commercial OK, **attribution required**) |
| Permissive and clean | faster-whisper, Whisper, whisper.cpp, Kokoro / kokoro-onnx, Chatterbox, Silero VAD, Smart Turn, Qwen3-ASR/TTS, CosyVoice3, VoxCPM2, Omnilingual ASR, Orpheus, Dia, MeloTTS/OpenVoice, VibeVoice |

---

## 7. Turn-detection models: exact integration details

### 7.1 Pipecat Smart Turn v3 (verified by loading the ONNX file locally)
- **HF repo:** `pipecat-ai/smart-turn-v3` (license tag **bsd-2-clause**). Files: `smart-turn-v3.0.onnx`,
  `smart-turn-v3.1-cpu.onnx`, `smart-turn-v3.1-gpu.onnx`, **`smart-turn-v3.2-cpu.onnx`** (8.7 MB, int8),
  `smart-turn-v3.2-gpu.onnx` (fp32). (The repo's `inference.py` still points at `smart-turn-v3.1.onnx`. Use v3.2.)
- **ONNX I/O (inspected with onnxruntime):** input `input_features` float32 `[batch, 80, 800]`,
  output `logits` float32 `[batch, 1]`. Despite the name, **the output is already a sigmoid
  probability** (per `inference.py`): `p > 0.5` means the turn is complete.
- **Features:** `transformers.WhisperFeatureExtractor(chunk_length=8)`, which gives an 80-bin log-mel
  at 100 frames/s, so 800 frames = 8 s. Called with `sampling_rate=16000, return_tensors="np", padding="max_length",
  max_length=8*16000, truncation=True, do_normalize=True`.
- **Input audio:** 16 kHz mono float32. Keep the **last 8 s**; shorter audio is **zero-padded at the
  beginning** (`truncate_audio_to_last_n_seconds`).
- Pipecat's `LocalSmartTurnAnalyzerV3(smart_turn_model_path=None, cpu_count=1)` bundles
  `smart-turn-v3.2-cpu` and computes Whisper features in **numpy** (`_whisper_features.py`), so it needs
  `onnxruntime` and `soxr` and **does not need torch or transformers**.
- **Languages (23):** ar, bn, zh, da, nl, de, en, fi, fr, hi, id, it, ja, ko, mr, no, pl, pt, ru, es, tr, uk, vi.
- **Published accuracy (v3.2-cpu benchmark, 31,527 samples):** overall accuracy **92.63%**, precision 0.909,
  recall 0.947, F1 0.927, FPR 4.73%, FNR 2.64%. Per language: ko 96.96, tr 96.79, de 96.37,
  ja 96.16, nl 95.92, pl 95.38, id 95.16, it 94.50, pt 94.49, **en 94.26**, uk 94.19, fi 94.16,
  fr 94.09, ru 93.53, da 92.81, no 91.81, hi 90.11, ar 89.76, **es 89.57**, zh 85.79, bn 83.80,
  mr 82.43, **vi 79.38**.
- **Inference:** about 10 ms on some CPUs, under 100 ms on most cloud instances, about 65 ms on a Pipecat Cloud 1x instance.

Minimal standalone adapter (no Pipecat):
```python
import numpy as np, onnxruntime as ort
from transformers import WhisperFeatureExtractor          # or port pipecat's numpy mel to drop transformers
fe = WhisperFeatureExtractor(chunk_length=8)
sess = ort.InferenceSession("smart-turn-v3.2-cpu.onnx", providers=["CPUExecutionProvider"])
def p_turn_complete(audio16k: np.ndarray) -> float:
    a = audio16k[-8*16000:]
    a = np.pad(a, (8*16000 - len(a), 0)) if len(a) < 8*16000 else a
    f = fe(a, sampling_rate=16000, return_tensors="np", padding="max_length",
           max_length=8*16000, truncation=True, do_normalize=True).input_features.astype(np.float32)
    return float(sess.run(None, {"input_features": f})[0][0, 0])
```

### 7.2 LiveKit turn detector (text EOU model)
- **HF repo:** `livekit/turn-detector`, which is Qwen2.5-0.5B-Instruct distilled from a fine-tuned
  Qwen2.5-7B. The plugin downloads **`onnx/model_q8.onnx`** (int8) at revision **`v0.4.1-intl`**
  (multilingual) or `v1.2.2-en` (English, deprecated), plus `languages.json` (per-language thresholds)
  and the tokenizer from the same revision. (The `main` branch carries an older `model_quantized.onnx`.)
- **Input format** (from `livekit-plugins-turn-detector/base.py`):
  1. Take the last **6 turns**.
  2. Normalize each message: NFKC, lowercase, strip punctuation except `'` and `-`, collapse whitespace.
  3. Merge adjacent same-role messages.
  4. Apply `tokenizer.apply_chat_template(msgs, add_generation_prompt=False, add_special_tokens=False, tokenize=False)`.
  5. **Cut the text at the last `<|im_end|>`**, so the final user turn is left open.
  6. Tokenize with `max_length=128, truncation=True`.
  7. Run `session.run(None, {"input_ids": ids.astype(int64)})`. The EOU probability is `outputs[0].flatten()[-1]`, which is P(`<|im_end|>` next).
- **Languages (14):** en, es, fr, de, it, pt, nl, zh, ja, ko, id, tr, ru, hi.
- **Published accuracy (TPR / TNR):** en 99.3 / 87.0, hi 99.4 / 96.3, ko 99.3 / 94.5, fr 99.3 / 88.9,
  es 99.3 / 86.0, zh 99.3 / 86.6, it 99.3 / 85.1. v0.4.1 claims a **39.23% relative improvement** on
  structured inputs (emails, phone numbers, addresses).
- **Cost:** about 50-160 ms per turn on CPU, under 500 MB RAM, 396 MB on disk.
- **License: "LiveKit Model License", which allows use only together with LiveKit Agents.** We cannot
  ship it in a framework-neutral library.
- **Status:** this text model is **deprecated** (to be removed in SDK 2.0). It is replaced by the audio
  `livekit.agents.inference.TurnDetector`:
  - `version="v1"` runs on LiveKit Cloud (free on Cloud-deployed agents, billed per request when self-hosted beyond a development allowance).
  - `version="v1-mini"` runs locally on CPU through the `livekit-local-inference` package (free, same model license).
  - The audio detector covers 14 languages: the text list minus ru, plus ar.

### 7.3 Head-to-head numbers: LiveKit eot-bench (vendor-run, but open and reproducible)
`livekit/eot-bench` (Apache-2.0; dataset `livekit/eot-bench-data` covers real human-to-agent turns in
14 languages) scores models at real pauses. The English table is below. Lower is better: a "false
cutoff" means the agent interrupted the user, and "latency" is endpointing delay after the true end of turn, not inference time.

| Model | False cutoffs @300 ms | @600 ms | Latency @5% cutoffs | @10% |
|---|---:|---:|---:|---:|
| LiveKit Turn Detector v1 (cloud) | 9.9% | 4.5% | 543 ms | 295 ms |
| LiveKit v1-mini (local) | 27.8% | 12.1% | 1070 ms | 698 ms |
| **SmartTurn v3.2** | 35.2% | 14.8% | 1051 ms | 739 ms |
| Deepgram Flux | 12.9% | 9.9% | 1151 ms | 548 ms |
| VAD-only baseline | 55.6% | 21.7% | 1600 ms | 1000 ms |

Interpretation: run locally, **v1-mini and SmartTurn v3.2 are close**, and both are far better than
VAD-only. The big win (v1) is a **paid cloud model**. Caveats: the benchmark was built by LiveKit to
showcase v1, and SmartTurn's own benchmark uses a different, mostly synthetic dataset and per-clip accuracy.

---

## 8. Real-time noise suppression (decision: integrate, do not write DSP)

### 8.1 Comparison

| Option | Quality (published) | CPU cost | Frame / latency | SR | Streaming in Python | License | pip |
|---|---|---|---|---|---|---|---|
| **GTCRN** (Xiaobin-Rong/gtcrn, ICASSP'24, 762 stars) | VB-DEMAND **PESQ 2.87**, SI-SNR 18.83, STOI 0.940. DNS3 blind DNSMOS OVRL 2.70 (RNNoise 2.53) | 48.2K params, 33 MMACs/s. **Measured here: RTF 0.051 single thread, 0.82 ms per 16 ms hop** | hop 256 (16 ms), window 512, **32 ms algorithmic latency** | **16 kHz native** | **Yes**: stateful ONNX (`gtcrn_simple.onnx`), plain onnxruntime, or sherpa-onnx `OnlineSpeechDenoiser` | **MIT** | none (onnxruntime + numpy) or `sherpa-onnx` |
| **DPDFNet** (ceva-ip/DPDFNet, arXiv 2512.16420; DeepFilterNet2 plus dual-path RNN) | Paper/eval set published; no numbers in README (claims gains over DFN2). 16 kHz `baseline` = 0.36 GMACs; `dpdfnet2/4/8` = 1.35/2.36/4.37 GMACs | ONNX CPU; dpdfnet2 is about 4x baseline compute | 10 ms hop, ~20 ms first window | **8 / 16 / 48 kHz models**; `StreamEnhancer` resamples internally | **Yes**: `dpdfnet.StreamEnhancer` (any chunk size), also sherpa-onnx | **Apache-2.0** (code and weights) | `dpdfnet` (0.6.0, py>=3.11; deps numpy, librosa, soundfile, onnxruntime) |
| **RNNoise** via `pyrnnoise` (xiph/rnnoise BSD-3; binding Apache-2.0) | VB-DEMAND PESQ ~2.29, DNSMOS OVRL 2.53 (both from the GTCRN table). Classic, mild | Tiny (~0.04 GMACs) | 480 samples @48 kHz = 10 ms | 48 kHz native; the binding resamples (libav graph via `audiolab`) | Yes: `denoise_chunk` / `denoise_frame` | BSD-3 / Apache-2.0 | `pyrnnoise` (0.4.5). Pipecat `RNNoiseFilter` uses it |
| **DeepFilterNet3** (Rikorose, 4.9k stars) | VB-DEMAND PESQ ~3.17 (paper-reported) | Rust/tract real-time on CPU; Python path needs torch | 10 ms hop @48 kHz, ~40 ms lookahead in default model | **48 kHz only** (resample 16 to 48 to 16) | **No official Python streaming.** `deepfilterlib` only exposes STFT/ERB features. Streaming lives in the Rust `deep-filter` binary / LADSPA. Unofficial `grazder/DeepFilterNet` (branch `torchDF_main`) exports a streaming `denoiser_model.onnx`, but it needs a custom `ai.onnx.contrib::ComplexMul` op (onnxruntime-extensions) and a re-tuned hop of 512 | MIT / Apache-2.0 | `deepfilternet` (0.5.6, **last release 2024**, pins `numpy<2`, `packaging<24`, needs torch + torchaudio) |
| FRCRN / ZipEnhancer / MossFormer2 (ModelScope ClearerVoice-Studio, 4.6k stars) | FRCRN VB-DEMAND PESQ ~3.2; ZipEnhancer and MossFormer2 claim higher (~3.5+) | GPU-oriented, heavy | offline | FRCRN_SE_16K, MossFormerGAN_SE_16K, MossFormer2_SE_48K | No (offline) | Apache-2.0 | `clearvoice` (0.1.2) |
| facebookresearch/denoiser (Demucs) | VB PESQ ~2.9 causal / ~3.1 non-causal (paper) | heavier than above | causal ~40 ms | 16 kHz | causal model can stream | CC-BY-NC (weights); **repo archived** | `denoiser` |
| resemble-ai/resemble-enhance | enhancement and super-resolution; no standard metrics | GPU, slow | offline | 44.1 kHz | No | MIT | `resemble-enhance` (0.0.1, stale) |
| Krisp VIVA, ai-coustics, Picovoice Koala, LiveKit BVC | commercial, strong | - | - | - | SDKs | **Proprietary**. LiveKit BVC/NC (`livekit-plugins-noise-cancellation`) **requires a LiveKit Cloud project** | context only |

### 8.2 Recommendation
- **Default for real-time agents: GTCRN** (MIT, 16 kHz native, 32 ms latency, RTF ~0.05 on one core,
  only onnxruntime + numpy, 535 KB model). It beats RNNoise on both VB-DEMAND and DNS3 at similar
  compute. Use it directly (adapter below) or through `sherpa-onnx.OnlineSpeechDenoiser`.
  **Upgrade tier:** `dpdfnet` (`dpdfnet2`/`dpdfnet4` at 16 kHz). It is also Apache-2.0 with a ready
  `StreamEnhancer`, at about 4-12x GTCRN's compute but still CPU real-time. Expose it as an option and pick by benchmark on your audio.
- **Max accuracy, offline:** ClearerVoice `MossFormer2_SE_48K` (Apache-2.0, GPU), or DeepFilterNet3
  through `deepfilternet` if a torch dependency and old pins are acceptable.
- **Important for ASR (evidence):** front-end enhancement often **hurts Whisper-class ASR**.
  - Chondhekar et al. 2025 (arXiv 2512.17562): MetricGAN+ made Whisper worse in all 40 configurations. At 10 dB SNR, semantic WER went from 8.82% to 25.83%.
  - Islam et al. 2026 (arXiv 2603.04710): SAM-Audio preprocessing raised Whisper WER/CER, and more so for larger models.
  - DENOASR 2024 (arXiv 2410.16712): results were mixed; Demucs-based pipelines sometimes helped.
  - **Design:** feed the **denoised signal to VAD / turn detection / barge-in** (cuts false
    interruptions from noise). Feed **raw (or attenuation-limited) audio to ASR by default**, with a
    per-pipeline flag and an A/B WER test. Use attenuation limits where offered (DPDFNet
    `attn_limit_db`, DFN `atten_lim_db`).

### 8.3 Exact APIs

**GTCRN via plain onnxruntime (streaming 16 kHz; verified locally).** The output matches the repo's
offline `enh.wav` to within a mean abs diff of 1.2e-4 at a 256-sample delay. RTF 0.051.
Model: `https://github.com/Xiaobin-Rong/gtcrn/raw/main/stream/onnx_models/gtcrn_simple.onnx` (DNS3-trained), also in sherpa-onnx release `speech-enhancement-models`.
ONNX I/O: inputs `mix [1,257,1,2]` (one STFT frame, real/imag), `conv_cache [2,1,16,16,33]`,
`tra_cache [2,3,1,1,16]`, `inter_cache [2,1,33,16]`. Outputs `enh [1,257,1,2]` and `*_out` caches.
STFT: n_fft 512, hop 256, sqrt-Hann window.
```python
import numpy as np, onnxruntime as ort

class GTCRNStreamer:
    N_FFT, HOP = 512, 256
    def __init__(self, model_path: str):
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.win = np.sqrt(np.hanning(self.N_FFT + 1)[:-1]).astype(np.float32)
        self.reset()
    def reset(self):
        self.conv = np.zeros((2, 1, 16, 16, 33), np.float32)
        self.tra = np.zeros((2, 3, 1, 1, 16), np.float32)
        self.inter = np.zeros((2, 1, 33, 16), np.float32)
        self.in_buf = np.zeros(self.N_FFT, np.float32); self.ola = np.zeros(self.N_FFT, np.float32)
        self.pending = np.zeros(0, np.float32)
    def _hop(self, x):
        self.in_buf = np.concatenate([self.in_buf[self.HOP:], x])
        spec = np.fft.rfft(self.in_buf * self.win)
        mix = np.stack([spec.real, spec.imag], -1).astype(np.float32)[None, :, None, :]
        enh, self.conv, self.tra, self.inter = self.sess.run(None, {
            "mix": mix, "conv_cache": self.conv, "tra_cache": self.tra, "inter_cache": self.inter})
        y = np.fft.irfft(enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1], n=self.N_FFT).astype(np.float32) * self.win
        self.ola += y
        out = self.ola[:self.HOP].copy()
        self.ola = np.concatenate([self.ola[self.HOP:], np.zeros(self.HOP, np.float32)])
        return out
    def process(self, chunk: np.ndarray) -> np.ndarray:   # float32 @16 kHz, any length
        self.pending = np.concatenate([self.pending, chunk.astype(np.float32)])
        n = len(self.pending) // self.HOP
        out = [self._hop(self.pending[i*self.HOP:(i+1)*self.HOP]) for i in range(n)]
        self.pending = self.pending[n*self.HOP:]
        return np.concatenate(out) if out else np.zeros(0, np.float32)
# whole file: GTCRNStreamer(path).process(audio) (+ process(np.zeros(512)) to flush the tail)
```

**GTCRN / DPDFNet via sherpa-onnx** (`pip install sherpa-onnx`, 2.0.1). Models come from the
`speech-enhancement-models` release: `gtcrn_simple.onnx`, `dpdfnet_baseline.onnx`, `dpdfnet2.onnx`,
`dpdfnet4.onnx`, `dpdfnet8.onnx`, `dpdfnet2_48khz_hr.onnx`.
```python
import numpy as np, sherpa_onnx
model_cfg = sherpa_onnx.OfflineSpeechDenoiserModelConfig(
    gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model="gtcrn_simple.onnx"),
    # dpdfnet=sherpa_onnx.OfflineSpeechDenoiserDpdfNetModelConfig(model="dpdfnet2.onnx"),
    num_threads=1, provider="cpu", debug=False)
# (a) streaming
sd = sherpa_onnx.OnlineSpeechDenoiser(sherpa_onnx.OnlineSpeechDenoiserConfig(model=model_cfg))
hop = sd.frame_shift_in_samples
out = [np.asarray(sd(chunk, 16000).samples, np.float32) for chunk in chunks]   # chunk: float32 ndarray
out.append(np.asarray(sd.flush().samples, np.float32))                           # output rate: sd.sample_rate
# (b) whole file
off = sherpa_onnx.OfflineSpeechDenoiser(sherpa_onnx.OfflineSpeechDenoiserConfig(model=model_cfg))
res = off(samples_float32, 16000); enhanced, sr = res.samples, res.sample_rate
```

**DPDFNet** (`pip install dpdfnet`; models download on demand)
```python
import dpdfnet
# (a) streaming: per-stream state, resamples internally, returns the same rate it is given
enh = dpdfnet.StreamEnhancer(model="dpdfnet2")          # 16 kHz family: baseline, dpdfnet2, dpdfnet4, dpdfnet8
y = enh.process(chunk_float32, sample_rate=16000)       # may return empty until the first ~20 ms window fills
tail = enh.flush(); enh.reset()                         # reset between independent segments
# (b) whole file / array
y = dpdfnet.enhance(audio, sample_rate=sr, model="dpdfnet4", attn_limit_db=12)
dpdfnet.enhance_file("noisy.wav", model="dpdfnet2", attn_limit_db=12)
```

**RNNoise** (`pip install pyrnnoise`)
```python
import numpy as np
from pyrnnoise import RNNoise
dn = RNNoise(sample_rate=16000)                          # resamples to 48 kHz internally and back
# (a) streaming: int16 ndarray [channels, samples] (1-D is promoted); yields per 10 ms frame
for speech_prob, frame in dn.denoise_chunk(chunk_int16[None, :]):   # partial=True on the final chunk
    ...                                                  # frame: int16 [channels, n]; speech_prob doubles as a VAD
# (b) whole file
for p in RNNoise(sample_rate=48000).denoise_wav("in.wav", "out.wav"): pass
```

**DeepFilterNet3** (`pip install torch torchaudio deepfilternet`; offline, 48 kHz)
```python
from df.enhance import enhance, init_df, load_audio, save_audio
model, df_state, _ = init_df()                           # DeepFilterNet3 by default; init_df(post_filter=False, ...)
audio, _ = load_audio("noisy.wav", sr=df_state.sr())     # torch [C, T] at 48 kHz
enhanced = enhance(model, df_state, audio, pad=True, atten_lim_db=None)
save_audio("enhanced.wav", enhanced, df_state.sr())
```

**ClearerVoice** (`pip install clearvoice`; offline, max quality)
```python
from clearvoice import ClearVoice
cv = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])   # or FRCRN_SE_16K, MossFormerGAN_SE_16K
out = cv(input_path="in.wav", online_write=False); cv.write(out, output_path="out.wav")
```

---

## 9. faster-whisper: hallucination control and domain vocabulary (verified in `faster_whisper/transcribe.py`, v1.2.1)

`WhisperModel.transcribe(...)` parameter names and defaults:

| Param | Default | Effect |
|---|---|---|
| `no_speech_threshold` | `0.6` | If `no_speech_prob` exceeds this **and** avg logprob is below `log_prob_threshold`, the segment is treated as silence |
| `log_prob_threshold` | `-1.0` | Average-logprob floor; below it, decoding retries at higher temperature |
| `compression_ratio_threshold` | `2.4` | gzip ratio above this (repetition loops) triggers a temperature fallback |
| `temperature` | `[0.0, 0.2, ..., 1.0]` | Fallback schedule |
| `condition_on_previous_text` | `True` | Set **False** for agents (stops hallucination propagation) |
| `prompt_reset_on_temperature` | `0.5` | Drop the prompt when fallback temperature exceeds this |
| `hallucination_silence_threshold` | `None` | **Requires `word_timestamps=True`.** Skips silent periods longer than N s when a hallucination is suspected |
| `hotwords` | `None` | Hint phrases inserted as a "previous text" prompt (`sot_prev` + tokens, capped at max_length/2). **No effect if `prefix` is set** |
| `initial_prompt` | `None` | Classic prompt biasing (str or token ids) |
| `suppress_blank` / `suppress_tokens` | `True` / `[-1]` | Token suppression |
| `repetition_penalty`, `no_repeat_ngram_size` | `1`, `0` | Anti-loop decoding knobs |
| `vad_filter`, `vad_parameters` | `False`, `None` | Bundled Silero VAD. The strongest anti-hallucination measure for silence/noise |
| `clip_timestamps` | `"0"` | Process only listed spans (disables `vad_filter`) |
| `multilingual` | `False` | Per-segment language detection |
| `language_detection_threshold`, `language_detection_segments` | `0.5`, `1` | LID robustness |

Agent-friendly preset: `transcribe(audio, language=None, beam_size=1-5, vad_filter=True, condition_on_previous_text=False, no_speech_threshold=0.6, log_prob_threshold=-1.0, compression_ratio_threshold=2.4, hotwords="Cachos, Kubernetes, ...")`.
Also drop segments whose `no_speech_prob` is high, as Pipecat's `WhisperSTTService` does with its `no_speech_prob` setting.

---

## 10. Runtime decision: Pipecat vs LiveKit Agents (evidence-based)

### 10.1 Measured / published numbers
| Dimension | Pipecat | LiveKit Agents |
|---|---|---|
| Stars / forks | 16,930 / 3,002 | 15,002 / 4,032 |
| License | BSD-2 (everything, including Smart Turn weights) | Apache-2.0 framework; **turn-detector models under LiveKit Model License (LiveKit Agents only)**; BVC noise cancellation needs LiveKit Cloud |
| Releases (last 90 days) | 10 (v1.11 to v1.15.0, 2026-10-29) | 16 (1.8.x patch train, 1.8.10 on 2026-10-29) |
| Open issues / open PRs | 133 / 348 | 246 / 631 |
| Contributors | 359 | 100+ (API cap) |
| Default end-of-turn | Smart Turn v3.2, **local**, BSD-2: 92.6% accuracy over 23 langs (own benchmark); eot-bench en: 35.2% false cutoffs @300 ms, 1051 ms @5% | `inference.TurnDetector` v1 (**cloud**): eot-bench en 9.9% @300 ms, 543 ms @5%. Local `v1-mini`: 27.8% / 1070 ms |
| Turn-model languages | 23 | 14 |
| Voice-to-voice latency | **No reproducible framework benchmark published.** Pipecat targets ~1.5 s voice-to-voice for phone agents (PhoneBench) and publishes `pipecat-ai/stt-benchmark` (STT TTFB and semantic WER, 23 models) | **No reproducible framework benchmark published.** eot-bench measures endpointing only |

Bottom line on latency: both frameworks add only a few ms of orchestration. End-to-end latency is
dominated by VAD stop time + turn model + STT finalization + LLM TTFT + TTS TTFB + transport. Any
claim needs our own measurement (p50/p95 first-audio latency) with a fixed model stack.

### 10.2 Interruption handling
- **Pipecat:** interruptions are user-turn *start strategies*.
  - Defaults: `VADUserTurnStartStrategy` + `TranscriptionUserTurnStartStrategy`.
  - `MinWordsUserTurnStartStrategy(min_words=N, use_interim=True)` replaces the old
    `MinWordsInterruptionStrategy` (that name no longer exists in the source). It only counts as a
    user turn, and therefore an interruption, while the bot is speaking once N words are transcribed.
  - Also available: `WakePhraseUserTurnStartStrategy`, Krisp/AIC VAD-based starts, user-mute
    strategies (for example, mute during function calls or the first bot message), and
    `EagerUserTurnStrategies` (speculative).
  - Mechanism: a start broadcasts `InterruptionFrame` (SystemFrame). Processors cancel their process
    tasks, TTS drops audio contexts, the output transport flushes, and the context keeps only the spoken words.
  - **There is no built-in "resume after false interruption".**
- **LiveKit:** `TurnHandlingOptions.interruption`:
  - `mode "adaptive"` (ML backchannel classifier) or `"vad"`; `min_duration` 0.5 s; `min_words` (STT);
  - `resume_false_interruption=True` with `false_interruption_timeout=2.0`: the agent **pauses**
    playout on overlap, and if no transcript or speech follows within the timeout, it **resumes** the
    same utterance. `backchannel_boundary=(1.0, 1.0)` suppresses backchannels near turn edges.
  - `discard_audio_if_uninterruptible`; per-`SpeechHandle` `allow_interruptions`.
  - "Adaptive" mode depends on LiveKit's models (cloud or local under the LiveKit license).

### 10.3 Noise filters
- **Pipecat:** `BaseAudioFilter` (`start(sample_rate)`, `filter(bytes) -> bytes`, `stop()`), plugged into
  the transport params (`audio_in_filter=`). Built-ins: `RNNoiseFilter` (pyrnnoise, soxr resampling to
  48 kHz), `KrispVivaFilter` (proprietary SDK), `KoalaFilter` (Picovoice, key needed), `AICFilter`
  (ai-coustics, key needed). **A GTCRN/DPDFNet filter is easy to add** as a `BaseAudioFilter` subclass.
- **LiveKit:** `livekit-plugins-noise-cancellation` (BVC/NC/BVCTelephony) **requires LiveKit Cloud**.
  `livekit-plugins-krisp` (proprietary VIVA SDK) and ai-coustics need your own key. Third-party
  `livekit-plugins-denoise` (MIT). Custom denoising means writing an audio FrameProcessor in the room input path.

### 10.4 Infrastructure
- **Pipecat:** no server required. Transports: `LocalAudioTransport` (mic/speaker), **SmallWebRTC**
  (peer-to-peer WebRTC served from your own Python process), FastAPI **WebSocket** (Twilio/Telnyx/Plivo
  serializers), and optional Daily, LiveKit, Vonage and WhatsApp. **Daily is optional.**
- **LiveKit:** an agent is a worker that joins a **LiveKit room**, so it needs a LiveKit server
  (self-hostable `livekit/livekit`, Apache-2.0, 21.9k stars) or LiveKit Cloud. A `console` mode exists
  for local terminal testing. Phone calls need LiveKit SIP. Some features (BVC, v1 turn detector) are Cloud-tied.

### 10.5 Plugging in a custom STT / TTS (verified against current source)
**Pipecat**
```python
from pipecat.services.stt_service import SegmentedSTTService      # batch STT: VAD-segmented WAV bytes
from pipecat.services.tts_service import TTSService
from pipecat.frames.frames import TranscriptionFrame, ErrorFrame
from pipecat.utils.time import time_now_iso8601

class MySTT(SegmentedSTTService):
    async def run_stt(self, audio: bytes):                         # AsyncGenerator[Frame | None]
        text, lang = await asyncio.to_thread(my_engine.transcribe, audio)
        yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), lang)

class MyTTS(TTSService):                                           # sentence aggregation is done by the base class
    async def run_tts(self, text: str, context_id: str):
        async for frame in self._stream_audio_frames_from_iterator(
                my_pcm16_async_iter(text), in_sample_rate=24000, context_id=context_id):
            yield frame                                            # base emits TTSAudioRawFrame (+ started/stopped)
```
(The `_stream_audio_frames_from_iterator` helper is what `PiperTTSService` uses. It is underscore-prefixed, so pin the Pipecat version.)
Streaming STT subclasses `STTService` (or `WebsocketSTTService`) and pushes `InterimTranscriptionFrame`/`TranscriptionFrame`.

**LiveKit**
```python
from livekit.agents import stt, tts, utils
from livekit.agents.types import APIConnectOptions

class MySTT(stt.STT):
    def __init__(self):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
    async def _recognize_impl(self, buffer, *, language, conn_options: APIConnectOptions) -> stt.SpeechEvent:
        text, lang = ...
        return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                               alternatives=[stt.SpeechData(language=lang, text=text)])
# use in a streaming session: stt.StreamAdapter(stt=MySTT(), vad=silero_vad)

class MyTTS(tts.TTS):
    def __init__(self):
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=24000, num_channels=1)
    def synthesize(self, text, *, conn_options=...):
        return MyChunkedStream(tts=self, input_text=text, conn_options=conn_options)

class MyChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter):
        output_emitter.initialize(request_id=utils.shortuuid(), sample_rate=24000, num_channels=1, mime_type="audio/pcm")
        async for pcm in my_pcm16_async_iter(self.input_text):
            output_emitter.push(pcm)
        output_emitter.flush()
# streaming: tts.StreamAdapter(tts=MyTTS(), sentence_tokenizer=tokenize.blingfire.SentenceTokenizer())
```
(Verified in source: `SpeechEvent(type, request_id="", alternatives=[...])`,
`ChunkedStream.__init__(*, tts, input_text, conn_options)` with an `input_text` property,
`TTS.__init__(*, capabilities, sample_rate, num_channels)`, `utils.shortuuid`, and `APIConnectOptions` exported from `livekit.agents`.)

### 10.6 Local Smart Turn and torch-free Silero in Pipecat
Yes to both. `LocalSmartTurnAnalyzerV3` bundles `smart-turn-v3.2-cpu.onnx` and uses onnxruntime +
soxr + numpy features. `SileroVADAnalyzer` loads a bundled `silero_vad.onnx` through **onnxruntime
(no torch)**. Both are the defaults in `LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())` and `UserTurnStrategies`.

### 10.7 Recommendation
**Use Pipecat as the runtime, or more precisely, mirror Pipecat's frame model and keep an optional
LiveKit transport/adapter.** Reasons:
1. **Licensing and portability.** Pipecat's whole local stack (Smart Turn weights, Silero ONNX, RNNoise
   filter) is permissive and runs fully offline. LiveKit's differentiators (v1 turn detector, adaptive
   interruption, BVC) are cloud-tied or restricted to LiveKit Agents, which conflicts with a neutral library.
2. **Infrastructure.** Pipecat runs with no media server (local audio, SmallWebRTC, WebSocket telephony).
   LiveKit always needs a LiveKit server.
3. **Turn quality on local-only stacks is comparable.** On eot-bench, v1-mini (local) and SmartTurn v3.2
   are close. Only the cloud v1 is clearly better. Smart Turn covers more languages (23 vs 14).
4. **Extensibility.** `run_stt`/`run_tts` plus `BaseAudioFilter` are small, explicit seams that fit our
   GTCRN/DPDFNet, faster-whisper and kokoro-onnx adapters. Pipecat already ships Whisper, Piper,
   Kokoro, XTTS, Moonshine and Pocket TTS services to learn from.
5. **What to borrow from LiveKit:** capability-flagged STT/TTS interfaces, `AudioEmitter`-style output
   normalization, and especially **false-interruption pause/resume** (`false_interruption_timeout`
   ~2 s), which Pipecat lacks. Implement it in our own output processor.

Choose LiveKit instead if the product is WebRTC-room-centric, already on LiveKit Cloud, and the best
available turn detection (v1, ~9.9% false cutoffs @300 ms vs ~35% for local models) is worth the vendor coupling.

Sources: [eot-bench](https://github.com/livekit/eot-bench), [LiveKit turn detector docs](https://docs.livekit.io/agents/logic/turns/turn-detector/),
[smart-turn-v3 benchmarks](https://huggingface.co/pipecat-ai/smart-turn-v3), [livekit-plugins-noise-cancellation (PyPI)](https://pypi.org/project/livekit-plugins-noise-cancellation/),
[LiveKit self-hosting](https://docs.livekit.io/transport/self-hosting/), [Pipecat vs LiveKit (forasoft)](https://www.forasoft.com/blog/article/pipecat-vs-livekit-agents),
[Pipecat STT benchmark write-up](https://hackernoon.com/pipecat-benchmarked-23-real-time-stt-models-for-voice-agents-there-isnt-one-winner?source=rss),
[When De-noising Hurts (arXiv 2512.17562)](https://arxiv.org/abs/2512.17562v1), [SAM-Audio + Whisper (arXiv 2603.04710)](https://arxiv.org/abs/2603.04710),
[DENOASR (arXiv 2410.16712)](https://arxiv.org/pdf/2410.16712).
