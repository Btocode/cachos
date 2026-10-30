# cachos

**Multilingual speech-to-text, text-to-speech and real-time voice agents — accuracy first, built on proven open source.**

```python
from cachos import STT, TTS, VoiceAgent

STT().transcribe_sync("call.mp3").text                    # language detected automatically
TTS().synthesize_sync("Bonjour à tous !").save("fr.wav")   # French voice picked automatically
VoiceAgent(llm_model="gpt-4.1-mini").run_local()           # talk to it: mic + speakers
```

- **535 voices, 80 languages** out of the box (Kokoro, Piper, Edge), plus voice cloning (Chatterbox, XTTS) and 1,100+ languages via MMS.
- **99-language speech recognition** with automatic language detection (Whisper large-v3).
- **A voice agent that replies in the language the user spoke**, with noise-robust turn detection, interruptions, and browser / phone / local transports.
- **Use it from any stack**: Python library, OpenAI-compatible REST API, TypeScript client, Pipecat services, Docker.

cachos does not reinvent the hard parts. Every component is a proven open-source project; cachos picks the best one for each job, wires them together correctly, and fixes the gaps it measured between them (see [What cachos adds](#what-cachos-adds)).

| Job | Built on |
|---|---|
| Speech recognition | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper large-v3, CTranslate2) |
| Real-time runtime, transports, interruptions | [Pipecat](https://github.com/pipecat-ai/pipecat) (BSD-2) |
| Voice activity detection | [Silero VAD](https://github.com/snakers4/silero-vad) (via Pipecat, ONNX) |
| End-of-turn detection | [Smart Turn v3.2](https://github.com/pipecat-ai/smart-turn) (local, 23 languages) |
| Noise suppression | [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) / DPDFNet via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) |
| Speech synthesis | [Kokoro](https://github.com/thewh1teagle/kokoro-onnx), [Piper](https://github.com/OHF-Voice/piper1-gpl), [edge-tts](https://github.com/rany2/edge-tts), [Chatterbox](https://github.com/resemble-ai/chatterbox), [XTTS](https://github.com/idiap/coqui-ai-TTS), [MMS](https://huggingface.co/facebook/mms-tts) |
| Language codes / detection / sentences | [langcodes](https://github.com/georgkrause/langcodes), [lingua](https://github.com/pemistahl/lingua-py), [blingfire](https://github.com/microsoft/BlingFire) |

The research behind each choice (benchmarks, licences, alternatives considered) is in [docs/RESEARCH.md](docs/RESEARCH.md).

---

## Install

```bash
pip install "cachos[recommended]"          # Whisper + Kokoro/Piper/Edge + agent runtime + REST server
pip install "cachos[recommended,cuda]"     # + NVIDIA GPU (CUDA 12 libraries from pip; no system CUDA needed)
pip install "cachos[recommended,mic]"      # + local mic/speakers for `cachos chat` (needs portaudio19-dev)
```

| Extra | Adds |
|---|---|
| `faster-whisper` | local speech recognition |
| `kokoro`, `piper`, `edge` | the three default TTS engines |
| `kokoro-cjk` | Kokoro's Japanese/Chinese voices (misaki G2P) |
| `chatterbox`, `xtts`, `mms` | voice cloning / 1,100-language TTS (PyTorch) |
| `agent` | Pipecat runtime, Silero, Smart Turn, GTCRN denoiser |
| `server` | REST API (`cachos serve`) |
| `cuda` | cuBLAS + cuDNN for GPU Whisper |
| `mic` | local audio transport |

Python ≥ 3.10. Models download on first use into `~/.cache/cachos` and `~/.cache/huggingface` (`CACHOS_CACHE` / `HF_HOME` to move them).

---

## Five ways to use it

### 1. Python library — speech-to-text and text-to-speech

```python
from cachos import STT, TTS

stt = STT()                                      # Whisper large-v3, language auto-detected
t = stt.transcribe_sync("customer_call.mp3")
print(t.language, t.language_probability, t.text)
for s in t.segments:
    print(s.start, s.end, s.text)

# Robustness knobs for real deployments
stt = STT(language_candidates=["en", "hi", "bn"],   # restrict language ID to what callers speak
          hotwords="Cachos, Dr. Okafor, Kubernetes")  # bias towards your names / jargon
stt = STT("fast")                                   # large-v3-turbo (escalates unsure audio to large-v3)
stt = STT("groq", fallback=["openai"])              # hosted, with a fallback chain

tts = TTS()                                      # Kokoro → Piper → Edge, first with a voice for the language
tts.synthesize_sync("नमस्ते, आपका स्वागत है।").save("hi.wav")        # Hindi detected → Hindi voice
tts.synthesize_sync("Hello!", voice="kokoro:af_heart").save("en.wav")
tts.voices(language="es", gender="male")         # browse the catalog
tts = TTS(voices={"es": "edge:es-MX-JorgeNeural"}, gender="female")   # per-language overrides

async for chunk in tts.stream(llm_token_stream):  # LLM tokens → audio, sentence by sentence
    play(chunk)
```

### 2. A complete voice agent

```python
from cachos import VoiceAgent

agent = VoiceAgent(
    llm_model="gpt-4.1-mini",                    # any OpenAI-compatible endpoint:
    # llm_base_url="http://localhost:11434/v1",  #   Ollama, vLLM, Groq, LM Studio, OpenRouter…
    system_prompt="You are the front desk of Hotel Lumen.",
    greeting="Welcome the guest.",
)
agent.run_local()                                # this machine's mic + speakers (use headphones)
agent.serve()                                    # browser client at http://localhost:7860/client
agent.serve(transport="twilio")                  # phone calls (also telnyx, plivo, exotel)
```

The same from the shell: `cachos chat`, `cachos agent -t webrtc`, `cachos agent -t twilio` (see `cachos agent --help`).

Agent options that matter for quality:

| Option | Default | What it does |
|---|---|---|
| `denoise` | `"gtcrn"` | denoiser for VAD/turn detection (`dpdfnet2`/`4`/`8` stronger, `None` off) |
| `denoise_stt` | `False` | also denoise what Whisper hears — usually *hurts* accuracy; A/B-test first |
| `smart_turn` | `True` | Smart Turn v3.2 decides when the user finished (vs. silence only) |
| `interrupt_min_words` | `None` | words needed to interrupt the agent — set 2–3 in noisy places |
| `pause_on_overlap` | on with `interrupt_min_words` | a cough or "mm-hm" *pauses* the agent; it resumes after `resume_after_secs` (2 s) if no real words follow |
| `filter_incomplete_turns` | `False` | the LLM waits when the user is mid-thought |
| `language` / `voice` | auto | pin the spoken language / reply voice |

### 3. Inside an existing Pipecat project

Keep your transport, LLM and tools; swap in the cachos pieces:

```python
from cachos.pipecat import CachosSTTService, CachosTTSService, DenoiseForVAD, LanguageTracker

tracker = LanguageTracker()                      # STT tells TTS which language the user spoke
Pipeline([
    transport.input(),
    CachosSTTService(tracker=tracker),           # hears the original audio
    DenoiseForVAD(),                             # VAD + Smart Turn hear clean audio
    user_aggregator, llm,
    CachosTTSService(tracker=tracker),           # replies in the user's language
    transport.output(), assistant_aggregator,
])
```

Full example: [examples/06_existing_pipecat_pipeline.py](examples/06_existing_pipecat_pipeline.py).

### 4. REST API from any language (OpenAI-compatible)

```bash
cachos serve --port 8000 --api-key SECRET        # docs at http://localhost:8000/docs
```

Existing OpenAI SDKs work by changing the base URL:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="SECRET")
client.audio.speech.create(model="auto", voice="nova", input="Hola, ¿qué tal?").write_to_file("hola.mp3")
client.audio.transcriptions.create(model="auto", file=open("hola.mp3", "rb"))
```

| Endpoint | |
|---|---|
| `POST /v1/audio/transcriptions` | `json`, `text`, `verbose_json` (language + segments), `srt`, `vtt` |
| `POST /v1/audio/speech` | `voice` = OpenAI name (alloy, nova…) or any cachos voice id; extras: `language`, `gender`, `stream` |
| `GET /v1/voices?language=&gender=&engine=` | the merged voice catalog |
| `GET /v1/languages` | languages with at least one voice |

**TypeScript** ([clients/ts](clients/ts)):

```ts
import { CachosClient } from "@cachos/client";                 // REST, browser + Node 18+
const cachos = new CachosClient({ baseUrl: "http://localhost:8000", apiKey: "SECRET" });
const { text, language } = await cachos.transcribe(file);
const mp3 = await cachos.speak("Bonjour !");

import { connectAgent } from "@cachos/client/agent";           // real-time agent, browser (WebRTC)
const agent = await connectAgent({ baseUrl: "http://localhost:7860", onUserTranscript: (t) => console.log(t.text) });
```

`@cachos/client/agent` uses Pipecat's official client SDK (`@pipecat-ai/client-js`, `@pipecat-ai/small-webrtc-transport` as peer dependencies).

### 5. Your own engines (plugins)

Any `STTEngine` / `TTSEngine` subclass joins the voice catalog and language routing:

```toml
# your package's pyproject.toml
[project.entry-points."cachos.engines"]
mytts = "my_pkg.tts:MyTTS"
```

```python
TTS(["mytts", "kokoro", "edge"])                 # yours first, built-ins as fallback
```

Implement `list_voices()` and `_synthesize(text, voice, language, speed)` (TTS) or `_transcribe(audio, language)` (STT).

---

## Docker

```bash
docker build -t cachos .                                            # CPU
docker build -t cachos:gpu --build-arg EXTRAS=recommended,cuda .    # NVIDIA GPU

docker run -p 8000:8000 -v cachos-models:/cache cachos              # REST API
docker run -p 7860:7860 -e OPENAI_API_KEY=… cachos agent --host 0.0.0.0     # voice agent
docker run --gpus all -p 8000:8000 -v cachos-models:/cache cachos:gpu
```

On slow networks: `--build-arg DEBIAN_MIRROR=https://mirrors.aliyun.com/debian --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple` (or any closer mirror).

---

## How the agent works

```
caller audio ──► Whisper (original audio: best accuracy, detects language)
             ──► GTCRN denoiser ──► Silero VAD + Smart Turn v3.2 ("did they finish?")
             ──► LLM (sees the transcript + history)
             ──► TTS: voice chosen for the reply's language, spoken sentence by sentence
             ──► caller   (talking over the agent interrupts it)
```

Denoised audio goes to voice/turn detection only: published studies show front-end denoising
*raises* Whisper's error rate (e.g. 8.8 % → 25.8 % WER at 10 dB SNR, arXiv 2512.17562), while
clean audio makes VAD and turn detection far less trigger-happy.

## What cachos adds

Gaps found by running the components together, and fixed:

| Problem (measured) | cachos fix |
|---|---|
| Pipecat's Whisper service drops the detected language | language flows from STT to TTS; the reply voice matches what the user spoke |
| Short sentences ("Sure.") are too short to detect a reply's language | fall back to the language the user spoke |
| Pipecat ends a turn after a fixed 1 s / 5 s wait — on slower hardware turns ended *empty* while Whisper was still working | the agent measures its real speech-end → transcript latency at start-up and keeps adapting; turn detection waits for it |
| A mid-sentence pause splits one turn into fragments that Whisper transcribes without context ("reservieren" → "rosavieren.") | queued fragments are merged into one clip; later fragments get the turn's earlier text as context and its language |
| large-v3-turbo heard Bengali as Gujarati / Hindi | large-v3 is the default; with turbo, low-confidence detections are re-run on large-v3 |
| Kokoro / Piper Japanese, Chinese and Thai produce gibberish or crash without extra phonemizers | those voices are only offered when their front end is installed; otherwise routed to an engine that speaks them |
| A TTS engine failing (network down, empty audio) kills the reply | the next engine's voice for that language takes over |
| Pipecat has no false-interruption recovery: any interruption discards the rest of the reply | LiveKit-style pause on overlap and resume (`PauseOnOverlap`); audio is released just ahead of real time so the pause takes effect within ~0.25 s |
| Pipecat's "words before interrupting" counts spaces: a whole Chinese/Japanese/Thai sentence is one "word" | multilingual word count |
| `device="auto"` crashed on GPUs without CUDA 12 libraries | probes CUDA, loads pip-installed CUDA libs, falls back to CPU with a warning |

## Measured results

Our own checks — a small set, not a benchmark; numbers depend on hardware. Speech was
synthesized by Edge TTS (a different engine from the agent's voices), fed in real time.

**Speech recognition, 6 languages (en, es, hi, bn, de, ja), NVIDIA GTX 1650 (4 GB, no tensor cores):**

| Model | Time / utterance | Language detected | Mean error* | Notes |
|---|---|---|---|---|
| **large-v3** (default) | 5.2 s | **6/6** | **7 %** | errors are only number formatting ("7.30" for "seven thirty") |
| large-v3-turbo (`"fast"`) | 3.9 s | 5/6 | 20 % | Bengali → Gujarati |
| medium | 3.0 s | 6/6 | 32 % | Bengali garbled |
| small | 1.1 s | 6/6 | 27 % | Hindi/Bengali errors |

\* WER, CER for Japanese. On CPU (no GPU) large-v3-turbo took 12–15 s per utterance: fine for files, too slow for live conversation — use a GPU or `STT("groq")`.

**Full voice loop** (caller speech → Whisper → Smart Turn → LLM → TTS → reply audio), 7 languages:
heard correctly in en, es, hi, de, ja, fr (Bengali with large-v3 only); every reply was spoken
in the caller's language by a native voice (Kokoro en/es/hi/fr, Piper de, Edge ja) and
re-transcribed correctly by Whisper. First reply audio ≈ 6.6–8.6 s on the GTX 1650 with the
Claude CLI as LLM — dominated by Whisper (~3.7 s) and LLM start-up (~2–4 s). With a modern GPU
or hosted Whisper and a direct LLM API, expect ~1–1.5 s.

**Noise:** GTCRN attenuated pure white noise by ~40 dB (RMS 0.100 → 0.001) at 87 ms CPU per second
of audio (single thread). Speech-in-noise WER with the denoiser on vs. off is the next measurement.

## Engines and licences

| Engine | Languages | Voices | Runs on | Licence |
|---|---|---|---|---|
| Whisper (faster-whisper) | 99 | — | GPU / CPU | MIT |
| Kokoro (`kokoro`, default #1) | en, es, fr, hi, it, pt (+ ja, zh with `kokoro-cjk`) | 41 (+ CJK) | CPU real-time | Apache-2.0 |
| Piper (`piper`, default #2) | 50 | 172 | CPU, Raspberry Pi | **GPL-3.0** (code); voices vary |
| Edge (`edge`, default #3) | 76 | 322 neural | network | unofficial Microsoft endpoint, no SLA |
| Chatterbox (`chatterbox`) | 23, voice cloning | — | GPU | MIT · *not yet verified in our test environment* |
| XTTS (`xtts`) | 17, voice cloning | 58 | GPU | ⚠️ CPML, non-commercial · *not yet verified* |
| MMS (`mms`) | 1,100+ | 1 per language | CPU | ⚠️ CC-BY-NC · *not yet verified* |
| Pipecat, Smart Turn, Silero, GTCRN, sherpa-onnx | — | — | CPU | BSD-2 / BSD-2 / MIT / MIT / Apache-2.0 |

**Commercial use:** the default stack is permissive except Piper (GPL) and Edge (terms of an
unofficial endpoint). For a fully permissive offline product use `TTS(["kokoro"])` plus a
permissive engine of your own via the plugin interface.

## Testing without an API key (Claude Code CLI)

If you use Claude Code, [examples/claude_cli_llm.py](examples/claude_cli_llm.py) exposes your logged-in
`claude` CLI as an OpenAI-compatible endpoint for local testing:

```bash
python examples/claude_cli_llm.py --port 8765 &
cachos chat --llm-base-url http://127.0.0.1:8765/v1 --llm-model haiku
```

## Development

```bash
uv venv && uv pip install -e ".[recommended,dev]"
pytest                       # unit tests (fast, no models)
pytest -m integration        # needs real models / network
```

Pipecat is pinned to `1.15.x`: `cachos.pipecat` extends two of its internal hooks
(segment queueing and the TTS audio helper).

Licence: Apache-2.0.
