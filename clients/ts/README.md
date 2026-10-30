# @cachos/client

TypeScript client for [cachos](../../README.md).

```bash
npm install @cachos/client
# real-time agent (browser) also needs:
npm install @pipecat-ai/client-js @pipecat-ai/small-webrtc-transport
```

## REST: speech-to-text / text-to-speech (browser + Node 18+)

Talks to `cachos serve` (OpenAI-compatible, so the official `openai` SDK works too).

```ts
import { CachosClient } from "@cachos/client";

const cachos = new CachosClient({ baseUrl: "http://localhost:8000", apiKey: "SECRET" });

const { text, language } = await cachos.transcribe(fileOrBlob);        // language auto-detected
const mp3 = await cachos.speak("¡Hola! ¿En qué puedo ayudarte?");       // Spanish voice auto-picked
const wav = await cachos.speak("Hello", { voice: "kokoro:af_heart", format: "wav" });
for await (const pcm of cachos.speakStream(longText)) play(pcm);       // s16le @ 24 kHz, sentence by sentence
const voices = await cachos.voices({ language: "hi", gender: "female" });
```

## Real-time voice agent (browser)

Talks to `cachos agent` / `VoiceAgent.serve()` over WebRTC (Pipecat's official client SDK),
so echo cancellation, noise suppression and jitter handling come from the browser.

```ts
import { connectAgent } from "@cachos/client/agent";

const agent = await connectAgent({
  baseUrl: "http://localhost:7860",
  onUserTranscript: (t) => t.final && console.log("you:", t.text),
  onBotText: (text) => console.log("agent:", text),
});
// agent.setMicEnabled(false); await agent.disconnect();
```

Call `connectAgent` from a user gesture (e.g. a button click) so the browser allows microphone
access and audio playback.

## Build

```bash
npm install && npm run build
```
