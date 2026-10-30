/**
 * @cachos/client — talk to a `cachos serve` instance from the browser or Node 18+.
 *
 *   const cachos = new CachosClient({ baseUrl: "http://localhost:8000" });
 *   const { text, language } = await cachos.transcribe(file);
 *   const mp3 = await cachos.speak("Bonjour !");            // voice auto-picked for French
 *
 *
 * The server speaks the OpenAI audio API, so the official `openai` SDK works too.
 * For real-time voice agents in the browser see `@cachos/client/agent`.
 */

export interface Voice {
  id: string;
  name: string;
  engine: string;
  languages: string[];
  gender: "female" | "male" | "neutral" | null;
  description: string;
  sample_rate: number | null;
  quality: string | null;
  cloned: boolean;
  tags: string[];
}

export interface Transcription {
  text: string;
  language: string | null;
  duration?: number;
  language_probability?: number;
  segments?: { text: string; start: number; end: number }[];
}

export interface SpeakOptions {
  /** e.g. "kokoro:af_heart", "edge:hi-IN-SwaraNeural"; omit to auto-pick by language */
  voice?: string;
  language?: string;
  gender?: "female" | "male";
  speed?: number;
  format?: "mp3" | "wav" | "opus" | "flac" | "aac" | "pcm";
}

export interface ClientOptions {
  baseUrl?: string;
  apiKey?: string;
  fetch?: typeof fetch;
}

export class CachosClient {
  readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly fetchImpl: typeof fetch;

  constructor(opts: ClientOptions = {}) {
    this.baseUrl = (opts.baseUrl ?? "http://localhost:8000").replace(/\/$/, "");
    this.apiKey = opts.apiKey;
    this.fetchImpl = opts.fetch ?? globalThis.fetch.bind(globalThis);
  }

  private headers(extra: Record<string, string> = {}): Record<string, string> {
    return this.apiKey ? { Authorization: `Bearer ${this.apiKey}`, ...extra } : extra;
  }

  private async request(path: string, init: RequestInit = {}): Promise<Response> {
    const res = await this.fetchImpl(this.baseUrl + path, {
      ...init,
      headers: this.headers((init.headers as Record<string, string>) ?? {}),
    });
    if (!res.ok) throw new Error(`cachos ${path} failed (${res.status}): ${await res.text()}`);
    return res;
  }

  /** Speech file -> text. Language is auto-detected unless given. */
  async transcribe(
    audio: Blob | ArrayBuffer | Uint8Array,
    opts: { language?: string; prompt?: string; verbose?: boolean } = {},
  ): Promise<Transcription> {
    const form = new FormData();
    const blob = audio instanceof Blob ? audio : new Blob([audio as BlobPart]);
    form.append("file", blob, "audio");
    if (opts.language) form.append("language", opts.language);
    if (opts.prompt) form.append("prompt", opts.prompt);
    form.append("response_format", opts.verbose ? "verbose_json" : "json");
    return (await this.request("/v1/audio/transcriptions", { method: "POST", body: form })).json();
  }

  /** Text -> encoded audio (mp3 by default). */
  async speak(text: string, opts: SpeakOptions = {}): Promise<ArrayBuffer> {
    const res = await this.request("/v1/audio/speech", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(this.speechBody(text, opts, false)),
    });
    return res.arrayBuffer();
  }

  /** Text -> streamed audio chunks (sentence by sentence). Use format "pcm" for raw s16le at 24 kHz. */
  async *speakStream(text: string, opts: SpeakOptions = {}): AsyncGenerator<Uint8Array> {
    const res = await this.request("/v1/audio/speech", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(this.speechBody(text, { format: "pcm", ...opts }, true)),
    });
    const reader = res.body!.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return;
      if (value) yield value;
    }
  }

  private speechBody(text: string, o: SpeakOptions, stream: boolean) {
    return {
      input: text, voice: o.voice, language: o.language, gender: o.gender,
      speed: o.speed ?? 1, response_format: o.format ?? "mp3", stream,
    };
  }

  async voices(filter: { language?: string; gender?: string; engine?: string } = {}): Promise<Voice[]> {
    const q = new URLSearchParams(Object.entries(filter).filter(([, v]) => v) as [string, string][]);
    return (await (await this.request(`/v1/voices?${q}`)).json()).data;
  }

  async languages(): Promise<{ code: string; name: string }[]> {
    return (await (await this.request("/v1/languages")).json()).tts;
  }
}
