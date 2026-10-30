/**
 * @cachos/client/agent — talk to a cachos voice agent from the browser.
 *
 * The agent (`cachos agent` / `VoiceAgent.serve()`) runs on Pipecat's runner, so the
 * browser side is Pipecat's official client SDK over WebRTC (echo cancellation,
 * noise suppression and jitter handling by the browser). This module only wires it
 * up and plays the agent's voice:
 *
 *   import { connectAgent } from "@cachos/client/agent";
 *   const agent = await connectAgent({
 *     baseUrl: "http://localhost:7860",
 *     onUserTranscript: (t) => console.log("you:", t.text),
 *     onBotText: (t) => console.log("agent:", t),
 *   });
 *   // ... later
 *   await agent.disconnect();
 *
 * Requires the peer dependencies `@pipecat-ai/client-js` and
 * `@pipecat-ai/small-webrtc-transport`.
 */

import { PipecatClient, type RTVIEventCallbacks, type TranscriptData } from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";

export interface AgentOptions {
  /** Where `cachos agent` runs (default http://localhost:7860). */
  baseUrl?: string;
  /** Extra data passed to the bot as `runner_args.body`. */
  body?: Record<string, unknown>;
  /** Final and interim transcripts of what the user said. */
  onUserTranscript?: (t: TranscriptData) => void;
  /** Text of the agent's reply as it is spoken. */
  onBotText?: (text: string) => void;
  onBotStartedSpeaking?: () => void;
  onBotStoppedSpeaking?: () => void;
  onError?: (error: unknown) => void;
  /** Element that plays the agent's voice; one is created if omitted. */
  audioElement?: HTMLAudioElement;
  /** Any other Pipecat client callbacks. */
  callbacks?: RTVIEventCallbacks;
}

export interface AgentSession {
  client: PipecatClient;
  disconnect(): Promise<void>;
  /** Mute or unmute the microphone. */
  setMicEnabled(enabled: boolean): void;
}

export async function connectAgent(opts: AgentOptions = {}): Promise<AgentSession> {
  const baseUrl = (opts.baseUrl ?? "http://localhost:7860").replace(/\/$/, "");
  const audio = opts.audioElement ?? Object.assign(document.createElement("audio"), { autoplay: true });

  const client = new PipecatClient({
    transport: new SmallWebRTCTransport(),
    enableMic: true,
    enableCam: false,
    callbacks: {
      ...opts.callbacks,
      onTrackStarted: (track, participant) => {
        if (!participant?.local && track.kind === "audio") {
          audio.srcObject = new MediaStream([track]);
          void audio.play().catch(() => undefined); // autoplay may need a user gesture
        }
        opts.callbacks?.onTrackStarted?.(track, participant);
      },
      onUserTranscript: (t) => {
        opts.onUserTranscript?.(t);
        opts.callbacks?.onUserTranscript?.(t);
      },
      onBotOutput: (o) => {
        opts.onBotText?.(o.text);
        opts.callbacks?.onBotOutput?.(o);
      },
      onBotStartedSpeaking: () => {
        opts.onBotStartedSpeaking?.();
        opts.callbacks?.onBotStartedSpeaking?.();
      },
      onBotStoppedSpeaking: () => {
        opts.onBotStoppedSpeaking?.();
        opts.callbacks?.onBotStoppedSpeaking?.();
      },
      onError: (m) => {
        opts.onError?.(m);
        opts.callbacks?.onError?.(m);
      },
    },
  });

  type StartRequest = Parameters<PipecatClient["startBotAndConnect"]>[0];
  await client.startBotAndConnect({
    endpoint: `${baseUrl}/start`,
    // the runner's /start body; JSON-serializable by construction
    requestData: { transport: "webrtc", body: opts.body ?? {} } as StartRequest["requestData"],
  });

  return {
    client,
    disconnect: async () => {
      await client.disconnect();
      audio.srcObject = null;
    },
    setMicEnabled: (enabled) => client.enableMic(enabled),
  };
}
