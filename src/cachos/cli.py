"""``cachos`` command line."""

from __future__ import annotations

import argparse
import json
import sys


def _tts(args):
    from .core import TTS

    engines = args.engines.split(",") if getattr(args, "engines", None) else "auto"
    return TTS(engines, allow_cloud=not getattr(args, "offline", False))


def _stt(args):
    from .core import STT

    opts = {}
    if getattr(args, "device", None):
        opts["device"] = args.device
    return STT(args.stt, **opts)


def _agent(args):
    from .agent import DEFAULT_SYSTEM_PROMPT, VoiceAgent

    return VoiceAgent(
        llm_model=args.llm_model, llm_base_url=args.llm_base_url, llm_api_key=args.llm_api_key,
        system_prompt=args.system_prompt or DEFAULT_SYSTEM_PROMPT, greeting=args.greeting,
        stt=_stt(args), tts=_tts(args), language=args.language, voice=args.voice,
        denoise=None if args.denoise == "off" else args.denoise, denoise_stt=args.denoise_stt,
        smart_turn=not args.no_smart_turn, interrupt_min_words=args.interrupt_min_words,
        filter_incomplete_turns=args.filter_incomplete_turns,
        pause_on_overlap=False if args.no_pause_on_overlap else None,
    )


def cmd_transcribe(args) -> None:
    from .server import _subtitles

    t = _stt(args).transcribe_sync(args.file, args.language)
    if args.format == "json":
        print(json.dumps(t.to_dict(), ensure_ascii=False, indent=2))
    elif args.format in ("srt", "vtt"):
        print(_subtitles(t, args.format))
    else:
        print(f"[{t.language} {t.language_probability or 0:.2f}] {t.text}" if args.verbose else t.text)


def cmd_speak(args) -> None:
    tts = _tts(args)
    text = args.text if args.text != "-" else sys.stdin.read()
    audio = tts.synthesize_sync(text, args.voice, args.language, gender=args.gender, speed=args.speed)
    if args.play:
        import sounddevice as sd

        sd.play(audio.samples, audio.sample_rate)
        sd.wait()
    out = args.output or ("speech.wav" if not args.play else None)
    if out:
        audio.save(out)
        print(f"wrote {out} ({audio.duration:.1f}s @ {audio.sample_rate} Hz)", file=sys.stderr)


def cmd_voices(args) -> None:
    vs = _tts(args).voices(args.language, args.gender, args.engine)
    if args.json:
        print(json.dumps([v.to_dict() for v in vs], ensure_ascii=False, indent=2))
        return
    for v in vs:
        langs = ",".join(v.languages[:6]) + ("…" if len(v.languages) > 6 else "")
        print(f"{v.id:<48} {v.gender or '-':<7} {langs:<22} {v.description[:50]}")
    print(f"\n{len(vs)} voices", file=sys.stderr)


def cmd_languages(args) -> None:
    from .languages import language_name

    tts = _tts(args)
    for code in tts.languages:
        n = len(tts.voices(code))
        print(f"{code:<6} {language_name(code):<22} {n} voices")


def cmd_detect(args) -> None:
    from .languages import detect_language

    print(detect_language(args.text))


def cmd_serve(args) -> None:
    import uvicorn

    from .server import create_app

    app = create_app(_stt(args), _tts(args), api_key=args.api_key)
    print(f"cachos REST API on http://{args.host}:{args.port}  (docs at /docs)", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_agent(args) -> None:
    transport = None if args.transport == "all" else args.transport
    _agent(args).serve(host=args.host, port=args.port, transport=transport)


def cmd_chat(args) -> None:
    try:
        _agent(args).run_local()
    except KeyboardInterrupt:
        pass


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cachos", description="Multilingual STT/TTS and voice agents")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_stt(sp):
        sp.add_argument("--stt", default="auto", help="STT engine: auto (=faster-whisper:large-v3), fast (=large-v3-turbo), groq, openai, ...")
        sp.add_argument("--device", help="cpu | cuda | auto")

    def add_tts(sp):
        sp.add_argument("--engines", help="comma-separated TTS engines in priority order (default: kokoro,piper,edge)")
        sp.add_argument("--offline", action="store_true", help="exclude cloud engines (edge, openai)")

    def add_agent(sp):
        sp.add_argument("--llm-model", default="gpt-4.1-mini", help="model on an OpenAI-compatible endpoint")
        sp.add_argument("--llm-base-url", help="e.g. http://localhost:11434/v1 for Ollama")
        sp.add_argument("--llm-api-key", help="default: $OPENAI_API_KEY")
        sp.add_argument("--system-prompt")
        sp.add_argument("--greeting", help="instruction for the agent's first message, e.g. 'Greet the user.'")
        sp.add_argument("-l", "--language", help="fix the spoken language (default: detect per utterance)")
        sp.add_argument("--voice", help="fix the reply voice (default: per reply language)")
        sp.add_argument("--denoise", default="gtcrn",
                        help="gtcrn | dpdfnet_baseline | dpdfnet2 | dpdfnet4 | dpdfnet8 | off")
        sp.add_argument("--denoise-stt", action="store_true", help="also denoise STT input (A/B-test WER first)")
        sp.add_argument("--no-smart-turn", action="store_true", help="end turns on VAD silence only")
        sp.add_argument("--interrupt-min-words", type=int, help="words needed to interrupt the bot")
        sp.add_argument("--filter-incomplete-turns", action="store_true",
                        help="let the LLM wait when the user is mid-thought")
        sp.add_argument("--no-pause-on-overlap", action="store_true",
                        help="with --interrupt-min-words: cut off instead of pausing on a cough")
        add_stt(sp)
        add_tts(sp)

    sp = sub.add_parser("transcribe", help="speech file -> text (language auto-detected)")
    sp.add_argument("file")
    sp.add_argument("-l", "--language")
    sp.add_argument("-f", "--format", choices=["text", "json", "srt", "vtt"], default="text")
    sp.add_argument("-v", "--verbose", action="store_true")
    add_stt(sp)
    sp.set_defaults(fn=cmd_transcribe)

    sp = sub.add_parser("speak", help="text -> speech (voice auto-picked per language)")
    sp.add_argument("text", help="text, or - for stdin")
    sp.add_argument("--voice", help="e.g. kokoro:af_heart, edge:fr-FR-DeniseNeural")
    sp.add_argument("-l", "--language")
    sp.add_argument("-g", "--gender", choices=["female", "male"])
    sp.add_argument("--speed", type=float, default=1.0)
    sp.add_argument("-o", "--output", help="output file (.wav/.mp3/.ogg/.flac)")
    sp.add_argument("--play", action="store_true")
    add_tts(sp)
    sp.set_defaults(fn=cmd_speak)

    sp = sub.add_parser("voices", help="list voices")
    sp.add_argument("-l", "--language")
    sp.add_argument("-g", "--gender", choices=["female", "male"])
    sp.add_argument("-e", "--engine")
    sp.add_argument("--json", action="store_true")
    add_tts(sp)
    sp.set_defaults(fn=cmd_voices)

    sp = sub.add_parser("languages", help="list TTS languages")
    add_tts(sp)
    sp.set_defaults(fn=cmd_languages)

    sp = sub.add_parser("detect", help="detect the language of a text")
    sp.add_argument("text")
    sp.set_defaults(fn=cmd_detect)

    sp = sub.add_parser("serve", help="OpenAI-compatible STT/TTS REST API")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--api-key", help="require this bearer token")
    add_stt(sp)
    add_tts(sp)
    sp.set_defaults(fn=cmd_serve)

    sp = sub.add_parser("agent", help="serve a voice agent: browser WebRTC client at /client, or telephony")
    sp.add_argument("--host", default="localhost")
    sp.add_argument("--port", type=int, default=7860)
    sp.add_argument("-t", "--transport", default="webrtc",
                    choices=["webrtc", "twilio", "telnyx", "plivo", "exotel", "websocket", "all"])
    add_agent(sp)
    sp.set_defaults(fn=cmd_agent)

    sp = sub.add_parser("chat", help="talk to a voice agent with this machine's mic and speakers")
    add_agent(sp)
    sp.set_defaults(fn=cmd_chat)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
