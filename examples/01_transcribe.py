"""Transcribe any audio file; the language is detected automatically."""
import sys

from cachos import STT

stt = STT()  # faster-whisper large-v3 (accuracy first); STT("fast") = large-v3-turbo, STT("groq") = cloud
t = stt.transcribe_sync(sys.argv[1])
print(f"[{t.language} p={t.language_probability:.2f}] {t.text}")
for s in t.segments:
    print(f"  {s.start:6.2f}-{s.end:6.2f}  {s.text}")
