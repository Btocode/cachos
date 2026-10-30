# cachos server image.
#
#   docker build -t cachos .                                   # CPU
#   docker build -t cachos:gpu --build-arg EXTRAS=recommended,cuda .
#
#   docker run -p 8000:8000 cachos                             # STT/TTS REST API (OpenAI-compatible)
#   docker run -p 7860:7860 -e OPENAI_API_KEY=... cachos agent --host 0.0.0.0   # voice agent, browser at /client
#   docker run --gpus all -p 8000:8000 cachos:gpu              # NVIDIA GPU (CUDA libs come from pip)
#
# Models download on first use into /cache; mount a volume to keep them:
#   docker run -v cachos-models:/cache -p 8000:8000 cachos
FROM python:3.11-slim

# ffmpeg: decoding uploads / mp3+opus output; libsndfile: soundfile
# On slow networks pass a closer mirror, e.g. --build-arg DEBIAN_MIRROR=https://mirrors.aliyun.com/debian
ARG DEBIAN_MIRROR=""
RUN if [ -n "$DEBIAN_MIRROR" ]; then \
        sed -i "s|http://deb.debian.org/debian|$DEBIAN_MIRROR|g" /etc/apt/sources.list.d/debian.sources; fi \
    && apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ARG EXTRAS=recommended
ARG PIP_INDEX_URL=https://pypi.org/simple
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" ".[${EXTRAS}]"

ENV CACHOS_CACHE=/cache \
    HF_HOME=/cache/huggingface \
    PYTHONUNBUFFERED=1
VOLUME ["/cache"]
EXPOSE 8000 7860

ENTRYPOINT ["cachos"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
