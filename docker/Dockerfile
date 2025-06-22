FROM python:3.11-slim as builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim

LABEL maintainer="Cyberguard AI <contact@cyberguard-ai.com>"
LABEL description="Local LLM Server - Self-hosted language model inference"
LABEL version="1.0.0"

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=builder /root/.local /root/.local

# Create non-root user
RUN groupadd -r llmuser && useradd -r -g llmuser llmuser

WORKDIR /app

COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

RUN mkdir -p models logs \
    && chown -R llmuser:llmuser /app

RUN chmod +x scripts/*.sh

USER llmuser

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "src.server"]