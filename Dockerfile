FROM python:3.11-slim-bullseye

RUN apt-get update && apt-get install -y \
    curl git gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/

EXPOSE 8000

# Set default model (can be overridden with ENV)
ENV OLLAMA_MODEL=llama3

# Start daemon, ensure model is pulled, then start API
ENTRYPOINT ["bash", "-c"]
CMD ["\
  echo 'Starting daemon...' && \
  ollama serve & \
  until curl -s http://localhost:11434/api/tags > /dev/null; do \
    echo 'Waiting for daemon...'; sleep 1; \
  done && \
  echo 'Ready... Checking for model: $OLLAMA_MODEL' && \
  ollama list | grep -q $OLLAMA_MODEL || ollama pull $OLLAMA_MODEL && \
  echo 'Starting server...' && \
  uvicorn main:app --host 0.0.0.0 --port 8000 \
"]
