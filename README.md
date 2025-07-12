
# Ollama Offline Agent

This project provides a containerized, offline-capable LLM API using [Ollama](https://ollama.com). It automatically pulls and serves a chosen model on first run and exposes a simple REST API for interaction. Designed to be portable and ready for homelab or production environments.

## Features

- Fully offline after initial model pull
- Configurable model selection (`llama3`, `mistral`, `phi3`, etc.)
- Dockerized for consistent deployment
- Health checks and persistent model storage
- Simple REST API powered by FastAPI

## Requirements

- Docker
- Docker Compose

## Getting Started

### Clone the repository
```bash
git clone https://github.com/yourusername/ollama-offline-agent.git
cd ollama-offline-agent
```

### Build and start the container
```bash
docker compose up --build
```

On first run, the container will:
1. Start the Ollama daemon.
2. Pull the specified model if it is not already cached.
3. Launch the API server on port 8000.

Subsequent runs will skip the model pull if the model is already present.

## Configuration

### Change the default model
The default model is `llama3`. To change it, edit `docker-compose.yml`:

```yaml
environment:
  - OLLAMA_MODEL=mistral
```

Rebuild the container to apply changes:
```bash
docker compose up --build
```

## API Endpoints

### Health Check
```http
GET /
```
Response:
```json
{
  "message": "Ollama Offline Agent is running."
}
```

### Ask a Question
```http
POST /ask
Content-Type: application/json
```
Body:
```json
{
  "prompt": "What is the capital of France?"
}
```
Response:
```json
{
  "prompt": "What is the capital of France?",
  "response": "The capital of France is Paris."
}
```

## Persistent Model Storage

The Docker volume `models` ensures that downloaded models persist across container rebuilds.

## Healthcheck

The container includes a Docker healthcheck that monitors the API’s availability.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
