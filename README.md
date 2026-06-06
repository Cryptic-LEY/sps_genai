# SPS GenAI - Assignment 1

FastAPI application with Bigram text generation and Spacy word embedding.

## Endpoints

- `GET /` — Health check
- `GET /embed?word={word}` — Returns word embedding vector using Spacy
- `POST /generate` — Generates text using a Bigram model

## Run with Docker

```bash
docker build -t sps_genai .
docker run -p 8000:8000 sps_genai
```

Then visit: http://localhost:8000/docs

## Run Locally

```bash
uv sync
uv run fastapi dev main.py
```
