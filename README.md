# SPS GenAI

FastAPI application with Bigram text generation, Spacy word embedding, and a CNN image classifier (CIFAR10).

## Endpoints

- `GET /` — Health check
- `GET /embed?word={word}` — Returns word embedding vector using Spacy
- `POST /generate` — Generates text using a Bigram model
- `POST /classify` — Classifies an uploaded image into one of 10 CIFAR10 classes using a CNN

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

## Train the CNN Classifier

```bash
uv run python train_cnn.py
```

Trains a CNN on CIFAR10 (resized to 64x64x3) and saves checkpoints to `checkpoints/`,
including the best model at `checkpoints/best/model.pth` used by the `/classify` endpoint.

## Project Structure

- `main.py` — FastAPI application and endpoints
- `helper_lib/` — Reusable ML library (data loading, models, training, evaluation, checkpoints)
- `train_cnn.py` — Training script for the CNN classifier
- `app/bigram_model.py` — Bigram text generation model
