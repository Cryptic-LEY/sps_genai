# SPS GenAI

FastAPI application with Bigram text generation, Spacy word embedding, a CNN image classifier (CIFAR10), a GAN handwritten digit generator (MNIST), an Energy-Based image generator (CIFAR10), a Diffusion image generator (CIFAR10), and an RL-post-trained word-chain generator constrained to a fixed answer format.

## Endpoints

- `GET /` — Health check
- `GET /embed?word={word}` — Returns word embedding vector using Spacy
- `POST /generate` — Generates text using a Bigram model
- `POST /classify` — Classifies an uploaded image into one of 10 CIFAR10 classes using a CNN
- `GET /generate-image?num_samples={n}` — Returns a PNG grid of `n` GAN-generated handwritten digits
- `GET /generate-ebm?num_samples={n}&steps={s}` — Returns a PNG grid of `n` CIFAR10-like images sampled via Langevin dynamics from an Energy-Based Model
- `GET /generate-diffusion?num_samples={n}&diffusion_steps={s}` — Returns a PNG grid of `n` CIFAR10-like images sampled via reverse diffusion from a UNet
- `GET /generate-formatted-text` — Returns a 5-word chain from the RL-post-trained policy, always starting with "group" and ending in a word that ends in "e"

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

## Train the GAN

```bash
uv run python train_gan.py
```

Trains a DCGAN on MNIST and saves checkpoints to `checkpoints_gan/`, including the
generator/discriminator at `checkpoints_gan/best/` used by the `/generate-image` endpoint.

## Train the Energy-Based Model

```bash
uv run python train_ebm.py
```

Trains an EnergyModel on CIFAR10 (32x32x3) using contrastive divergence and a Langevin
sampling replay buffer, saving checkpoints to `checkpoints_ebm/`, including the best
model at `checkpoints_ebm/best/model.pth` used by the `/generate-ebm` endpoint.

## Train the Diffusion Model

```bash
uv run python train_diffusion.py
```

Trains a UNet noise predictor on CIFAR10 (64x64x3) with an offset-cosine diffusion
schedule and an EMA shadow network, saving checkpoints to `checkpoints_diffusion/`,
including the best EMA weights at `checkpoints_diffusion/best/unet_ema.pth` used by
the `/generate-diffusion` endpoint.

## Post-train the word-chain generator with RL

```bash
uv run python train_rl_formatter.py
```

Starting from the Module 10 word-chain base model, uses REINFORCE (policy gradient)
to post-train it so every generated chain follows a fixed format: it always starts
with the word "group" and always ends with a word ending in "e". Saves the trained
policy to `checkpoints_rl/best/policy.pth`, used by the `/generate-formatted-text`
endpoint.

## Project Structure

- `main.py` — FastAPI application and endpoints
- `helper_lib/` — Reusable ML library (data loading, models, training, evaluation, checkpoints, sample generation)
- `helper_lib/diffusion.py` — UNet architecture and diffusion schedules
- `train_cnn.py` — Training script for the CNN classifier
- `train_gan.py` — Training script for the GAN
- `train_ebm.py` — Training script for the Energy-Based Model
- `train_diffusion.py` — Training script for the Diffusion Model
- `helper_lib/rl_lm.py` — Word-chain base model, RL environment, and REINFORCE policy
- `train_rl_formatter.py` — RL post-training script for the word-chain generator
- `app/bigram_model.py` — Bigram text generation model
