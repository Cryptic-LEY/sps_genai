from typing import Union
import io

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from app.bigram_model import BigramModel
from helper_lib.data_loader import CLASSES
from helper_lib.diffusion import offset_cosine_diffusion_schedule
from helper_lib.generator import generate_diffusion_grid_png, generate_ebm_grid_png, generate_image_grid_png
from helper_lib.model import get_model
from helper_lib.utils import get_device
import spacy

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)
nlp = spacy.load("en_core_web_md")

device = get_device()
cnn_model = get_model("CNN")
cnn_model.load_state_dict(
    torch.load("checkpoints/best/model.pth", map_location=device)["model_state_dict"]
)
cnn_model.to(device)
cnn_model.eval()

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

GAN_Z_DIM = 100
gan_generator, _ = get_model("GAN", z_dim=GAN_Z_DIM)
gan_generator.load_state_dict(
    torch.load("checkpoints_gan/best/generator.pth", map_location=device)["model_state_dict"]
)
gan_generator.to(device)
gan_generator.eval()

EBM_IMAGE_SIZE = 32
ebm_model = get_model("EBM")
ebm_model.load_state_dict(
    torch.load("checkpoints_ebm/best/model.pth", map_location=device)["model_state_dict"]
)
ebm_model.to(device)
ebm_model.eval()

DIFFUSION_IMAGE_SIZE = 64
diffusion_model = get_model("Diffusion")
diffusion_model.load_state_dict(
    torch.load("checkpoints_diffusion/best/unet_ema.pth", map_location=device)["model_state_dict"]
)
diffusion_model.to(device)
diffusion_model.eval()


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/embed")
def get_embedding(word: str):
    token = nlp(word)
    return {
        "word": word,
        "vector": token.vector.tolist()
    }

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = cnn_model(input_tensor)
        predicted_index = torch.argmax(output, dim=1).item()

    return {"predicted_class": CLASSES[predicted_index]}


@app.get("/generate-image")
def generate_image(num_samples: int = 16):
    buf = generate_image_grid_png(gan_generator, device, num_samples=num_samples, z_dim=GAN_Z_DIM)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/generate-ebm")
def generate_ebm(num_samples: int = 16, steps: int = 256):
    buf = generate_ebm_grid_png(
        ebm_model, device, num_samples=num_samples, image_size=EBM_IMAGE_SIZE, steps=steps
    )
    return StreamingResponse(buf, media_type="image/png")


@app.get("/generate-diffusion")
def generate_diffusion(num_samples: int = 16, diffusion_steps: int = 20):
    buf = generate_diffusion_grid_png(
        diffusion_model,
        offset_cosine_diffusion_schedule,
        device,
        num_samples=num_samples,
        image_size=DIFFUSION_IMAGE_SIZE,
        diffusion_steps=diffusion_steps,
    )
    return StreamingResponse(buf, media_type="image/png")
