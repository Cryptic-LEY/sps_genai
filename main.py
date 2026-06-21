from typing import Union
import io

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from app.bigram_model import BigramModel
from helper_lib.data_loader import CLASSES
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