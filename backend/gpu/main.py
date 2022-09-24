from CLIP import embed
import clip
import torch
import numpy as np
import cv2
from PIL import Image
import json
from fastapi import FastAPI
from typing import Dict

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

@app.post('/')
async def handle(request: Dict):
    frames = request['video_frames']
    response = embed(frames, model, preprocess, device)
    return response