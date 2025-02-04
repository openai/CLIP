import logging
import time
import numpy as np
import habana_frameworks.torch.core as ht
import torch
from PIL import Image

import clip
from clip.utils import get_device_initial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_model(model_name, device):
    model, transform = clip.load(model_name, device=get_device_initial(device), jit=False)

    image = transform(Image.open("CLIP.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return probs


if __name__ == "__main__":
    logger.info("Running on HPU")
    start_time = time.time()
    run_model("RN50", "hpu")
    end_time = time.time()
    logger.info(f"HPU execution time: {end_time - start_time:.4f} seconds")

    logger.info("Running on CPU")
    start_time = time.time()
    run_model("RN50", "cpu")
    end_time = time.time()
    logger.info(f"CPU execution time: {end_time - start_time:.4f} seconds")
