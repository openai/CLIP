import logging
import time
import numpy as np
import torch
from PIL import Image

import clip
from clip.utils import get_device_initial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_model(model_name, device):
    model, transform = clip.load(
        model_name, device=get_device_initial(device), jit=False
    )

    image = transform(Image.open("CLIP.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad():
        start_time = time.perf_counter()

        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        end_time = time.perf_counter()
        logger.info(f"Execution time: {end_time - start_time:.4f} seconds")
    return probs, end_time - start_time


def run_n_times(model_name, device, n):
    times = []
    logger.info(f"Running {model_name} on {device} {n} times")
    for _ in range(n):
        logger.info(f"Run {_ + 1} of {n}")
        _, time = run_model(model_name, device)
        times.append(time)
    return np.mean(times)


if __name__ == "__main__":
    hpu_time = run_n_times("RN50", "hpu", 10)
    cpu_time = run_n_times("RN50", "cpu", 10)

    logger.info(f"HPU time: {hpu_time:.4f} seconds")
    logger.info(f"CPU time: {cpu_time:.4f} seconds")
