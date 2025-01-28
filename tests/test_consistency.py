import numpy as np
import pytest
import torch
from PIL import Image

import clip


@pytest.mark.parametrize("model_name", clip.available_models())
def test_consistency(model_name):
    device = "cpu"
    jit_model, transform = clip.load(model_name, device=device, jit=True)
    py_model, _ = clip.load(model_name, device=device, jit=False)

    image = transform(Image.open("CLIP.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad():
        logits_per_image, _ = jit_model(image, text)
        jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        logits_per_image, _ = py_model(image, text)
        py_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    assert np.allclose(jit_probs, py_probs, atol=0.01, rtol=0.1)


@pytest.mark.parametrize("model_name", clip.available_models())
def test_hpu_support(model_name):
    devices = ["hpu", "cpu"]
    all_probs = []
    for device in devices:
        print(f"=== Testing {model_name} on {device} ===")
        model, transform = clip.load(model_name, device=device, jit=False)

        image = transform(Image.open("CLIP.png")).unsqueeze(0).to(device)
        text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

        with torch.no_grad():
            logits_per_image, _ = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            all_probs.append(probs)

    assert np.allclose(all_probs[0], all_probs[1], atol=0.01, rtol=0.1)
