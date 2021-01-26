import clip
import torch
import torch
from PIL import Image
def test_smoke_simple_cpu():
    device = 'cpu'
    model, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(Image.open('CLIP.png')).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    with torch.no_grad():
        model.encode_image(image)
        model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    assert True

