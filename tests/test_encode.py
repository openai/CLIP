import clip
import torch
import torch
from PIL import Image
def test_simple_cpu():
    device = 'cpu'
    model, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(Image.open('CLIP.png')).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    with torhc.no_grad():
        assert model.encode_image(image), "Encoding an image does not work"
        assert model.encode_text(text), "Encoding text does not work"
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print("Label probs:", probs)

