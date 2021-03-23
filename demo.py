import os
import clip
import torch
from torchvision.datasets import CIFAR100
import gradio as gr

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

def classify(img):
    image = transform(img).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    text=""
    # Print the result
    for value, index in zip(values, indices):
        text+=f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%\n"
    return text

inputs = gr.inputs.Image(type='pil', label="Original Image")
outputs = gr.outputs.Textbox(type="str", label="Text Output")

title = "CLIP"
description = "CLIP demo"

gr.Interface(classify, inputs, outputs, title=title, description=description).launch(debug=True)