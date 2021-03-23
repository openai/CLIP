import os
import clip
import torch
from torchvision.datasets import CIFAR100
import gradio as gr

# Load the model
device = "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

def classify(img, user_text):
    image = preprocess(img).unsqueeze(0).to(device)
    user_texts = user_text.split(",")
    text_sources = cifar100.classes + user_texts
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in text_sources]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    result = {}
    for value, index in zip(values, indices):
        result[text_sources[index]] = value.item()
    return result

inputs = [
  gr.inputs.Image(type='pil', label="Original Image"),
  gr.inputs.Textbox(lines=1, label="optional categories")
]
outputs = gr.outputs.Label(type="confidences",num_top_classes=5)

title = "CLIP"
description = "demo for OpenAI's CLIP. To use it, simply upload your image, or click one of the examples to load them and optionally add a text label seperated by commas to help clip classify the image better. Read more at the links below."
article = "<p style='text-align: center'><a href='https://openai.com/blog/clip/'>CLIP: Connecting Text and Images</a> | <a href='https://github.com/openai/CLIP'>Github Repo</a></p>"

examples = [
    ["zebra.jpg"],
    ["giraffe.jpg"]
]

gr.Interface(classify, inputs, outputs, title=title, description=description, examples=examples, article=article).launch(debug=True)