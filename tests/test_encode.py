import clip
import torch
device = 'cpu'
model, preprocess = clip.load("ViT-B/32", device=device)
