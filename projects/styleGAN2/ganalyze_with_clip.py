import argparse
import json
import subprocess
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as torch_transforms
import ganalyze_transformations as transformations
import ganalyze_common_utils as common
import pickle
import os
import pathlib
sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append('/data/scratch/swamiviv/projects/stylegan2-ada-pytorch/')
from clip_classifier_utils import SimpleTokenizer
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gan_output_transform(imgs):
    # Input:
    # img: NCHW
    #
    # Output
    # img_np: HWC RGB image

    imgs = (imgs * 127.5 + 128).clamp(0, 255).float()
    return imgs


def clip_input_transform(images):
    # Input
    # img_np: torch tensor of shape NHWC, RGB image
    #
    # Output
    # image_input: torch tensor of shape NHWC

    image_mean = (0.48145466, 0.4578275, 0.40821073)
    image_std = (0.26862954, 0.26130258, 0.27577711)

    transform = torch.nn.Sequential(
        torch_transforms.Resize((256, 256)),
        torch_transforms.CenterCrop((224, 224)),
        torch_transforms.Normalize(image_mean, image_std),
    )

    return transform(images)

def get_clip_scores(image_inputs, encoded_text, model, class_index=0):
    #TODO: clarify class index
    image_inputs = clip_input_transform(image_inputs).to(device)
    image_feats = model.encode_image(image_inputs).float()
    image_feats = F.normalize(image_feats, p=2, dim=-1)

    similarity_scores = torch.matmul(image_feats, torch.transpose(encoded_text, 0, 1))
    similarity_scores = similarity_scores.to(device)
    return similarity_scores.narrow(dim=-1, start=class_index, length=1).squeeze(dim=-1)

def get_clip_probs(image_inputs, encoded_text, model, class_index=0):
    image_inputs = clip_input_transform(image_inputs).to(device)
    image_feats = model.encode_image(image_inputs).float()
    image_feats = F.normalize(image_feats, p=2, dim=-1)

    clip_probs = (100.0 * torch.matmul(image_feats, torch.transpose(encoded_text, 0, 1))).softmax(dim=-1)
    clip_probs = clip_probs.to(device)

    return clip_probs.narrow(dim=-1, start=class_index, length=1).squeeze(dim=-1)

# Set up GAN
gan_model_path = '../pretrained/ffhq.pkl'
# Initialize GAN generator and transforms
with open(gan_model_path, 'rb') as f:
    G = pickle.load(f)['G_ema']
G.eval()
G.to(device)
latent_space_dim = G.z_dim

# Set up clip classifier
clip_model_path = '../pretrained/clip_ViT-B-32.pt'
clip_model = torch.jit.load(clip_model_path)
clip_model.eval()
clip_model.to(device)
input_resolution = clip_model.input_resolution.item()
context_length = clip_model.context_length.item()
vocab_size = clip_model.vocab_size.item()

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

# Extract text features for clip
attributes = ["an evil face", "a radiant face", "a criminal face", "a beautiful face", "a handsome face", "a smart face"]
class_index = 2 #which attribute do we want to maximize
tokenizer = SimpleTokenizer()
sot_token = tokenizer.encoder['<|startoftext|>']
eot_token = tokenizer.encoder['<|endoftext|>']
text_descriptions = [f"This is a photo of {label}" for label in attributes]
text_tokens = [[sot_token] + tokenizer.encode(desc) + [eot_token] for desc in text_descriptions]
text_inputs = torch.zeros(len(text_tokens), clip_model.context_length, dtype=torch.long)

for i, tokens in enumerate(text_tokens):
    text_inputs[i, :len(tokens)] = torch.tensor(tokens)

# These are held constant through the optimization, akin to labels
text_inputs = text_inputs.to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(text_inputs).float()
    text_features = F.normalize(text_features, p=2, dim=-1)
text_features.to(device)

# Setting up Transformer
# --------------------------------------------------------------------------------------------------------------
transformer_params = ['OneDirection', 'None']
transformer = transformer_params[0]
transformer_arguments = transformer_params[1]
if transformer_arguments != "None":
    key_value_pairs = transformer_arguments.split(",")
    key_value_pairs = [pair.split("=") for pair in key_value_pairs]
    transformer_arguments = {pair[0]: pair[1] for pair in key_value_pairs}
else:
    transformer_arguments = {}

transformation = getattr(transformations, transformer)(latent_space_dim, vocab_size, **transformer_arguments)
transformation = transformation.to(device)

# function that is used to score the (attribute, image) pair
scoring_fun = get_clip_probs


# Training
# --------------------------------------------------------------------------------------------------------------
# optimizer
optimizer = torch.optim.Adam(transformation.parameters(), lr=0.0002)
losses = common.AverageMeter(name='Loss')

#  training settings
optim_iter = 0
batch_size = 6
train_alpha_a = -0.5 # Lower limit for step sizes
train_alpha_b = 0.5 # Upper limit for step sizes
num_samples = 400000 # Number of samples to train for

# create training set
#np.random.seed(seed=0)
truncation = 1
zs = common.truncated_z_sample(num_samples, latent_space_dim, truncation)

checkpoint_dir = f'/data/scratch/swamiviv/projects/stylegan2-ada-pytorch/clip_steering/results_maximize_{attributes[class_index]}_probability'
pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

# loop over data batches
for batch_start in range(0, num_samples, batch_size):

    # input batch
    s = slice(batch_start, min(num_samples, batch_start + batch_size))
    z = torch.from_numpy(zs[s]).type(torch.FloatTensor).to(device)
    y = None
    step_sizes = (train_alpha_b - train_alpha_a) * \
        np.random.random(size=(batch_size)) + train_alpha_a  # sample step_sizes
    step_sizes_broadcast = np.repeat(step_sizes, latent_space_dim).reshape([batch_size, latent_space_dim])
    step_sizes_broadcast = torch.from_numpy(step_sizes_broadcast).type(torch.FloatTensor).to(device)

    # ganalyze steps
    gan_images = G(z, None)
    gan_images = gan_output_transform(gan_images)
    out_scores = scoring_fun(
        image_inputs=gan_images, encoded_text=text_features, model=clip_model, class_index=class_index,
    )
    # TODO: ignore z vectors with less confident clip scores
    target_scores = out_scores + torch.from_numpy(step_sizes).to(device).float()

    z_transformed = transformation.transform(z, None, step_sizes_broadcast)
    gan_images_transformed = G(z_transformed, None)
    gan_images_transformed = gan_output_transform(gan_images_transformed).to(device)
    out_scores_transformed = scoring_fun(
        image_inputs=gan_images_transformed, encoded_text=text_features, model=clip_model, class_index=class_index,
    ).to(device).float()

    # compute loss
    loss = transformation.criterion(out_scores_transformed, target_scores)

    # backwards
    loss.backward()
    optimizer.step()

    # print loss
    losses.update(loss.item(), batch_size)
    if optim_iter % 100 == 0:
        logger.info(f'[Maximizing score for {attributes[class_index]}] Progress: [{batch_start}/{num_samples}] {losses}')

    if optim_iter % 500 == 0:
        logger.info(f"saving checkpoint at iteration {optim_iter}")
        torch.save(transformation.state_dict(), os.path.join(checkpoint_dir, "pytorch_model_{}.pth".format(batch_start)))
    optim_iter = optim_iter + 1
