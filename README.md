<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# CLIP

[\[Blog\]](https://openai.com/blog/clip/) [\[Paper\]](https://arxiv.org/abs/2103.00020) [\[Model Card\]](model-card.md) [\[Colab\]](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)

CLIP (Contrastive Language-Image Pre-Training) is a [neural network](https://www.ultralytics.com/glossary/neural-network-nn) trained on a diverse set of (image, text) pairs sourced from the internet. Developed by OpenAI, it can be instructed using [natural language](https://www.ultralytics.com/glossary/natural-language-processing-nlp) to predict the most relevant text snippet for a given image, without needing task-specific training data. This capability mirrors the [zero-shot learning](https://www.ultralytics.com/glossary/zero-shot-learning) performance seen in models like [GPT-2](https://openai.com/research/gpt-2) and [GPT-3](https://www.ultralytics.com/glossary/gpt-3). Notably, CLIP matches the performance of the original [ResNet50](https://arxiv.org/abs/1512.03385) on [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/) [classification tasks](https://docs.ultralytics.com/tasks/classify/) "zero-shot," meaning it achieves this without using any of the 1.28 million labeled examples from the ImageNet training set, thereby overcoming significant challenges in traditional [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv).

## 🖼️ Approach

CLIP learns visual concepts from natural language supervision. It uses a [Vision Transformer (ViT)](https://www.ultralytics.com/glossary/vision-transformer-vit) or a ResNet as its image encoder and a text transformer for its text encoder. These encoders project images and text into a shared [embedding](https://www.ultralytics.com/glossary/embeddings) space. The model is trained using [contrastive learning](https://www.ultralytics.com/glossary/contrastive-learning) to maximize the cosine similarity between the embeddings of correct image-text pairs while minimizing the similarity for incorrect pairs within a batch.

![CLIP Architecture Diagram](CLIP.png)

## 🚀 Usage

To get started with CLIP, first install [PyTorch](https://pytorch.org/get-started/locally/) (version 1.7.1 or later) and [TorchVision](https://pytorch.org/vision/stable/index.html), along with a few small dependencies. Then, install this repository as a Python package. If you have a machine with a [CUDA](https://developer.nvidia.com/cuda-zone)-enabled GPU, you can use the following commands:

```bash
# Install PyTorch with CUDA support (adjust cudatoolkit version if needed)
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0

# Install required libraries
pip install ftfy regex tqdm

# Install the CLIP package from GitHub
pip install git+https://github.com/openai/CLIP.git
```

Remember to replace `cudatoolkit=11.0` with the appropriate CUDA version for your system or use `cpuonly` if installing on a machine without a GPU.

Here's a basic example demonstrating how to use CLIP to match an image with text descriptions:

```python
import torch
from PIL import Image

import clip

# Check for GPU availability and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and the necessary preprocessing function
# Available models can be listed with clip.available_models()
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess the image
# Replace "CLIP.png" with the path to your image
image_path = "CLIP.png"
try:
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}")
    exit()
except Exception as e:
    print(f"Error processing image: {e}")
    exit()


# Prepare text inputs by tokenizing them
text_descriptions = ["a diagram", "a dog", "a cat"]
text = clip.tokenize(text_descriptions).to(device)

# Perform inference
with torch.no_grad():
    # Encode the image and text
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # Calculate similarity scores (logits)
    # model() returns logits before softmax
    logits_per_image, logits_per_text = model(image, text)

    # Convert logits to probabilities
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probabilities:", probs)
# Example output (exact values may vary slightly):
# Label probabilities: [[0.9927937  0.00421068 0.00299572]]
```

## 🛠️ API

The `clip` module provides the following core functions:

#### `clip.available_models()`

- **Description:** Returns a list of strings naming the available pre-trained CLIP models (e.g., `'RN50'`, `'ViT-B/32'`).
- **Returns:** `List[str]`

#### `clip.load(name, device=..., jit=False, download_root=None)`

- **Description:** Loads a specified CLIP model and its associated TorchVision preprocessing transform. It automatically downloads the model weights if they are not found locally. The `name` can be one of the models returned by `clip.available_models()` or a path to a local checkpoint file (`.pt`).
- **Arguments:**
  - `name` (str): The name of the model or path to a checkpoint.
  - `device` (str or `torch.device`, optional): The device to load the model onto ('cuda', 'cpu', etc.). Defaults to the first available CUDA device, otherwise CPU.
  - `jit` (bool, optional): If `True`, loads the JIT-scripted version of the model. Defaults to `False`.
  - `download_root` (str, optional): Path to download the model weights. Defaults to `~/.cache/clip`.
- **Returns:** `Tuple[torch.nn.Module, Callable]` - A tuple containing the loaded `torch.nn.Module` and the preprocessing function.

#### `clip.tokenize(text: Union[str, List[str]], context_length=77, truncate=False)`

- **Description:** Tokenizes the input text into sequences suitable for the CLIP model. Handles padding and truncation.
- **Arguments:**
  - `text` (Union[str, List[str]]): The text input(s) to tokenize. Can be a single string or a list of strings.
  - `context_length` (int, optional): The fixed sequence length for the model. Defaults to 77.
  - `truncate` (bool, optional): If `True`, truncates the text to fit the `context_length`. Defaults to `False`, raising an error if text exceeds length.
- **Returns:** `torch.LongTensor` - A tensor of shape `(N, context_length)` containing the tokenized sequences, where `N` is the number of input strings.

---

The `model` object returned by `clip.load()` has the following methods:

#### `model.encode_image(image: Tensor)`

- **Description:** Takes a batch of preprocessed images and returns their encoded [feature embeddings](https://www.ultralytics.com/glossary/feature-extraction).
- **Arguments:**
  - `image` (`torch.Tensor`): A tensor of preprocessed images, typically of shape `(N, 3, H, W)`.
- **Returns:** `torch.Tensor` - A tensor containing the image features, shape `(N, embedding_dim)`.

#### `model.encode_text(text: Tensor)`

- **Description:** Takes a batch of tokenized text sequences and returns their encoded feature embeddings.
- **Arguments:**
  - `text` (`torch.Tensor`): A tensor of tokenized text sequences, typically of shape `(N, context_length)`.
- **Returns:** `torch.Tensor` - A tensor containing the text features, shape `(N, embedding_dim)`.

#### `model(image: Tensor, text: Tensor)`

- **Description:** Computes the cosine similarity scores between batches of image and text features.
- **Arguments:**
  - `image` (`torch.Tensor`): A tensor of preprocessed images.
  - `text` (`torch.Tensor`): A tensor of tokenized text sequences.
- **Returns:** `Tuple[torch.Tensor, torch.Tensor]` - Two tensors:
  - `logits_per_image`: Shape `(N, M)`, similarity scores for each image against each text.
  - `logits_per_text`: Shape `(M, N)`, similarity scores for each text against each image.
    The logits are scaled by `100`.

## ✨ More Examples

### Zero-Shot Prediction on CIFAR-100

This example demonstrates CLIP's zero-shot classification capability on the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). It predicts the label for an image without being explicitly trained on CIFAR-100 labels.

```python
import os

import torch
from torchvision.datasets import CIFAR100

import clip

# Ensure cache directory exists
cache_dir = os.path.expanduser("~/.cache")
os.makedirs(cache_dir, exist_ok=True)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Download the CIFAR-100 dataset (if not already downloaded)
try:
    cifar100 = CIFAR100(root=cache_dir, download=True, train=False)
except Exception as e:
    print(f"Error downloading or loading CIFAR100 dataset: {e}")
    exit()

# Select an image from the dataset (e.g., index 3637)
image_index = 3637
try:
    image, class_id = cifar100[image_index]
    print(f"Selected image index: {image_index}, Class ID: {class_id}, Class Name: {cifar100.classes[class_id]}")
except IndexError:
    print(f"Error: Index {image_index} is out of bounds for the dataset.")
    exit()

# Preprocess the image and create text prompts
image_input = preprocess(image).unsqueeze(0).to(device)
# Create text prompts like "a photo of a [CLASS_NAME]"
text_prompts = [f"a photo of a {c}" for c in cifar100.classes]
text_inputs = torch.cat([clip.tokenize(prompt) for prompt in text_prompts]).to(device)

# Calculate image and text features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Normalize features for cosine similarity calculation
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Calculate cosine similarity and convert to probabilities
# Scale similarity by 100 as done during training
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# Get the top 5 predictions
values, indices = similarity[0].topk(5)

# Print the results
print("\nTop 5 predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")

# Expected output might look like:
# Selected image index: 3637, Class ID: 80, Class Name: snake
#
# Top 5 predictions:
#
#            snake: 65.31%
#           turtle: 12.29%
#     sweet_pepper: 3.83%
#           lizard: 1.88%
#        crocodile: 1.75%
# (Exact percentages may vary slightly)
```

This example highlights the use of `encode_image()` and `encode_text()` to get feature embeddings for comparison.

### Linear-Probe Evaluation

This example demonstrates how to perform a linear-probe evaluation. We extract CLIP image features for a dataset ([CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) again) and train a simple linear classifier (Logistic Regression from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)) on top of these features. This is a common way to evaluate the quality of pre-trained features.

```python
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

import clip

# Ensure cache directory exists
cache_dir = os.path.expanduser("~/.cache")
os.makedirs(cache_dir, exist_ok=True)

# Load the CLIP model and preprocessing function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load the CIFAR-100 dataset
try:
    train_dataset = CIFAR100(root=cache_dir, download=True, train=True, transform=preprocess)
    test_dataset = CIFAR100(root=cache_dir, download=True, train=False, transform=preprocess)
except Exception as e:
    print(f"Error downloading or loading CIFAR100 dataset: {e}")
    exit()


# Function to extract features from a dataset
def get_features(dataset, model, device, batch_size=100):
    all_features = []
    all_labels = []
    dataloader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features = model.encode_image(images)
            all_features.append(features.cpu())  # Move features to CPU to save GPU memory
            all_labels.append(labels.cpu())

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()


# Extract features for training and testing sets
print("Extracting training features...")
train_features, train_labels = get_features(train_dataset, model, device)
print("Extracting test features...")
test_features, test_labels = get_features(test_dataset, model, device)

# Train a logistic regression classifier
# Note: The hyperparameter C should ideally be tuned using a validation set.
# See https://docs.ultralytics.com/guides/hyperparameter-tuning/ for tuning strategies.
print("Training logistic regression classifier...")
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=0)  # Reduced verbosity
classifier.fit(train_features, train_labels)

# Evaluate the classifier on the test set
print("Evaluating classifier...")
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.0
print(f"Linear probe accuracy = {accuracy:.3f}%")

# Expected output might be around 70-80% accuracy for ViT-B/32 on CIFAR-100
```

**Note:** The regularization strength `C` is a crucial [hyperparameter](https://www.ultralytics.com/glossary/hyperparameter-tuning). Its optimal value should be found through techniques like [cross-validation](https://www.ultralytics.com/glossary/cross-validation) or using a dedicated validation split, rather than using a fixed value as shown here for simplicity. Refer to our [Hyperparameter Tuning Guide](https://docs.ultralytics.com/guides/hyperparameter-tuning/) for more details.

## 🔗 See Also

- **[OpenCLIP](https://github.com/mlfoundations/open_clip):** An open-source implementation offering various pre-trained CLIP models, including larger ones like ViT-G/14, trained on the LAION dataset.
- **[Hugging Face Transformers `CLIPModel`](https://huggingface.co/docs/transformers/model_doc/clip):** Provides an implementation of CLIP integrated within the popular [Hugging Face](https://www.ultralytics.com/glossary/hugging-face) ecosystem, facilitating easier use with other Transformers models and tools.
- **[Ultralytics YOLO Models](https://docs.ultralytics.com/models/):** Explore state-of-the-art [object detection](https://www.ultralytics.com/glossary/object-detection) models like [YOLOv8](https://docs.ultralytics.com/models/yolov8/) and [YOLOv10](https://docs.ultralytics.com/models/yolov10/) which can be used alongside or as alternatives to CLIP for various vision tasks.
- **[Multi-Modal Learning Glossary](https://www.ultralytics.com/glossary/multi-modal-learning):** Understand the broader context of models that process information from multiple modalities like text and images.

## 🤝 Contributing

Contributions to enhance CLIP or integrate it further are welcome! Please see the [Ultralytics Contributing Guidelines](https://docs.ultralytics.com/help/contributing/) for more information on how to get started. We appreciate your help in improving our open-source resources for the AI community!
