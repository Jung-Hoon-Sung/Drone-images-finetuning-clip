import numpy as np
import torch
import clip
import os
import skimage
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch
import glob
from torchvision.datasets import CIFAR100

print("Torch version:", torch.__version__)
print(clip.available_models())

model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)

custom_image_directory = "/home/jhun/custom_test/"
image_paths = glob.glob(os.path.join(custom_image_directory, "*.jpg"))
images = []

for image_path in image_paths:
    custom_image = Image.open(image_path).convert("RGB")
    custom_preprocessed_image = preprocess(custom_image)
    images.append(custom_preprocessed_image)

image_input = torch.tensor(np.stack(images)).cuda()

with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)

cifar100 = CIFAR100(os.path.expanduser("~/.cache"), transform=preprocess, download=True)

text_descriptions = [f"This is a photo of a {label}" for label in cifar100.classes]
text_tokens = clip.tokenize(text_descriptions).cuda()

with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

def display_custom_image_predictions(image_paths, top_probs, top_labels):
    plt.figure(figsize=(8, 12))
    
    for i, (image, prob, label_indices) in enumerate(zip(image_paths, top_probs, top_labels)):
        plt.subplot(len(image_paths), 2, 2 * i + 1)
        plt.imshow(Image.open(image))
        plt.axis("off")
    
        plt.subplot(len(image_paths), 2, 2 * i + 2)
        y = np.arange(prob.shape[-1])
        plt.grid()
        plt.barh(y, prob)
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [cifar100.classes[index] for index in label_indices.numpy()])
        plt.xlabel("probability")
    
    plt.tight_layout()
    plt.show()

display_custom_image_predictions(image_paths, top_probs, top_labels)
