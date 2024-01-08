import transformers
print(transformers.__version__)
import torch.nn.functional as F
import datetime

# Imports
import os
import datasets
from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt
import requests
import random
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from pdb import set_trace
from tqdm import tqdm

from transformers import (
    VisionTextDualEncoderProcessor,
    VisionTextDualEncoderModel,
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

import clip
from torchvision.datasets import CIFAR100

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import glob

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt")


# keywords = [
#     "drone", "tree", "sky", "building", "person", "car", "cloud", "mountain", "river", 
#     "road", "forest", "ocean", "beach", "bridge", "field", "rooftop", "sunset", "sunrise",
#     "bird", "urban", "rural", "park", "vehicle", "construction site", "farmland", "lake",
#     "playground", "residential area", "highway", "railroad", "boat", "harbor", "cityscape",
#     "waterfall", "desert", "snow", "fence", "garden", "parking lot", "monument", "hill",
#     "valley", "tower", "streetlight", "intersection", "tunnel", "windmill", "lighthouse", 
#     "island", "rainbow", "swamp", "meadow", "volcano", "canyon", "dam", "downtown", "stadium",
#     "airport", "dock", "skyscraper", "mall", "fair", "carnival", "campsite", "vineyard",
#     "orchard", "pasture", "sand dune", "temple", "church", "mosque", "synagogue", "school",
#     "university", "factory", "statue", "fountain", "billboard", "barn", "silo", "wind turbine",
#     "solar panel", "power line", "helipad", "golf course", "swimming pool", "greenhouse",
#     "rainforest", "hot spring", "geyser", "pier", "marina", "reef", "peninsula", "glacier", 
#     "cliff", "cave", "hut", "shed", "log cabin", "ruins", "graveyard", "quarry", "mine",
#     "tundra", "savannah", "jungle", "archipelago",  "sea"
# ]

keywords = [
    "tree","building", "tower", "car", "mountain", "water", "person",
    "road", "forest", "ground", "field", "boat", "parking station"
]

# Base Model 
model = VisionTextDualEncoderModel.from_vision_text_pretrained(
    "openai/clip-vit-base-patch32", "roberta-base"
)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

model.save_pretrained("clip-roberta")
processor.save_pretrained("clip-roberta")

# Arguments 
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "The data directory containing input files."})
    image_column: Optional[str] = field(
        default="image_path",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    caption_column: Optional[str] = field(
        default="caption",
        metadata={"help": "The name of the column in the datasets containing the image captions."},
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    
args_dict = {
    'output_dir': './clip-roberta-finetuned',
    'model_name_or_path': './clip-roberta',
    'data_dir': './data',
    'dataset_name': 'arampacha/rsicd',
    'image_column': 'image',
    'caption_column': 'captions',
    'remove_unused_columns': False,
    'per_device_train_batch_size': 64,
    'per_device_eval_batch_size': 64,
    'learning_rate': 5e-05,
    'warmup_steps': 0,
    'weight_decay': 0.1,
    'overwrite_output_dir': True,
    'push_to_hub': False,
    'num_train_epochs': 2
}

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_dict(args_dict)

# model_args, data_args

# Dataset Preparation 
class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )
    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x
    
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }

# dataset = datasets.load_dataset("arampacha/rsicd")
dataset = load_dataset("arampacha/rsicd", cache_dir='/media/jhun/4TBHDD/datasets')
# dataset = datasets.load_dataset('./path_to_my_data', split='train')

print(dataset)

# Model Preparation
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
)

image_processor = AutoImageProcessor.from_pretrained(
    model_args.image_processor_name or model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)

model = AutoModel.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
config = model.config

set_seed(training_args.seed)

image_transformations = Transform(
    config.vision_config.image_size, image_processor.image_mean, image_processor.image_std
)
image_transformations = torch.jit.script(image_transformations)

def tokenize_captions(examples):
    captions = [example[0] for example in examples[data_args.caption_column]]
    text_inputs = tokenizer(captions, max_length=data_args.max_seq_length, padding="max_length", truncation=True)
    examples["input_ids"] = text_inputs.input_ids
    examples["attention_mask"] = text_inputs.attention_mask
    return examples

def transform_images(examples):
    images = [torch.tensor(np.array(image)).permute(2, 0, 1) for image in examples[data_args.image_column]]
    examples["pixel_values"] = [image_transformations(image) for image in images]
    return examples

def filter_corrupt_images(examples):
    """remove problematic images"""
    valid_images = []
    for image_file in examples[data_args.image_column]:
        try:
            Image.open(image_file)
            valid_images.append(True)
        except Exception:
            valid_images.append(False)
    return valid_images

train_dataset = dataset["train"]
train_dataset = train_dataset.map(
    function=tokenize_captions,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
    load_from_cache_file=not data_args.overwrite_cache,
    desc="Running tokenizer on train dataset",
)
train_dataset.set_transform(transform_images)

print(train_dataset)

eval_dataset = dataset["valid"]
eval_dataset = eval_dataset.map(
    function=tokenize_captions,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
    load_from_cache_file=not data_args.overwrite_cache,
    desc="Running tokenizer on validation dataset",
)
eval_dataset.set_transform(transform_images)

print(train_dataset, eval_dataset)

_, preprocess = clip.load("ViT-B/32")

processor =  VisionTextDualEncoderProcessor(image_processor, tokenizer)

cifar100 = CIFAR100(os.path.expanduser("~/.cache"), download=True)
cifar100_classes = cifar100.classes

###### Finetuning CLIP 

def show_results_for_keywords(model, local_images, keywords, top_n=5):
    image_results = {img_idx: [] for img_idx in range(len(local_images))}
    
    print(f"Processing {len(local_images)} local images for {len(keywords)} keywords...")

    for idx, keyword in enumerate(keywords):
        print(f"Processing keyword {idx+1}/{len(keywords)}: {keyword} ...")
        
        inputs = processor(text=[keyword]*len(local_images), images=local_images, return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        inputs['pixel_values'] = inputs['pixel_values'].cuda()

        model = model.cuda()
        outputs = model(**inputs)
        logits = outputs.logits_per_image

        # 현재 키워드에 대한 각 이미지의 로짓을 가져옴
        for img_idx in range(len(local_images)):
            score = logits[img_idx, 0].item()  # 각 이미지에 대한 로짓 점수
            image_results[img_idx].append((keyword, score))

    # 각 이미지에 대해 점수에 따라 키워드를 정렬
    for img_idx, keyword_scores in image_results.items():
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        image_results[img_idx] = keyword_scores[:top_n]

    print("Processing completed.")
    return image_results

# def show_results_for_keywords(model, local_images, keywords, top_n=5):
#     image_results = {img_idx: [] for img_idx in range(len(local_images))}
    
#     print(f"Processing {len(local_images)} local images for {len(keywords)} keywords...")

#     # CLIP에서 가져온 preprocess 함수를 사용하여 이미지 전처리
#     preprocessed_images = [preprocess(img) for img in local_images]
#     image_inputs = torch.stack(preprocessed_images).cuda()

#     for idx, keyword in enumerate(keywords):
#         print(f"Processing keyword {idx+1}/{len(keywords)}: {keyword} ...")
        
#         # 텍스트는 여전히 tokenizer를 사용하여 처리
#         text_inputs = tokenizer([keyword]*len(local_images), return_tensors="pt", padding=True, truncation=True)
#         text_inputs = {key: val.cuda() for key, val in text_inputs.items()}

#         # 모델을 사용하여 예측 실행
#         with torch.no_grad():
#             outputs = model(pixel_values=image_inputs, **text_inputs)
#             logits = outputs.logits_per_image

#         # 현재 키워드에 대한 각 이미지의 로짓을 가져옴
#         for img_idx in range(len(local_images)):
#             score = logits[img_idx, 0].item()  # 각 이미지에 대한 로짓 점수
#             image_results[img_idx].append((keyword, score))

#     # 각 이미지에 대해 점수에 따라 키워드를 정렬
#     for img_idx, keyword_scores in image_results.items():
#         keyword_scores.sort(key=lambda x: x[1], reverse=True)
#         image_results[img_idx] = keyword_scores[:top_n]

#     print("Processing completed.")
#     return image_results

def visualize_top_results(local_images, results):
    num_images = len(results)
    plt.figure(figsize=(10, num_images * 5))

    for idx, (img_idx, top_keywords) in enumerate(results.items()):
        plt.subplot(num_images, 2, 2 * idx + 1)
        plt.imshow(local_images[img_idx])
        plt.axis("off")
        plt.title(f"Image {img_idx + 1}")
        
        plt.subplot(num_images, 2, 2 * idx + 2)
        
        top_keywords.sort(key=lambda x: abs(x[1]), reverse=True)
        
        keywords = [f"{k} ({s:.2f})" for k, s in top_keywords]
        scores = [abs(s) for _, s in top_keywords]
        
        y = np.arange(len(keywords))
        plt.grid()
        plt.barh(y, scores, align='center', color='skyblue')
        plt.gca().invert_yaxis() 
        plt.gca().set_axisbelow(True)
        plt.yticks(y, keywords)
        plt.xlabel("Score (Absolute Value)")

    plt.tight_layout()
    plt.show()

image_folder_path = "/home/jhun/custom_test_2"

extensions = ['jpg', 'png', 'JPG', 'JPEG', 'jpeg']
all_image_paths = []

for ext in extensions:
    all_image_paths.extend(glob.glob(f"{image_folder_path}/*.{ext}"))
    
local_images = [Image.open(image_path) for image_path in all_image_paths]

results = show_results_for_keywords(model, local_images, keywords)
visualize_top_results(local_images, results)

results = show_results_for_keywords(model, local_images, cifar100_classes)
visualize_top_results(local_images, results)

# results = show_results_for_keywords(model, local_images, keywords)
# visualize_top_results(local_images, results)

# 8. Initalize our trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
)

# 9. Training
train_result = trainer.train()
trainer.log_metrics("train", train_result.metrics)
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)

# results = show_results_for_keywords(model, local_images, keywords)
# visualize_top_results(local_images, results)

results = show_results_for_keywords(model, local_images, cifar100_classes)
visualize_top_results(local_images, results)


# def show_results_for_keywords(model, local_images, keywords, top_n=5):
#     image_results = {img_idx: [] for img_idx in range(len(local_images))}
#     num_local_images = len(local_images)
#     num_keywords = len(keywords)

#     model = model.cuda()
#     print(f"Processing {num_local_images} local images for {num_keywords} keywords...")

#     for idx, keyword in enumerate(keywords):
#         print(f"Processing keyword {idx+1}/{num_keywords}: {keyword} ...")
        
#         inputs = processor(text=[keyword]*num_local_images, images=local_images, return_tensors="pt", padding=True)
#         inputs = {key: value.cuda() for key, value in inputs.items()}

#         outputs = model(**inputs)
#         logits = outputs.logits_per_image

#         for img_idx in range(num_local_images):
#             score = logits[img_idx, 0].item()
#             image_results[img_idx].append((keyword, score))

#     for img_idx, keyword_scores in image_results.items():
#         keyword_scores.sort(key=lambda x: x[1], reverse=True)
#         image_results[img_idx] = keyword_scores[:top_n]

#     print("Processing completed.")
#     return image_results

# def process_batch(model, images, keywords):
#     texts = [kw for kw in keywords for _ in images]
#     extended_images = images * len(keywords)
    
#     print("Preparing inputs for the model...")
#     inputs = processor(text=texts, images=extended_images, return_tensors="pt", padding=True)

#     # Convert everything to FP16 except input_ids
#     for key, value in tqdm(inputs.items(), desc="Converting to CUDA", leave=True):
#         if key != "input_ids":
#             inputs[key] = value.cuda().half()
#         else:
#             inputs[key] = value.cuda()

#     print("Processing model inputs...")
#     outputs = model(**inputs)
#     logits = outputs.logits_per_image
#     print("Batch processing completed.")
#     return logits

# def show_results_for_keywords(model, local_images, keywords, top_n=5):
#     model = model.cuda().half()  # Convert model to FP16
#     num_local_images = len(local_images)
#     image_results = {img_idx: [] for img_idx in range(num_local_images)}
#     num_keywords = len(keywords)
    
#     print(f"Processing {num_local_images} local images for {num_keywords} keywords...")
    
#     logits = process_batch(model, local_images, keywords)
#     for kw_idx, keyword in enumerate(keywords):
#         for img_idx in range(num_local_images):
#             score = logits[kw_idx * num_local_images + img_idx, 0].item()
#             image_results[img_idx].append((keyword, score))
#         print(f"Processing keyword {kw_idx+1}/{num_keywords} completed.")
    
#     for img_idx, keyword_scores in image_results.items():
#         keyword_scores.sort(key=lambda x: x[1], reverse=True)
#         image_results[img_idx] = keyword_scores[:top_n]

#     print("Processing completed.")
#     return image_results