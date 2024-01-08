import transformers
print(transformers.__version__)
import torch.nn.functional as F
import argparse
import datetime
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
import glob
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
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

import json

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt")

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

###### Finetuning CLIP 

def show_results_for_keywords(model, local_image_paths, keywords, top_n=5):

    image_results = {img_path: [] for img_path in local_image_paths}
    
    local_images = [Image.open(image_path) for image_path in local_image_paths]
    
    print(f"Processing {len(local_images)} local images for {len(keywords)} keywords...")

    for idx, keyword in enumerate(keywords):
        print(f"Processing keyword {idx+1}/{len(keywords)}: {keyword} ...")
        
        inputs = processor(text=[keyword]*len(local_images), images=local_images, return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        inputs['pixel_values'] = inputs['pixel_values'].cuda()

        model = model.cuda()
        outputs = model(**inputs)
        print(outputs.keys())

        text_features = outputs['text_embeds']
        image_features = outputs['image_embeds']

        # Normalize the features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate the cosine similarity
        similarity = (100.0 * image_features @ text_features.T).detach().cpu().numpy()
        
        # 현재 키워드에 대한 각 이미지의 유사도를 가져옴
        for img_idx in range(len(local_images)):
            img_path = local_image_paths[img_idx]  # Get the path of the image being processed
            score = similarity[img_idx][img_idx]  # 각 이미지에 대한 유사도 점수
            image_results[img_path].append((keyword, score))  # Use img_path to update image_results

    # 각 이미지에 대해 유사도 값으로 키워드를 정렬
    for img_path, keyword_scores in image_results.items():
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        image_results[img_path] = keyword_scores[:top_n]

    print("Processing completed.")
    return image_results

def visualize_top_results(local_image_paths, local_images, results, vis_save_path):
    num_images = len(results)
    plt.figure(figsize=(10, num_images * 5))

    for idx, (img_path, top_keywords) in enumerate(results.items()):
        # 이미지의 경로로부터 해당 이미지의 인덱스를 찾습니다.
        img_idx = local_image_paths.index(img_path)
        
        plt.subplot(num_images, 2, 2 * idx + 1)
        plt.imshow(local_images[img_idx])
        plt.axis("off")
        plt.title(f"Image {img_idx + 1}")
        
        plt.subplot(num_images, 2, 2 * idx + 2)
        
        # 원래의 값을 라벨에 사용
        keywords = [f"{k} ({s:.2f})" for k, s in top_keywords]
        # 음수 값은 0으로 바꾸어 차트에 표시
        scores = [max(0, s) for _, s in top_keywords]
        
        y = np.arange(len(keywords))
        plt.grid()
        plt.barh(y, scores, align='center', color='skyblue')
        plt.gca().invert_yaxis() 
        plt.gca().set_axisbelow(True)
        plt.yticks(y, keywords)
        plt.xlabel("Score")

    plt.tight_layout()
    plt.savefig(vis_save_path, format='png', dpi=300)
    plt.close()  # Close the figure

def float32_converter(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the model.")
    parser.add_argument("--train", action="store_true", help="If set, training will be performed.")
    parser.add_argument("--eval", action="store_true", help="If set, evaluation will be performed.")
    parser.add_argument("--model_weights_save_path", type=str, default=f"/workspace/weights/epoch50.pth")
    parser.add_argument("--model_load_weights_path", type=str, default="/workspace/weights/epoch50.pth")
    parser.add_argument("--image_folder_path", type=str, default="/workspace/images/", 
                        help="Path to folder containing images for evaluation.")
    parser.add_argument("--vis_save_path", type=str, default="/workspace//visualization/custom_epoch50.png")
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--results_save_json", type=str, default="/workspace/epoch_50_results.json", help="Path to save the results JSON.")

    args = parser.parse_args()
    
    args_dict = {
    'output_dir': './clip-roberta-finetuned',
    'model_name_or_path': './clip-roberta',
    'data_dir': './data',
    'dataset_name': 'arampacha/rsicd',
    'logging_dir': './logs',
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
    'num_train_epochs': args.num_train_epochs
    }  

    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y%m%d_%H%M%S")
    
    custom_classes = [
    "Tree",
    "Rock",
    "Sea",
    "Seaweed",
    "Building",
    "Fishfarm",
    "Sand",
    "Car",
    "Road",
    "Breakwater",
    "Green wood"
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

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(args_dict)

    # dataset = datasets.load_dataset("arampacha/rsicd")
    dataset = load_dataset("arampacha/rsicd", cache_dir='/media/jhun/4TBHDD2/datasets')
    # dataset = datasets.load_dataset('./path_to_my_data', split='train')

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

    train_dataset = dataset["train"]
    train_dataset = train_dataset.map(
        function=tokenize_captions,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )
    train_dataset.set_transform(transform_images)

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

    # cifar100 = CIFAR100(os.path.expanduser("~/.cache"), download=True)
    # cifar100_classes = cifar100.classes

    image_folder_path = args.image_folder_path

    extensions = ['jpg', 'png', 'JPG', 'JPEG', 'jpeg']
    all_image_paths = []

    for ext in extensions:
        all_image_paths.extend(glob.glob(f"{image_folder_path}/*.{ext}"))
        
    local_images = [Image.open(image_path) for image_path in all_image_paths]

    if args.train:
        # Training code
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
        )
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)

        torch.save(model.state_dict(), args.model_weights_save_path)
        print(f"Model saved to {args.model_weights_save_path}")
        
    if args.eval:
        # Evaluation code
        model_for_eval = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model_for_eval.load_state_dict(torch.load(args.model_load_weights_path))
        model_for_eval.cuda().eval()

        trainer_for_eval = Trainer(
            model=model_for_eval,
            args=training_args,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
        )

        metrics = trainer_for_eval.evaluate()
        trainer_for_eval.log_metrics("eval", metrics)
        
        results = show_results_for_keywords(model_for_eval, all_image_paths, custom_classes)
        torch.cuda.empty_cache()
        visualize_top_results(all_image_paths, local_images, results, vis_save_path=args.vis_save_path)
        torch.cuda.empty_cache()

        updated_results = {}

        for path, values in results.items():
            file_name = os.path.basename(path)
            updated_results[file_name] = values
        print(updated_results)
        
        with open(args.results_save_json, 'w') as json_file:
            json.dump(updated_results, json_file, ensure_ascii=False, indent=4, default=float32_converter)
        
        
        ## tensorboard --logdir=./logs


