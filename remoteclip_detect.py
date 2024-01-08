# @title Load packages and download model weights
from huggingface_hub import hf_hub_download
import torch, open_clip
from PIL import Image
from IPython.display import display
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

def visualize_top_results(local_image_names, local_images, results, vis_save_path):
    num_images = len(results)
    plt.figure(figsize=(10, num_images * 5))

    for idx, (img_name, top_keywords) in enumerate(results.items()):
        img_idx = local_image_names.index(img_name)
        
        plt.subplot(num_images, 2, 2 * idx + 1)
        plt.imshow(local_images[img_idx])
        plt.axis("off")
        plt.title(img_name)
        
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

# for model_name in ['RN50'] #, 'ViT-B-32', 'ViT-L-14']: #faster loading
for model_name in ['RN50', 'ViT-B-32', 'ViT-L-14']: #all models
    checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt", cache_dir='checkpoints')
    print(f'{model_name} is downloaded to {checkpoint_path}.')
    
# @title Select Model
model_name = 'ViT-L-14' # @param ['RN50', 'ViT-B-32', 'ViT-L-14']
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

path_to_your_checkpoints = 'checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38'

# ckpt = torch.load(f"{path_to_your_checkpoints}/RemoteCLIP-{model_name}.pt", map_location="qpu")
ckpt = torch.load(f"{path_to_your_checkpoints}/RemoteCLIP-{model_name}.pt", map_location="cuda:0")

message = model.load_state_dict(ckpt)
print(message)
model = model.cuda().eval()

text_queries = [
    "tree", "sky", "building", "person", "car", "cloud", "mountain", "river", 
    "road", "forest", "beach", "bridge", "field", "rooftop", "sunset", "sunrise",
    "bird", "rural", "park", "construction site", "farmland", "lake",
    "playground", "residential area", "highway", "railroad", "boat", "harbor", "cityscape",
    "waterfall", "desert", "snow", "fence", "garden", "parking lot", "monument", "hill",
    "valley", "tower", "streetlight", "intersection", "tunnel", "windmill", "lighthouse", 
    "island", "rainbow", "meadow", "volcano", "canyon", "dam", "downtown", "stadium",
    "airport", "dock", "skyscraper", "carnival", "campsite", "vineyard",
    "orchard", "pasture", "sand", "temple", "church", "mosque", "synagogue", "school",
    "university", "factory", "statue", "fountain", "billboard", "barn", "silo", "wind turbine",
    "solar panel", "power line", "helipad", "golf course", "swimming pool", "greenhouse",
    "rainforest", "hot spring", "geyser", "pier", "marina", "reef", "glacier", 
    "cliff", "cave", "shed", "log cabin", "ruins", "graveyard", "quarry", "mine",
    "tundra", "savannah", "sea"
]

text = tokenizer(text_queries)

image_dir = "/home/jhun/custom_test_4/"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

local_image_paths = []
local_images = []
results_dict = {} 

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path).convert('RGB')
    
    local_image_paths.append(image_path)
    local_images.append(image)
    display(image)

    # @title Predicted probabilities
    image = preprocess(image).unsqueeze(0)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image.cuda())
        text_features = model.encode_text(text.cuda())
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]

    print(f'Predictions of {model_name} for {image_file}:')

    sorted_results = sorted(zip(text_queries, text_probs), key=lambda x: x[1], reverse=True)

    top_keywords = []
    for query, prob in sorted_results[:5]:
        print(f"{query:<40} {prob * 100:5.1f}%")
        top_keywords.append((query, prob * 100))

    print("\n" + "-"*50 + "\n")
    
    results_dict[os.path.basename(image_path)] = top_keywords

vis_save_path = "visualized_results.png"
local_image_names = [os.path.basename(path) for path in local_image_paths]
visualize_top_results(local_image_names, local_images, results_dict, vis_save_path)

output_json_path = "predictions.json"
with open(output_json_path, 'w') as json_file:
    json.dump(results_dict, json_file, ensure_ascii=False, indent=4, default=float32_converter)
            