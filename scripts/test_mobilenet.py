import torch
from torchvision import models, transforms
from PIL import Image
import os
import sys
import argparse
import json
import random
import time

# Command line arguments
parser = argparse.ArgumentParser(description='Test MobileNetV2 on fashion images.')
parser.add_argument('--model', type=str, default='best_mobilenet_v3.pth',
                    help='Path to the model weights file')
parser.add_argument('--source', type=str, choices=['LAT', 'AAT', 'both'], default='both',
                    help='Source dataset to test on')
parser.add_argument('--num_images', type=int, default=10,
                    help='Number of images to test')
args = parser.parse_args()

# Load model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
model = models.mobilenet_v2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, 4)
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)  # Datasets/
model_path = os.path.join(base_dir, args.model)
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found.")
    sys.exit(1)
model.load_state_dict(torch.load(model_path, weights_only=True))
model = model.to(device)
model.eval()

# Category mapping
label_map = {0: 'shoes', 1: 'clothing', 2: 'accessories', 3: 'bags'}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test data
json_path = os.path.join(base_dir, 'data', 'test.json')
with open(json_path, 'r') as f:
    data = json.load(f)

# Filter data by source
lat_data = [item for item in data if 'P' in item["filename"]]
aat_data = [item for item in data if 'A' in item["filename"]]
if args.source == 'LAT':
    test_pool = lat_data
elif args.source == 'AAT':
    test_pool = aat_data
else:  # both
    test_pool = data

test_images = random.sample(test_pool, min(args.num_images, len(test_pool)))

# Testing
correct = 0
total = 0
for item in test_images:
    image_name = item["filename"]
    true_label = item["label"]
    image_dir = os.path.join(base_dir, 'data', 'LAT/image/' if 'P' in image_name else 'data/AAT/image/')
    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found, skipping.")
        continue
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_label = label_map[predicted.item()]
    inference_time = time.time() - start

    source = 'LAT' if 'P' in image_name else 'AAT'
    print(f"Image: {image_name} (from {source}), Predicted: {predicted_label}, True: {true_label}, Time: {inference_time:.4f}s")
    if predicted_label == true_label:
        correct += 1
    total += 1

accuracy = 100 * correct / total if total > 0 else 0
print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")