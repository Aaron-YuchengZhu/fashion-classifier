import torch
from torchvision import models, transforms
from PIL import Image
import os
import sys
import argparse
import json
import time
from collections import defaultdict

# Command line arguments
parser = argparse.ArgumentParser(description='Test ResNet50 on fashion images.')
parser.add_argument('--model', type=str, default='best_resnet50.pth',
                    help='Path to the model weights file')
parser.add_argument('--source', type=str, choices=['LAT', 'AAT', 'both'], default='both',
                    help='Source dataset to test on')
parser.add_argument('--sample', type=int, default=0,
                    help='Number of random images to test (0 for all images)')
args = parser.parse_args()

# Load model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)

# load the model weights
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)  # Datasets/
model_path = os.path.join(base_dir, 'models', args.model)
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found.")
    sys.exit(1)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Category mapping
label_map = {0: 'shoes', 1: 'clothing', 2: 'accessories', 3: 'bags'}
reverse_label_map = {'shoes': 0, 'clothing': 1, 'accessories': 2, 'bags': 3}

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

print(f"Test set size: {len(test_pool)} images")

# If sample count is specified, randomly sample
if args.sample > 0:
    import random
    test_images = random.sample(test_pool, min(args.sample, len(test_pool)))
    print(f"Random sampling: {len(test_images)} images")
else:
    test_images = test_pool
    print(f"Using entire test set: {len(test_images)} images")

# Confusion matrix and per-class metrics
confusion_matrix = [[0 for _ in range(4)] for _ in range(4)]
class_total = defaultdict(int)
class_correct = defaultdict(int)

# Testing
correct = 0
total = 0
total_time = 0

for i, item in enumerate(test_images):
    if i % 50 == 0:
        print(f"Progress: {i}/{len(test_images)}...")
        
    image_name = item["filename"]
    true_label = item["label"]
    true_idx = reverse_label_map[true_label]
    
    if 'P' in image_name:
        image_dir = os.path.join(base_dir, 'data', 'LAT/image/')
    else:
        image_dir = os.path.join(base_dir, 'data', 'AAT/image/')
    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found, skipping.")
        continue
    
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        start = time.time()
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted_idx = predicted.item()
            predicted_label = label_map[predicted_idx]
        inference_time = time.time() - start
        total_time += inference_time

        # Update confusion matrix
        confusion_matrix[true_idx][predicted_idx] += 1
        
        # Update class metrics
        class_total[true_label] += 1
        if predicted_label == true_label:
            class_correct[true_label] += 1
            correct += 1
        
        total += 1
        
        if i < 10:  # Just show the first 10 predictions
            source = 'LAT' if 'P' in image_name else 'AAT'
            print(f"Image: {image_name} (from {source}), Predicted: {predicted_label}, True: {true_label}, Time: {inference_time:.4f}s")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

# Overall metrics
accuracy = 100 * correct / total if total > 0 else 0
avg_time = total_time / total if total > 0 else 0
print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")
print(f"Average inference time: {avg_time:.4f}s")

# Per-class metrics
print("\nAccuracy by category:")
for label in ['shoes', 'clothing', 'accessories', 'bags']:
    if class_total[label] > 0:
        class_acc = 100 * class_correct[label] / class_total[label]
        print(f"  {label}: {class_acc:.2f}% ({class_correct[label]}/{class_total[label]})")
    else:
        print(f"  {label}: No test samples")

# Print confusion matrix
print("\nConfusion Matrix:")
print("      Predicted:")
print("       " + " ".join(f"{label:>10}" for label in ["shoes", "clothing", "accessories", "bags"]))
for i, row in enumerate(confusion_matrix):
    print(f"{label_map[i]:>10}:", end=" ")
    for val in row:
        print(f"{val:>10}", end=" ")
    print() 