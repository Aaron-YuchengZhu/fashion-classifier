import torch
from torchvision import models, transforms
from PIL import Image
import os
import sys
import argparse
import json
import random

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Test a fashion classification model on images.')
parser.add_argument('--model', type=str, default='best_fashion_classifier_v1.pth',
                    help='Path to the model weights file (e.g., best_fashion_classifier_v1.pth)')
parser.add_argument('--source', type=str, choices=['LAT', 'AAT', 'both'], default='both',
                    help='Source dataset to test on: LAT, AAT, or both')
parser.add_argument('--num_images', type=int, default=10,
                    help='Number of images to randomly select for testing')
args = parser.parse_args()

# Load the model
# Device selection based on platform
if sys.platform == 'darwin':  # macOS
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
elif sys.platform == 'win32':  # Windows
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:  # Linux or others
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)
# Dynamically get the path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, args.model)
# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found. Please train the model or check the file path.")
    sys.exit(1)
model.load_state_dict(torch.load(model_path, weights_only=True))
model = model.to(device)
model.eval()

# Class mapping
label_map = {0: 'shoes', 1: 'clothing', 2: 'accessories', 3: 'bags'}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load LAT.json and AAT.json
lat_json_path = os.path.join(current_dir, 'LAT/label/LAT.json')
aat_json_path = os.path.join(current_dir, 'AAT/label/AAT.json')


# Function to parse JSON and extract image names and labels
def parse_json(json_path, dataset_name):
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_label_pairs = []
    for item in data:
        # Get the ground truth answer (correct answer)
        gt_index = item['gt']
        answers = item['answers']
        if not (0 <= gt_index < len(answers)):
            print(f"Warning: Invalid gt index {gt_index} for answers {answers}, skipping.")
            continue
        # Extract the correct answer (e.g., "Top_P00440102")
        correct_answer = answers[gt_index]
        # Parse the image name and category (e.g., "Top_P00440102" -> "P00440102.jpg", "clothing")
        parts = correct_answer.split('_')
        if len(parts) != 2:
            print(f"Warning: Invalid answer format {correct_answer}, skipping.")
            continue
        category, image_id = parts
        image_name = image_id + '.jpg'  # e.g., P00440102.jpg
        # Map category to label
        if 'Shoes' in category:
            label = 'shoes'
        elif 'Top' in category or 'Dress' in category or 'Pants' in category or 'Outwear' in category or 'Skirt' in category:
            label = 'clothing'
        elif 'Bag' in category:
            label = 'bags'
        else:
            label = 'accessories'
        image_label_pairs.append((image_name, label, dataset_name))
    return image_label_pairs


# Load images from LAT and AAT
lat_images = parse_json(lat_json_path, 'LAT') if os.path.exists(lat_json_path) else []
aat_images = parse_json(aat_json_path, 'AAT') if os.path.exists(aat_json_path) else []

# Select images based on --source argument
if args.source == 'LAT':
    available_images = lat_images
elif args.source == 'AAT':
    available_images = aat_images
else:  # both
    available_images = lat_images + aat_images

# Randomly select images
if len(available_images) < args.num_images:
    print(f"Warning: Only {len(available_images)} images available, requested {args.num_images}")
    test_images = available_images
else:
    test_images = random.sample(available_images, args.num_images)

# Test the selected images
correct = 0
total = len(test_images)

for image_name, true_label, dataset_name in test_images:
    image_dir = os.path.join(current_dir, f'{dataset_name}/image/')
    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found, skipping.")
        total -= 1
        continue
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_label = label_map[predicted.item()]

    print(f"Image: {image_name} (from {dataset_name}), Predicted: {predicted_label}, True: {true_label}")
    if predicted_label == true_label:
        correct += 1

# Calculate accuracy
if total > 0:
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
else:
    print("No valid images found for testing.")