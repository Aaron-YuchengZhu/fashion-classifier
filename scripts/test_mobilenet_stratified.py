import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Device setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Model loading
model = models.mobilenet_v2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, 4)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'models', 'mobilenet_v2_fold5.pth')
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found.")
    exit(1)
model.load_state_dict(torch.load(model_path, weights_only=True))
model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data loading
data_dir = os.path.join(base_dir, 'ATT_augmented')
categories = ['Accessories', 'Bags', 'Clothings', 'Shoes']
image_paths = []
labels = []

for category in categories:
    folder_path = os.path.join(data_dir, category)
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} not found.")
        exit(1)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(img_path)
            labels.append(category)

print(f"Total samples: {len(image_paths)}")

# Select 100 images (fixed seed for reproducibility)
np.random.seed(42)
indices = np.random.choice(len(image_paths), 100, replace=False)
test_paths = [image_paths[i] for i in indices]
test_labels = [labels[i] for i in indices]

# Testing
label_map = {0: 'Accessories', 1: 'Bags', 2: 'Clothings', 3: 'Shoes'}
true_labels = []
pred_labels = []
for img_path, true_label in zip(test_paths, test_labels):
    try:
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            pred_labels.append(label_map[predicted.item()])
        true_labels.append(true_label)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        continue

# Metrics
cm = confusion_matrix(true_labels, pred_labels, labels=categories)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None)
overall_f1 = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')[2]
accuracy = np.sum(np.diag(cm)) / np.sum(cm) * 100

# Print results
print("\nTest Results")
print(f"{'Category':<12} {'Accuracy (%)':<12} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
print("-" * 54)
for i, cat in enumerate(categories):
    acc = 100 * cm[i,i] / cm[i,:].sum() if cm[i,:].sum() > 0 else 0
    print(f"{cat:<12} {acc:<12.2f} {precision[i]:<10.2f} {recall[i]:<10.2f} {f1[i]:<10.2f}")
print(f"{'Overall':<12} {accuracy:<12.2f} {'-':<10} {'-':<10} {overall_f1:<10.2f}")