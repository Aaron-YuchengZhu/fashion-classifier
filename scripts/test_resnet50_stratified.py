import torch
from torchvision import models, transforms
from PIL import Image
import os
import argparse
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.gridspec as gridspec

# Command line arguments
parser = argparse.ArgumentParser(description='Test ResNet50 on ATT_augmented.')
parser.add_argument('--model', type=str, default='resnet50_fold1.pth')
parser.add_argument('--num_images', type=int, default=100)
parser.add_argument('--runs', type=int, default=5, help='Number of test runs')
parser.add_argument('--viz_samples', type=int, default=8, help='Number of sample images to visualize')
args = parser.parse_args()

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'models', args.model)
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found.")
    exit(1)
model.load_state_dict(torch.load(model_path, weights_only=True))
model = model.to(device)
model.eval()

# Class mapping
label_map = {0: 'Accessories', 1: 'Bags', 2: 'Clothings', 3: 'Shoes'}
idx_to_class = {i: c for i, c in enumerate(sorted(['Accessories', 'Bags', 'Clothings', 'Shoes']))}
class_to_idx = {c: i for i, c in idx_to_class.items()}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Display image preprocessing
display_transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

# Load all images
data_dir = os.path.join(base_dir, 'data', 'ATT_augmented')
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
        if os.path.isfile(img_path) and img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(img_path)
            labels.append(category)

print(f"Total samples: {len(image_paths)}")
print(f"Label distribution: {Counter(labels)}")
if len(image_paths) == 0:
    print("Error: No images found.")
    exit(1)

# Multiple test runs
for run in range(args.runs):
    print(f"\nRun {run + 1}/{args.runs}")
    seed = random.randint(0, 1000)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.num_images, random_state=seed)
    for _, test_idx in sss.split(image_paths, labels):
        test_paths = [image_paths[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

    # Testing
    correct = 0
    total = 0
    true_labels = []
    pred_labels = []
    errors = []
    # Save some images and their prediction results for visualization
    viz_images = []
    viz_true = []
    viz_pred = []
    viz_paths = []
    
    for img_path, true_label in zip(test_paths, test_labels):
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found, skipping.")
            continue
        image = Image.open(img_path).convert('RGB')
        # Save the original image for display
        orig_img = display_transform(image)
        
        # Transform for model
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            predicted_label = label_map[predicted.item()]

        true_labels.append(true_label)
        pred_labels.append(predicted_label)
        
        # Collect samples for visualization
        if len(viz_images) < args.viz_samples:
            viz_images.append(orig_img)
            viz_true.append(true_label)
            viz_pred.append(predicted_label)
            viz_paths.append(os.path.basename(img_path))
        
        if predicted_label == true_label:
            correct += 1
        else:
            errors.append(f"Image: {os.path.basename(img_path)}, Predicted: {predicted_label}, True: {true_label}")
        total += 1

    # Output results
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Run {run + 1} Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"True label distribution: {Counter(true_labels)}")
    print(f"Predicted label distribution: {Counter(pred_labels)}")
    if errors:
        print("Errors:")
        for error in errors:
            print(error)
    
    # Generate confusion matrix
    y_true = [class_to_idx[lbl] for lbl in true_labels]
    y_pred = [class_to_idx[lbl] for lbl in pred_labels]
    cm = confusion_matrix(y_true, y_pred)
    
    # Save confusion matrix image
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Run {run+1}')
    cm_path = os.path.join(base_dir, 'results', f'confusion_matrix_resnet50_run{run+1}.png')
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Visualize some prediction results
    if viz_images:
        n_samples = len(viz_images)
        fig = plt.figure(figsize=(15, 3 * ((n_samples + 3) // 4)))
        gs = gridspec.GridSpec(((n_samples + 3) // 4), 4)
        
        for i, (img, true_lbl, pred_lbl, img_path) in enumerate(zip(viz_images, viz_true, viz_pred, viz_paths)):
            ax = plt.subplot(gs[i])
            ax.imshow(img)
            title_color = 'green' if true_lbl == pred_lbl else 'red'
            ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl}", color=title_color)
            ax.axis('off')
        
        plt.tight_layout()
        viz_path = os.path.join(base_dir, 'results', f'sample_predictions_resnet50_run{run+1}.png')
        plt.savefig(viz_path)
        plt.close()
        print(f"Sample predictions visualization saved to: {viz_path}") 