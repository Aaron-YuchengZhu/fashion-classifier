import torch
from torchvision import models, transforms
from PIL import Image
import os
import random
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from efficientnet_pytorch import EfficientNet  # 需要安装 efficientnet-pytorch

# Setup
base_dir = '/Users/i/Downloads/COMP9444/Group Project/Datasets'
data_dir = os.path.join(base_dir, 'ATT_augmented')
plot_dir = os.path.join(base_dir, 'plot')
os.makedirs(plot_dir, exist_ok=True)
categories = ['Accessories', 'Bags', 'Clothings', 'Shoes']
device = torch.device('cpu')  # CPU for safety

# Model configs
models_info = {
    'mobilenet_v2': {'best_fold': 5},  # Val Loss 0.0610
    'resnet18': {'best_fold': 4},      # Val Loss 0.1027
    'resnet50': {'best_fold': 5},      # Val Loss 0.0610
    'efficientnet_b0': {'best_fold': 4} # Val Loss 0.0660
}

# Load 500 images (125 per category)
image_paths = []
labels = []
for cat in categories:
    folder = os.path.join(data_dir, cat)
    imgs = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(imgs) < 125:
        print(f"Warning: Only {len(imgs)} images in {cat}, need 125.")
        continue
    image_paths.extend(random.sample(imgs, 125))
    labels.extend([cat] * 125)
print(f"Total samples: {len(image_paths)}")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Test function
def test_model(model_name, fold, paths, labels, transform, device, categories):
    model_map = {
        'mobilenet_v2': models.mobilenet_v2,
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'efficientnet_b0': lambda **kwargs: EfficientNet.from_name('efficientnet-b0')
    }
    model = model_map[model_name](weights=None)
    if model_name.startswith('resnet'):
        model.fc = torch.nn.Linear(model.fc.in_features, 4)
    elif model_name == 'mobilenet_v2':
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)
    elif model_name == 'efficientnet_b0':
        model._fc = torch.nn.Linear(model._fc.in_features, 4)
    model_path = os.path.join(base_dir, 'models', f'{model_name}_fold{fold}.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return None, None, None
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None, None, None
    model = model.to(device).eval()
    true, pred, probs = [], [], []
    for p, l in zip(paths, labels):
        try:
            img = transform(Image.open(p).convert('RGB')).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img)
                prob = torch.softmax(output, dim=1).cpu().numpy()[0]
                true.append(l)
                pred.append(categories[torch.max(output, 1)[1]])
                probs.append(prob)
        except Exception as e:
            print(f"Error processing {p}: {e}")
    return true, pred, probs

# Run tests
results = {}
for model_name in models_info:
    fold = models_info[model_name]['best_fold']
    print(f"Testing {model_name} Fold {fold}...")
    true, pred, probs = test_model(model_name, fold, image_paths, labels, transform, device, categories)
    if true is None or len(true) == 0:
        print(f"No results for {model_name} Fold {fold}.")
        continue
    acc = accuracy_score(true, pred) * 100
    cm = confusion_matrix(true, pred, labels=categories)
    results[model_name] = {'fold': fold, 'accuracy': acc, 'cm': cm, 'true': true, 'pred': pred, 'probs': probs}
    print(f"{model_name} Fold {fold} Accuracy: {acc:.2f}%")