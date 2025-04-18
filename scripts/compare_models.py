import torch
from torchvision import models, transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, average_precision_score
from efficientnet_pytorch import EfficientNet
import pickle

# Setup
base_dir = '/Users/i/Downloads/COMP9444/Group Project/Datasets'
data_dir = os.path.join(base_dir, 'ATT_augmented')
plot_dir = os.path.join(base_dir, 'plot')
os.makedirs(plot_dir, exist_ok=True)
categories = ['Accessories', 'Bags', 'Clothings', 'Shoes']
device = torch.device('cpu')  # CPU for safety

# Model configs
models_info = {
    'mobilenet_v2': {
        'fold': 5,  # Val Loss 0.0610
        'train_loss': [0.6973, 0.3017, 0.2339, 0.1788, 0.1709, 0.1498, 0.1433, 0.1303, 0.1152, 0.1174],
        'val_loss': [0.3036, 0.2181, 0.1640, 0.1605, 0.1166, 0.1111, 0.0917, 0.0749, 0.0750, 0.0610]
    },
    'resnet18': {
        'fold': 4,  # Val Loss 0.1027
        'train_loss': [0.7124, 0.3678, 0.2534, 0.1987, 0.1823, 0.1654, 0.1412, 0.1345, 0.1098, 0.1123],
        'val_loss': [0.3245, 0.2512, 0.1789, 0.1456, 0.1321, 0.1187, 0.1045, 0.1156, 0.0989, 0.1027]
    },
    'resnet50': {
        'fold': 5,  # Val Loss 0.0610
        'train_loss': [0.6897, 0.2987, 0.2345, 0.2056, 0.1678, 0.1432, 0.1567, 0.1198, 0.1054, 0.0987],
        'val_loss': [0.2987, 0.1876, 0.1654, 0.1098, 0.0976, 0.0854, 0.0923, 0.0678, 0.0712, 0.0610]
    },
    'efficientnet_b0': {
        'fold': 4,  # Val Loss 0.0660
        'train_loss': [0.6618, 0.2826, 0.2029, 0.1692, 0.1607, 0.1430, 0.1310, 0.1202, 0.1025, 0.0964],
        'val_loss': [0.2841, 0.1791, 0.1356, 0.1195, 0.1009, 0.0834, 0.1063, 0.0986, 0.0749, 0.0660]
    }
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

# Noise/Background functions
def add_gaussian_noise(image, mean=0, std=0.1):
    """Add Gaussian noise to an image tensor."""
    noise = torch.normal(mean=mean, std=std, size=image.shape)
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)

def add_random_background(image, background_dir):
    """Add random background to an image."""
    bg_files = [f for f in os.listdir(background_dir) if f.endswith(('.jpg', '.png'))]
    if not bg_files:
        return image
    bg_path = os.path.join(background_dir, random.choice(bg_files))
    bg = Image.open(bg_path).resize(image.size)
    blended = Image.blend(bg, image, alpha=0.5)
    return blended

# Test function
def test_model(model_name, fold, paths, labels, transform, device, categories, noise_type='none'):
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
    background_dir = os.path.join(base_dir, 'backgrounds')  # Folder for background images
    true, pred, probs = [], [], []
    for p, l in zip(paths, labels):
        try:
            img = Image.open(p).convert('RGB')
            # Apply noise or background based on noise_type (50% probability)
            if random.random() < 0.5:
                if noise_type == 'gaussian':
                    # Add Gaussian noise after transform
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    img_tensor = add_gaussian_noise(img_tensor, mean=0, std=0.1)
                elif noise_type == 'background':
                    # Add background before transform
                    img = add_random_background(img, background_dir)
                    img_tensor = transform(img).unsqueeze(0).to(device)
                else:
                    # No noise (default)
                    img_tensor = transform(img).unsqueeze(0).to(device)
            else:
                img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
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
    fold = models_info[model_name]['fold']
    noise_type = 'gaussian'  # Switch to 'background' or 'none'
    print(f"Testing {model_name} Fold {fold} with {noise_type}...")
    true, pred, probs = test_model(model_name, fold, image_paths, labels, transform, device, categories, noise_type=noise_type)
    if true is None or len(true) == 0:
        print(f"No results for {model_name} Fold {fold}.")
        continue
    acc = accuracy_score(true, pred) * 100
    cm = confusion_matrix(true, pred, labels=categories)
    results[model_name] = {
        'fold': fold,
        'accuracy': acc,
        'cm': cm,
        'true': true,
        'pred': pred,
        'probs': probs,
        'noise_type': noise_type
    }
    print(f"{model_name} Fold {fold}: Accuracy {acc:.2f}%")

with open(os.path.join(base_dir, 'docs', 'results.pkl'), 'wb') as f:
    pickle.dump(results, f)

# Plot function
def plot_results(model_name, fold, cm, true, probs, noise_type='none'):
    plt.figure(figsize=(12, 5))
    # 1. Confusion Matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories, cbar=False)
    plt.title(f'{model_name} Fold {fold}\nConfusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # 2. Per-Class Accuracy
    plt.subplot(1, 2, 2)
    cm_diag = cm.diagonal()
    cm_sum = cm.sum(axis=1)
    acc = cm_diag / cm_sum * 100
    plt.bar(categories, acc, color='mediumseagreen')
    plt.title('Per-Class Accuracy')
    plt.xlabel('Category')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 110)
    for i, v in enumerate(acc):
        plt.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
    plt.xticks(rotation=45)
    # Save
    plt.tight_layout()
    out_path = os.path.join(plot_dir, f'{model_name}_{"noise" if noise_type != "none" else "no_noise"}.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    # mAP
    ap_scores = []
    for i, cat in enumerate(categories):
        true_binary = [1 if t == cat else 0 for t in true]
        prob = [p[i] for p in probs]
        ap = average_precision_score(true_binary, prob)
        ap_scores.append(ap)
    mAP = np.mean(ap_scores)
    print(f"{model_name} Fold {fold} mAP: {mAP:.3f}")


# Generate plots
for model_name in results:
    result = results[model_name]
    fold = result['fold']
    cm = result['cm']
    true = result['true']
    probs = result['probs']
    noise_type = results[model_name].get('noise_type', 'none')  # Use stored noise_type or default
    print(f"Generating plot for {model_name} Fold {fold}...")
    plot_results(model_name, fold, cm, true, probs, noise_type=noise_type)