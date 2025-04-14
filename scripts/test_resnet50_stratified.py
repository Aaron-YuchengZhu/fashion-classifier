import torch
from torchvision import models, transforms
from PIL import Image
import os
import argparse
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import random

# 命令行参数
parser = argparse.ArgumentParser(description='Test ResNet50 on ATT_augmented.')
parser.add_argument('--model', type=str, default='resnet50_fold1.pth')
parser.add_argument('--num_images', type=int, default=100)
parser.add_argument('--runs', type=int, default=5, help='Number of test runs')
args = parser.parse_args()

# 加载模型
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
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

# 类别映射
label_map = {0: 'Accessories', 1: 'Bags', 2: 'Clothings', 3: 'Shoes'}

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载所有图片
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

# 多轮测试
for run in range(args.runs):
    print(f"\nRun {run + 1}/{args.runs}")
    seed = random.randint(0, 1000)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.num_images, random_state=seed)
    for _, test_idx in sss.split(image_paths, labels):
        test_paths = [image_paths[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

    # 测试
    correct = 0
    total = 0
    true_labels = []
    pred_labels = []
    errors = []
    for img_path, true_label in zip(test_paths, test_labels):
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found, skipping.")
            continue
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted_label = label_map[predicted.item()]

        true_labels.append(true_label)
        pred_labels.append(predicted_label)
        if predicted_label == true_label:
            correct += 1
        else:
            errors.append(f"Image: {os.path.basename(img_path)}, Predicted: {predicted_label}, True: {true_label}")
        total += 1

    # 输出结果
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Run {run + 1} Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"True label distribution: {Counter(true_labels)}")
    print(f"Predicted label distribution: {Counter(pred_labels)}")
    if errors:
        print("Errors:")
        for error in errors:
            print(error) 