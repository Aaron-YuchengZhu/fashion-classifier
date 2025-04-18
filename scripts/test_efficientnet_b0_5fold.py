import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import os
import argparse
import random
import csv
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

# 命令行参数
parser = argparse.ArgumentParser(description='Test EfficientNet-B0 on ATT_augmented with random sampling.')
parser.add_argument('--model', type=str, default='efficientnet_b0_fold1.pth')
parser.add_argument('--num_images', type=int, default=100)
parser.add_argument('--runs', type=int, default=5, help='Number of test runs')
args = parser.parse_args()

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = EfficientNet.from_name('efficientnet-b0', num_classes=4)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'models', args.model)
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found.")
    exit(1)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 类别映射
label_map = {0: 'Accessories', 1: 'Bags', 2: 'Clothings', 3: 'Shoes'}
reverse_map = {v: k for k, v in label_map.items()}

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载所有图像路径
data_dir = os.path.join(base_dir, 'ATT_augmented')
categories = list(label_map.values())
image_paths, labels = [], []

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

# 创建输出目录
output_dir = os.path.join(base_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

# 多轮测试
for run in range(args.runs):
    print(f"\nRun {run + 1}/{args.runs}")
    seed = random.randint(0, 1000)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.num_images, random_state=seed)
    for _, test_idx in sss.split(image_paths, labels):
        test_paths = [image_paths[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

    correct = 0
    total = 0
    true_labels = []
    pred_labels = []
    errors = []
    filenames = []

    for img_path, true_label in zip(test_paths, test_labels):
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found, skipping.")
            continue
        try:
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                predicted_label = label_map[predicted.item()]

            true_labels.append(true_label)
            pred_labels.append(predicted_label)
            filenames.append(os.path.basename(img_path))
            if predicted_label == true_label:
                correct += 1
            else:
                errors.append(f"Image: {os.path.basename(img_path)}, Predicted: {predicted_label}, True: {true_label}")
            total += 1
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue

    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Run {run + 1} Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"True label distribution: {Counter(true_labels)}")
    print(f"Predicted label distribution: {Counter(pred_labels)}")
    if errors:
        print("Errors:")
        for error in errors:
            print(error)

    # 保存为 CSV
    model_name = os.path.splitext(args.model)[0]
    output_file = os.path.join(output_dir, f"{model_name}_run{run+1}.csv")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'true_label', 'predicted_label'])
        for fname, t, p in zip(filenames, true_labels, pred_labels):
            writer.writerow([fname, t, p])
    print(f"✅ Saved prediction log to {output_file}")

    # 保存 y_true 和 y_pred 为 pkl
    result_pkl = os.path.join(output_dir, f"{model_name}_run{run+1}_results.pkl")
    with open(result_pkl, 'wb') as f:
        pickle.dump({'y_true': true_labels, 'y_pred': pred_labels}, f)
    print(f"✅ Saved raw result data to {result_pkl}")