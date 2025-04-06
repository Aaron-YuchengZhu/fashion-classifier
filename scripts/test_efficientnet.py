import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import os
import sys
import argparse
import json
import random
import time

# 强制使用 CPU
device = torch.device("cpu")  # 使用 CPU

# 命令行参数
parser = argparse.ArgumentParser(description='Test EfficientNet on fashion images.')
parser.add_argument('--model', type=str, default='scripts/efficientnet_epoch_10.pth', help='Path to the model weights文件')  # 修改默认路径
parser.add_argument('--source', type=str, choices=['LAT', 'AAT', 'both'], default='both', help='Source dataset to test on')
parser.add_argument('--num_images', type=int, default=10, help='Number of images to test')
args = parser.parse_args()

# 修复路径定义
current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts 目录
base_dir = os.path.dirname(current_dir)                   # 项目根目录

# 加载模型
print(f"Using device: {device}")
model = EfficientNet.from_pretrained('efficientnet-b0')
num_ftrs = model._fc.in_features
model._fc = torch.nn.Linear(num_ftrs, 4)  # 调整输出层，适应4个类别

# 修复模型路径（动态处理绝对/相对路径）
if not os.path.isabs(args.model):
    model_path = os.path.join(current_dir, args.model)  # 基于 scripts 目录的路径
else:
    model_path = args.model

print(f"Loading model from: {model_path}")  # 调试输出

if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found.")
    sys.exit(1)

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()


# 类别映射
label_map = {0: 'shoes', 1: 'clothing', 2: 'accessories', 3: 'bags'}

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载测试数据（修复路径）
json_path = os.path.join(base_dir, 'data', 'test.json')
print(f"Loading test data from: {json_path}")
with open(json_path, 'r') as f:
    data = json.load(f)

# 按来源过滤数据
lat_data = [item for item in data if 'P' in item["filename"]]
aat_data = [item for item in data if 'A' in item["filename"]]
if args.source == 'LAT':
    test_pool = lat_data
elif args.source == 'AAT':
    test_pool = aat_data
else:  # both
    test_pool = data

test_images = random.sample(test_pool, min(args.num_images, len(test_pool)))

# 测试
correct = 0
total = 0
for item in test_images:
    image_name = item["filename"]
    true_label = item["label"]
    
    # 修复图像路径
    subdir = 'LAT/image' if 'P' in image_name else 'AAT/image'
    image_dir = os.path.join(base_dir, 'data', subdir)
    image_path = os.path.join(image_dir, image_name)
    
    print(f"Checking image path: {image_path}")  # 调试输出
    
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