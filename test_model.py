import torch
from torchvision import models, transforms
from PIL import Image
import os

# 加载模型
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)
model.load_state_dict(torch.load('/Users/i/Downloads/Datasets/fashion_classifier.pth', weights_only=True))
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

# 测试多张图片
image_dir = '/Users/i/Downloads/Datasets/LAT/image/'
test_images = [
    'P00416047.jpg',  # 之前测过的，应该是 clothing
    'P00435036.jpg',  # 随便挑几张
    'P00440168.jpg',
    'P00447042.jpg',
    'P00462477.jpg'
]

# 手动标注的真实标签（根据 LAT.json 里的单品名）
ground_truth = {
    'P00416047': 'clothing',  # Top_P00416047
    'P00435036': 'shoes',     # Shoes_P00435036
    'P00440168': 'clothing',  # Top_P00440168
    'P00447042': 'shoes',     # Shoes_P00447042
    'P00462477': 'clothing'   # Dress_P00462477
}

correct = 0
total = len(test_images)

for image_name in test_images:
    image_path = os.path.join(image_dir, image_name)
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_label = label_map[predicted.item()]

    # 真实标签
    true_label = ground_truth[image_name.split('.')[0]]

    print(f"图片: {image_name}, 预测: {predicted_label}, 真实: {true_label}")
    if predicted_label == true_label:
        correct += 1

accuracy = 100 * correct / total
print(f"测试准确率: {accuracy:.2f}% ({correct}/{total})")