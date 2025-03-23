import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import json
import os

# 自定义数据集类
class FashionDataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.transform = transform
        self.label_map = {'shoes': 0, 'clothing': 1, 'accessories': 2, 'bags': 3}
        # 过滤掉图片不存在的样本
        self.valid_data = []
        for image_name, label in self.data:
            image_path = os.path.join(self.image_dir, image_name + '.jpg')
            if os.path.exists(image_path):
                self.valid_data.append((image_name, label))

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        image_name, label = self.valid_data[idx]
        image_path = os.path.join(self.image_dir, image_name + '.jpg')
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.label_map[label]
        return image, label

# 数据预处理（加了数据增强）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),      # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
image_dir = '/Users/i/Downloads/Datasets/LAT/image/'
dataset = FashionDataset('/Users/i/Downloads/Datasets/fashion_dataset.json', image_dir, transform=transform)
print(f"有效样本数量：{len(dataset)}")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 加载预训练的 ResNet18
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4 个类别

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}')

# 保存模型
torch.save(model.state_dict(), '/Users/i/Downloads/Datasets/fashion_classifier.pth')
print("模型已保存到 /Users/i/Downloads/Datasets/fashion_classifier.pth")