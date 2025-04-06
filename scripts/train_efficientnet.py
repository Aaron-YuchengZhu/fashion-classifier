import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import json
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import time

# Dataset class
class FashionDataset(Dataset):
    def __init__(self, json_data, base_dir, transform=None):
        self.data = json_data
        self.base_dir = base_dir
        self.transform = transform
        self.label_map = {'shoes': 0, 'clothing': 1, 'accessories': 2, 'bags': 3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]['filename']
        label_str = self.data[idx]['label']
        label = self.label_map[label_str]  # 转换为整数
        
        # 构建图像路径
        subdir = 'LAT/image/' if 'P' in img_name else 'AAT/image/'
        image_path = os.path.join(self.base_dir, 'data', subdir, img_name)
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label  # 返回图像张量和整数标签

# Command-line arguments
import argparse
parser = argparse.ArgumentParser(description='Train EfficientNet on fashion images.')
parser.add_argument('--model', type=str, default='efficientnet-b0', help='EfficientNet model variant')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
args = parser.parse_args()

# Prepare device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 正确：指向项目根目录

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)  # 项目根目录

# json_path = os.path.join(base_dir, 'data', 'train.json')
json_path = os.path.join(base_dir, 'data', 'train.json')  # 正确路径

print(f"Looking for train.json at: {json_path}")

with open(json_path, 'r') as f:
    train_data = json.load(f)

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prepare DataLoader
train_dataset = FashionDataset(train_data, base_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# Load EfficientNet model
model = EfficientNet.from_pretrained(args.model)
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, 4)  # Change the final layer to match 4 categories
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Training loop
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_time = time.time() - start_time
    print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s")
    
    # Save the model after each epoch
    torch.save(model.state_dict(), f"efficientnet_epoch_{epoch+1}.pth")

print("Training finished!")
