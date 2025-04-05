import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import json
import os
import sys

class FashionDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        self.label_map = {'shoes': 0, 'clothing': 1, 'accessories': 2, 'bags': 3}
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Datasets/
        self.valid_data = []
        for item in self.data:
            image_name = item["filename"]
            label = item["label"]
            if label not in self.label_map:
                print(f"Warning: Label {label} not in label_map, skipping {image_name}")
                continue
            image_dir = os.path.join(self.base_dir, 'data/LAT/image/' if 'P' in image_name else 'data/AAT/image/')  # 修复路径
            image_path = os.path.join(image_dir, image_name)
            if os.path.exists(image_path):
                self.valid_data.append((image_path, label))
            else:
                print(f"Warning: Image {image_path} not found, skipping.")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        image_path, label = self.valid_data[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            label = 0
        if self.transform:
            image = self.transform(image)
        label = self.label_map[label]
        return image, label

# Data preprocessing
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)  # Datasets/
train_dataset = FashionDataset(os.path.join(base_dir, 'data', 'train.json'), transform=transform)
val_dataset = FashionDataset(os.path.join(base_dir, 'data', 'val.json'), transform=transform)
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load MobileNetV2
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 4)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Save path
model_path = os.path.join(base_dir, 'models', 'best_mobilenet_v3.pth')
print(f"Model will be saved as: {model_path}")

# Training loop
num_epochs = 20
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss = val_loss / len(val_loader)
    print(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}')

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)
        print(f"Best model saved with Val Loss: {best_val_loss:.4f}")
        counter = 0
    else:
        counter += 1
    if counter >= patience:
        print("Early stopping")
        break

model.load_state_dict(torch.load(model_path, weights_only=True))
print(f"Best model loaded from {model_path}")
