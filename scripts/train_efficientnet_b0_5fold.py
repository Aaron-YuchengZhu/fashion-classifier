import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import os
from sklearn.model_selection import KFold
from collections import Counter

class FashionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.label_map = {'Accessories': 0, 'Bags': 1, 'Clothings': 2, 'Shoes': 3}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            label = 'Accessories'
        if self.transform:
            image = self.transform(image)
        label_id = self.label_map[label]
        return image, label_id

# EfficientNet-B0 → 输入尺寸为 224x224
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载图像路径和标签
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'ATT_augmented')
categories = ['Accessories', 'Bags', 'Clothings', 'Shoes']
image_paths, labels = [], []

for cat in categories:
    folder = os.path.join(data_dir, cat)
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, fname)
            try:
                Image.open(path).verify()
                image_paths.append(path)
                labels.append(cat)
            except:
                print(f"Skipping invalid image: {path}")

print(f"Total images: {len(image_paths)}")
print(f"Label distribution: {Counter(labels)}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

fold_results = []
for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
    print(f"\nFold {fold+1}/5")

    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_dataset = FashionDataset(train_paths, train_labels, transform)
    val_dataset = FashionDataset(val_paths, val_labels, transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=4)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'efficientnet_b0_fold{fold+1}.pth')

    best_val_loss = float('inf')
    patience = 3
    counter = 0

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model for Fold {fold+1}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    fold_results.append(best_val_loss)

print(f"\nAverage Val Loss across 5 folds: {sum(fold_results)/len(fold_results):.4f}")