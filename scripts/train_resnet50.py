import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
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
        if len(image_paths) != len(labels):
            print(f"Error: image_paths ({len(image_paths)}) and labels ({len(labels)}) mismatch!")
            exit(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            label = 'Accessories'
        if self.transform:
            image = self.transform(image)
        label_id = self.label_map[label]
        return image, label_id

# 数据预处理
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载所有图片
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data', 'ATT_augmented')
categories = ['Accessories', 'Bags', 'Clothings', 'Shoes']
image_paths = []
labels = []

for category in categories:
    folder_path = os.path.join(data_dir, category)
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} not found.")
        exit(1)
    print(f"Scanning {folder_path}...")
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if os.path.isfile(img_path) and img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                Image.open(img_path).verify()
                image_paths.append(img_path)
                labels.append(category)
            except Exception as e:
                print(f"Invalid image {img_path}: {e}")

print(f"Total samples: {len(image_paths)}")
print(f"Label distribution: {Counter(labels)}")
if len(image_paths) == 0:
    print("Error: No images found.")
    exit(1)
if len(image_paths) != len(labels):
    print(f"Error: image_paths ({len(image_paths)}) and labels ({len(labels)}) mismatch!")
    exit(1)

# 五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 检查 KFold 索引
print(f"Checking KFold indices...")
for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
    print(f"Fold {fold + 1}: Train indices: {len(train_idx)}, Val indices: {len(val_idx)}")
    if max(train_idx) >= len(image_paths) or max(val_idx) >= len(image_paths):
        print(
            f"Error: Invalid indices in Fold {fold + 1}! Max train_idx: {max(train_idx)}, Max val_idx: {max(val_idx)}, Total samples: {len(image_paths)}")
        exit(1)
    if min(train_idx) < 0 or min(val_idx) < 0:
        print(f"Error: Negative indices in Fold {fold + 1}!")
        exit(1)

# 训练和验证
fold_results = []
for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
    print(f"\nFold {fold + 1}/5")
    print(f"Train indices: {len(train_idx)}, Val indices: {len(val_idx)}")
    # 再次确认 labels 长度
    if len(labels) != len(image_paths):
        print(
            f"Error: labels length ({len(labels)}) != image_paths length ({len(image_paths)}) before Fold {fold + 1}!")
        exit(1)
    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_dataset = FashionDataset(train_paths, train_labels, transform=transform)
    val_dataset = FashionDataset(val_paths, val_labels, transform=transform)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 模型
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model = model.to(device)

    # 损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 保存路径
    model_dir = os.path.join(base_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f'resnet50_fold{fold + 1}.pth')

    # 训练循环
    num_epochs = 20
    best_val_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels_batch in train_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}, Val Loss: {val_loss:.4f}')

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

    fold_results.append(best_val_loss)

# 输出平均结果
avg_val_loss = sum(fold_results) / len(fold_results)
print(f"\nAverage Val Loss across 5 folds: {avg_val_loss:.4f}") 