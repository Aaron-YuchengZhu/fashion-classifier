import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from PIL import Image
import json
import os
import sys
from sklearn.model_selection import train_test_split

# Custom dataset class for fashion classification
class FashionDataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None):
        # Load JSON data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.transform = transform
        self.label_map = {'shoes': 0, 'clothing': 1, 'accessories': 2, 'bags': 3}
        # Filter out samples where the image doesn't exist
        self.valid_data = []
        for image_name, label in self.data:
            image_path = os.path.join(self.image_dir, image_name + '.jpg')
            if os.path.exists(image_path):
                self.valid_data.append((image_name, label))
            else:
                print(f"Warning: Image {image_path} not found, skipping.")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        image_name, label = self.valid_data[idx]
        image_path = os.path.join(self.image_dir, image_name + '.jpg')
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image and default label to avoid crashing
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            label = 0
        if self.transform:
            image = self.transform(image)
        label = self.label_map[label]
        return image, label

# Data preprocessing with augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dynamically get paths
current_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(current_dir, 'LAT/image/')
json_path = os.path.join(current_dir, 'fashion_dataset.json')
dataset = FashionDataset(json_path, image_dir, transform=transform)
print(f"Number of valid samples: {len(dataset)}")

# Split dataset into training and validation sets
train_indices, val_indices = train_test_split(
    list(range(len(dataset))), test_size=0.2, random_state=42
)
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4 classes

# Device selection based on platform
if sys.platform == 'darwin':  # macOS
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
elif sys.platform == 'win32':  # Windows
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:  # Linux or others
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Determine the model file path with version number
version = 1
model_path = os.path.join(current_dir, f'best_fashion_classifier_v{version}.pth')
while os.path.exists(model_path):
    version += 1
    model_path = os.path.join(current_dir, f'best_fashion_classifier_v{version}.pth')
print(f"Model will be saved as: {model_path}")

# Train the model
num_epochs = 20
best_val_loss = float('inf')
patience = 3
counter = 0

for epoch in range(num_epochs):
    # Training phase
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

    # Validation phase
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

    # Adjust learning rate
    scheduler.step(val_loss)

    # Early stopping and save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)
        print(f"Best model saved with Val Loss: {best_val_loss:.4f} at {model_path}")
        counter = 0
    else:
        counter += 1
    if counter >= patience:
        print("Early stopping")
        break

# Load the best model
model.load_state_dict(torch.load(model_path, weights_only=True))
print(f"Best model loaded from {model_path}")