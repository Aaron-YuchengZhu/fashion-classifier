import json
import os
import random

# Get the script directory and base directory
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)  # Parent directory: Datasets/

# Load JSON data
with open(os.path.join(base_dir, 'data', 'LAT', 'label', 'LAT.json'), 'r') as f:
    lat_data = json.load(f)

with open(os.path.join(base_dir, 'data', 'AAT', 'label', 'AAT.json'), 'r') as f:
    aat_data = json.load(f)

# Define category classification function
def get_category(item_name):
    if item_name.startswith('Shoes'):
        return 'shoes'
    elif item_name.startswith(('Pants', 'Top', 'Outwear', 'Dress', 'Skirt')):
        return 'clothing'
    elif item_name.startswith('Bags'):
        return 'bags'
    elif item_name.startswith(('Earing', 'Watches', 'Bracelet', 'Hat', 'Sunglasses', 'Neckline')):
        return 'accessories'
    else:
        return 'unknown'

# Extract existing labels
label_dict = {}

# Process LAT labels
for entry in lat_data:
    for item in entry['question'] + entry['answers']:
        prefix, img_id = item.split('_')
        filename = f"{img_id}.jpg"
        label_dict[filename] = get_category(prefix)

# Process AAT labels
for entry in aat_data:
    for item in entry['question'] + entry['answers']:
        prefix, rest = item.split('/')
        img_id = rest.split('_')[-1]
        filename = f"{img_id}.jpg"
        label_dict[filename] = get_category(prefix)

# Scan all images from LAT and AAT
all_images = []
for base in ['LAT', 'AAT']:
    img_dir = os.path.join(base_dir, 'data', base, 'image')
    all_images.extend([
        fname for fname in os.listdir(img_dir) if fname.lower().endswith('.jpg')
    ])

# Build dataset (ensure every image has a label)
dataset = []
missing_images = []
for img_name in all_images:
    label = label_dict.get(img_name)
    if not label:
        if img_name.startswith('P'):
            label = 'bags'
        elif 'Q' in img_name:
            label = 'clothing'
        elif 'A' in img_name:
            label = 'shoes'
        elif 'W' in img_name:
            label = 'accessories'
        else:
            label = 'unknown'
    # Skip images with 'unknown' label
    if label != 'unknown':
        dataset.append({"filename": img_name, "label": label})

# Remove duplicates
unique_dataset = [dict(t) for t in {tuple(d.items()) for d in dataset}]

# Verify image existence
for entry in unique_dataset:
    base_dir_entry = 'LAT' if entry['filename'].startswith('P') else 'AAT'
    path = os.path.join(base_dir, 'data', base_dir_entry, 'image', entry['filename'])
    if not os.path.exists(path):
        missing_images.append(entry['filename'])

if missing_images:
    print("Missing images:", missing_images)
else:
    print("✅ All image files exist!")

# Print category distribution
from collections import Counter
print("Category distribution:", Counter([d['label'] for d in unique_dataset]))

# Split dataset (7:2:1)
random.seed(42)
random.shuffle(unique_dataset)

total = len(unique_dataset)
train_idx = int(0.7 * total)
val_idx = int(0.9 * total)

train_set = unique_dataset[:train_idx]
val_set = unique_dataset[train_idx:val_idx]
test_set = unique_dataset[val_idx:]

# Save files to data/
data_dir = os.path.join(base_dir, 'data')
with open(os.path.join(data_dir, 'train.json'), 'w') as f:
    json.dump(train_set, f, indent=4)
with open(os.path.join(data_dir, 'val.json'), 'w') as f:
    json.dump(val_set, f, indent=4)
with open(os.path.join(data_dir, 'test.json'), 'w') as f:
    json.dump(test_set, f, indent=4)

print(f"Training set: {len(train_set)}, Validation set: {len(val_set)}, Test set: {len(test_set)}")
print("Data successfully saved to data/train.json, data/val.json, and data/test.json.")
print("Data successfully saved to data/train.json, data/val.json, and data/test.json.")