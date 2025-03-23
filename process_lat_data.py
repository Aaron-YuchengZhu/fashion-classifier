import json
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to LAT.json
lat_json_path = os.path.join(script_dir, 'LAT/label/LAT.json')

# Read LAT.json
with open(lat_json_path, 'r') as f:
    lat_data = json.load(f)

# Extract all items (from both "question" and "answers")
dataset = []  # Format: [(image_filename, category), ...]

# Define categorization rules
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
        print(f"Unknown item: {item_name}")
        return 'unknown'

# Process each entry in LAT.json
for entry in lat_data:
    # Extract items from "question"
    for question_item in entry['question']:
        image_name = question_item.split('_')[1]  # e.g., Pants_P00462138 -> P00462138
        label = get_category(question_item)
        dataset.append((image_name, label))

    # Extract items from "answers"
    for answer_item in entry['answers']:
        image_name = answer_item.split('_')[1]  # e.g., Top_P00440101 -> P00440101
        label = get_category(answer_item)
        dataset.append((image_name, label))

# Remove duplicates (some items may appear multiple times)
dataset = list(set(dataset))

# Count the distribution of the four main categories
category_counts = {'shoes': 0, 'clothing': 0, 'accessories': 0, 'bags': 0, 'unknown': 0}
for _, label in dataset:
    category_counts[label] = category_counts.get(label, 0) + 1

print("Category distribution:", category_counts)

# Check if image files exist
image_dir = os.path.join(script_dir, 'LAT/image/')
missing_images = []
for image_name, label in dataset:
    image_path = os.path.join(image_dir, image_name + '.jpg')  # e.g., P00462138.jpg
    if not os.path.exists(image_path):
        missing_images.append(image_name)

if missing_images:
    print("The following image files are missing:", missing_images)
else:
    print("All image files exist!")

# Save the new dataset (image paths and labels)
output_path = os.path.join(script_dir, 'fashion_dataset.json')
with open(output_path, 'w') as f:
    json.dump(dataset, f, indent=4)

print(f"Dataset has been saved to {output_path}")