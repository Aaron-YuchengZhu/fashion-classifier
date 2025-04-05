# Fashion Classifier

This is a group project for COMP9444. We trained a fashion classification model using MobileNetV2 to classify items into four categories: `shoes`, `clothing`, `accessories`, and `bags`. The dataset has been cleaned and organized, and the latest model (`best_mobilenet_v3.pth`) achieves high accuracy on the test set.

## Directory Structure
- `data/`: Datasets
  - `LAT/`: Images (`image/`) and labels (`label/LAT.json`)
  - `AAT/`: Images (`image/`) and labels (`label/AAT.json`)
  - `train.json`, `val.json`, `test.json`: Processed datasets (7783 images, 7:2:1 split, no `unknown`)
- `models/`: Trained model weights (e.g., `best_mobilenet_v3.pth`)
- `scripts/`: Core scripts
  - `process_data.py`: Generates `train/val/test.json` from `LAT/AAT.json`
  - `train_mobilenet.py`: Trains MobileNetV2 model
  - `test_mobilenet.py`: Tests the model
- `docs/`: Documentation (e.g., `README.md`, report)

## Dataset
- **Total**: 7783 images (after removing 631 `unknown` labels)
- **Categories**: 
  - `bags`: 7036 (90%+), `clothing`: 343, `shoes`: 316, `accessories`: 88
- **Split**: `train: 5448`, `val: 1557`, `test: 778` (7:2:1)
- **Note**: Data is imbalanced (`bags` dominant); consider weights in training.

Download datasets (too large for GitHub):
- [LAT.zip](https://1drv.ms/u/c/b82bee97bf2cbc97/EXAnIefyHDdKkICyWkXQWCwBC04x8-RmxcRE_g1yXcpuPA?e=6n23kW)
- [AAT.zip](https://1drv.ms/u/c/b82bee97bf2cbc97/EYfLrQV2hM9Em73MpfoOk7kBCJxP_4cHEsul0Vcpteax7A?e=zUAjFb)

## Models
- `best_mobilenet_v3.pth`: Latest MobileNetV2 model (April 2025)
- Older models: `best_mobilenet_v1.pth`, `best_mobilenet_v2.pth`, `best_fashion_classifier_v1.pth` (ResNet18)

Download: [OneDrive](https://1drv.ms/u/c/b82bee97bf2cbc97/EU84bStu4oBPlaNC2a6-HfcBYhJb7jhXu8CAyg-FxQXezw?e=78KB30) (check latest).

## Requirements
- Python 3.10
- PyTorch 2.5.0
- torchvision
- PIL
- scikit-learn

Install:
```bash
pip install torch==2.5.0 torchvision pillow scikit-learn

## Usage
1. Download the dataset and model files from the links above.

2. Extract `LAT.zip` and `AAT.zip` into the Datasets/ directory.

3. Run `process_lat_data.py` to generate the dataset:
```bash
python process_lat_data.py
```

4. Run `train_model.py` to train a new model (or use the pre-trained models):
```bash
python train_model.py
```
This will save the model with a version number (e.g., `best_fashion_classifier_v2.pth` if v1 exists).

5. Run `test_model.py` to test the model on new images:
```bash
# Test on 10 random images from LAT using v1 model
python test_model.py --model best_fashion_classifier_v1.pth --source LAT --num_images 10

# Test on 5 random images from AAT using v1 model
python test_model.py --model best_fashion_classifier_v1.pth --source AAT --num_images 5

# Test on 15 random images from both LAT and AAT using initial model
python test_model.py --model fashion_classifier.pth --source both --num_images 15
```

## Results
Model `best_fashion_classifier_v1.pth` (tag v1.0):
Tested on LAT: 100% accuracy (83/83 images).
Note: AAT testing may skip some images due to invalid gt indices in AAT.json.

## Changelog
2025-03-23 (v1.0):
Updated `train_model.py` to support versioned model saving (e.g., `best_fashion_classifier_v1.pth`).
Updated `test_model.py` to support random testing from LAT and AAT datasets with command-line arguments (--model, --source, --num_images).
Added `best_fashion_classifier_v1.pth` with 100% accuracy on LAT (83/83 images).
Ignored `.idea/`, `.pth`, `LAT/`, and `AAT/` in `.gitignore`.
Previous Versions:
Initial version included `fashion_classifier.pth`, `process_lat_data.py`, and `fashion_dataset.json`.
