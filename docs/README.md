# Fashion Classifier

This is a group project for COMP9444. We trained fashion classification models to classify items into four categories: `shoes`, `clothing`, `accessories`, and `bags`. The dataset has been cleaned and organized, and the latest models achieve high accuracy on the test set. This repository reflects the v2.0 state (April 2025).

## Directory Structure

```bash
Datasets
├── ATT_augmented
│   ├── Accessories
│   ├── Bags
│   ├── Clothings
│   └── Shoes
├── models
│   ├── mobilenet_v2_fold1.pth
│   ├── mobilenet_v2_fold2.pth
│   ├── mobilenet_v2_fold3.pth
│   ├── mobilenet_v2_fold4.pth
│   ├── mobilenet_v2_fold5.pth
│   ├── best_efficientnet.pth
│   ├── best_mobilenet_v3.pth
│   ├── best_resnet18.pth
│   └── best_resnet50.pth
├── notebooks
│   └── fashion_classifier.ipynb
└── scripts
    ├── process_data.py
    ├── test_efficientnet.py
    ├── test_mobilenet_stratified.py
    ├── test_resnet18.py
    ├── test_resnet50.py
    ├── train_efficientnet.py
    ├── train_mobilenet_5fold.py
    ├── train_resnet18.py
    └── train_resnet50.py
```



- `ATT_augmented/`: Dataset (8000 images)
- `models/`: Trained weights (v3 + older)
- `scripts/`: Training/testing scripts

## Dataset
- **Total**: 8000 images
- **Categories**: `bags`, `clothing`, `shoes`, `accessories` (2000 each)
- **Note**: Balanced, no JSON required.

Download:
- [ATT_augmented.zip](https://1drv.ms/u/c/b82bee97bf2cbc97/EVEzFa9TF5ZDtEpNJ7KGjnUBjLmcYQPG8WqTiZ_DRLwhPw?e=Ys7Rtz)

## Models
- **v3.0**:
  - **MobileNetV2**: 99.6% accuracy (5-fold, `mobilenet_v2_fold1.pth` to `fold5.pth`), Avg Val Loss 0.0686.
- **v2.0**:
  - **ResNet50**: 98.0% accuracy, Val Loss 0.3151 (`best_resnet50.pth`).
  - **EfficientNet-B0**: 88.0% accuracy, Val Loss 0.3243 (`best_efficientnet.pth`).
  - **MobileNetV3**: 91.84% accuracy, Val Loss 0.2937 (`best_mobilenet_v3.pth`).
- **v1.0**:
  - **ResNet18**: 91.3% accuracy, Val Loss 0.3046 (`best_resnet18.pth`).

Download:
- [Models](https://1drv.ms/u/c/b82bee97bf2cbc97/ETXp2gUGHzxGgVYAU1QczBYBIyBL_Nzcpjr--Ljv4xF3uA)

## Requirements
- Python 3.10
- PyTorch 2.5.0
- torchvision, PIL, scikit-learn

```bash
pip install torch==2.5.0 torchvision pillow scikit-learn
```

## Usage

1. Download and Extract: Get LAT.zip, AAT.zip, and best_mobilenet_v3.pth from the links above, extract datasets to data/.
2. Process Data: Generate train/val/test.json:
   ```bash
   python scripts/process_data.py
   ```

3. **Train Models:**
   - Train MobileNetV2 (saves to models/best_mobilenet_v3.pth):
       ```bash
       python scripts/train_mobilenet.py
       ```
   - Train ResNet18 (saves to models/best_resnet18.pth):
     ```bash
     python scripts/train_resnet18.py
     ```
     
   - Train ResNet50 (saves to `models/best_resnet50.pth`):
     ```bash
     python scripts/train_resnet50.py
     ```

   - Train EfficientNet-B0 (saves to models/best_efficientnet.pth):
     ```bash 
     python scripts/train_efficientnet.py
     ```
     
4. **Test Model:** 
    - Test on test.json:
    ```bash
    # Test 50 random images from both LAT and AAT
    python scripts/test_mobilenet.py --model models/best_mobilenet_v3.pth --source both --num_images 50
    ```
    - Test ResNet18:
    ```bash
    python scripts/test_resnet.py --model models/best_resnet18.pth --source both --num_images 50
    ```
   
    - Test ResNet50:
     ```bash
     python scripts/test_resnet50.py --model models/best_resnet50.pth --source both --sample 50
     ```

    - Test EfficientNet-B0:
     ```bash
     python scripts/test_efficientnet.py --model models/best_efficientnet.pth --source both --num_images 50
     ```

## Results

- **MobileNetV2**: 99.6% accuracy (100/100, 5 runs), Avg Val Loss 0.0686 
- **MobileNetV2**: Test Accuracy 91.84% (45/49), Val Loss 0.2937.
- **ResNet18**: Test Accuracy 91.30% (42/46), Val Loss 0.3046.
- **EfficientNet-B0**: Test Accuracy 88.00% (44/50), Val Loss 0.3243.
- Older results:
  - `best_mobilenet_v2.pth`: ~90% (45/50), Val Loss 0.3303.
  - `best_fashion_classifier_v1.pth` (ResNet18): 100% on LAT (83/83), Val Loss 0.2709.

## Changelog

- 2025-04-13:
  - MobileNetV2 5-fold, 99.6% accuracy (train_mobilenet_5fold.py, test_mobilenet_stratified.py).
  - Update notebooks floder, submit fashion_classifier.ipynb
- 2025-04-07:
  - Added ResNet50 scripts, 98.00% accuracy (note: biased towards bags, to be optimized)
- 2025-04-06:
  - Added ResNet18 scripts, 91.30% accuracy.
  - Added EfficientNet-B0 scripts, 88.00% accuracy, fixed test script paths and MPS support.
- 2025-04-05 (v2.0):
  - Reset repository to v2: Cleaned dataset (7783 images, no unknown), MobileNetV2, new structure (`data/`, `models/`, `scripts/`, `docs/`).
  - Added `best_mobilenet_v3.pth` (91.84% accuracy).
  - Updated scripts and README.
- 2025-03-23 (v1.0):
  - Initial ResNet18 model (`best_fashion_classifier_v1.pth`), 100% on LAT.

---

