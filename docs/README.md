# Fashion Classifier

This is a group project for COMP9444 (25T1), developing a deep learning-based fashion image classifier for four categories: Accessories, Bags, Clothing, and Shoes. Using a balanced dataset of 8000 images, we trained and evaluated four CNN models (MobileNetV2, ResNet18, ResNet50, EfficientNet-B0), achieving up to 99.8% accuracy (no noise) and 93.0% (Gaussian noise, std=0.1). This repository reflects v3.0 (April 2025).


## Directory Structure

```bash
Datasets/
├── ATT_augmented/
│   ├── Accessories/ (2000 images)
│   ├── Bags/ (2000 images)
│   ├── Clothings/ (2000 images)
│   ├── Shoes/ (2000 images)
│   └── dataset_info.json
├── data/
│   ├── AAT/
│   ├── LAT/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── models/
│   ├── mobilenet_v2_fold[1-5].pth
│   ├── resnet18_fold[1-5].pth
│   ├── resnet50_fold[1-5].pth
│   ├── efficientnet_b0_fold[1-5].pth
│   └── best_*.pth
├── notebooks/
│   └── fashion_classifier.ipynb
├── plot/
│   ├── mobilenet_v2_[no_noise|noise].png
│   ├── resnet18_[no_noise|noise].png
│   ├── resnet50_[no_noise|noise].png
│   ├── efficientnet_b0_[no_noise|noise].png
│   ├── comparison_*.png
│   └── all_models_*.png
├── scripts/
│   ├── process_data.py
│   ├── train_*.py
│   ├── test_*.py
│   ├── compare_models.py
│   └── generate_comparison_plots.py
└── docs/
    └── results.pkl
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

Download:
- [Models](https://1drv.ms/u/c/b82bee97bf2cbc97/ETXp2gUGHzxGgVYAU1QczBYBIyBL_Nzcpjr--Ljv4xF3uA)

## Requirements
- Python 3.10
- PyTorch 2.5.0
- torchvision, PIL, scikit-learn

```bash
pip install torch==2.5.0 torchvision pillow scikit-learn
```

## Results

- No Noise: All models near-perfect (98.8%-99.8%, mAP 1.000).
- Gaussian Noise (std=0.1): EfficientNet-B0 leads (93.0%, mAP 0.990), ResNet50 weakest (89.0%).
- Plots: Confusion matrices, PR curves, accuracy/mAP bars, heatmaps in plot/.
- Notebook: fashion_classifier.ipynb details data processing, training, evaluation, and analysis.

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

