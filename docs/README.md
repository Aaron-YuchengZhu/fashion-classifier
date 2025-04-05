# Fashion Classifier

This is a group project for COMP9444. We trained a fashion classification model using MobileNetV2 to classify items into four categories: `shoes`, `clothing`, `accessories`, and `bags`. The dataset has been cleaned and organized, and the latest model (`best_mobilenet_v3.pth`) achieves high accuracy on the test set. This repository reflects the v2.0 state (April 2025).

## Directory Structure

```bash
Datasets/
├── data/
│   ├── LAT/
│   │   ├── image/
│   │   └── label/
│   ├── AAT/
│   │   ├── image/
│   │   └── label/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── models/
│   └── best_mobilenet_v3.pth
├── scripts/
│   ├── process_data.py
│   ├── train_mobilenet.py
│   └── test_mobilenet.py
├── docs/
│   └── README.md
└── .gitignore
```


- `data/`: Datasets (images and processed JSON files)
- `models/`: Trained model weights
- `scripts/`: Core scripts for processing, training, and testing
- `docs/`: Documentation

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
- `best_mobilenet_v3.pth`: Latest MobileNetV2 model (April 2025, v2.0)
- Older models available on cloud: `best_mobilenet_v1.pth`, `best_mobilenet_v2.pth`, `best_fashion_classifier_v1.pth` (ResNet18)

Download models:
- [v2 Models Folder](https://1drv.ms/f/c/b82bee97bf2cbc97/Etsxa7VIRvJFl2Y6D77RMfIBhxIxXXAJIkTcYa4hTAM68Q?e=fDKfx2) (contains `best_mobilenet_v1.pth`, `v2.pth`, `v3.pth`)
- [v1 Models](https://1drv.ms/f/c/b82bee97bf2cbc97/EeR7b_PygH5IlvW8uNiT5RUBr-WmOQ8VGMoUGeXERmHiIQ) (older ResNet18 models)

## Requirements
- Python 3.10
- PyTorch 2.5.0
- torchvision
- PIL
- scikit-learn

Install:
```bash
pip install torch==2.5.0 torchvision pillow scikit-learn
```

## Usage

1. Download and Extract: Get LAT.zip, AAT.zip, and best_mobilenet_v3.pth from the links above, extract datasets to data/.
2. Process Data: Generate train/val/test.json:
   ```bash
   python scripts/process_data.py
   ```
3. Train Model: Train MobileNetV2 (saves to models/best_mobilenet_v3.pth):
   ```bash
   python scripts/train_mobilenet.py
   ```
4. Test Model: Test on test.json:
   ```bash
   # Test 50 random images from both LAT and AAT
   python scripts/test_mobilenet.py --model models/best_mobilenet_v3.pth --source both --num_images 50
   ```

## Results

- **v2 (MobileNetV2):** Test Accuracy 91.84% (45/49, 50 sampled, 1 skipped), Val Loss 0.2937.
- Older results:
  - best_mobilenet_v2.pth: ~90% (45/50), Val Loss 0.3303.
  - best_fashion_classifier_v1.pth (ResNet18): 100% on LAT (83/83), Val Loss 0.2709.

## Changelog

- 2025-04-05 (v2.0):
  - Reset repository to v2: Cleaned dataset (7783 images, no unknown), MobileNetV2, new structure (data/, models/, scripts/, docs/).
  - Added best_mobilenet_v3.pth (91.84% accuracy).
  - Updated scripts and README.
- 2025-03-23 (v1.0):
  - Initial ResNet18 model (best_fashion_classifier_v1.pth), 100% on LAT.

---

