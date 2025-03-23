# Fashion Classifier

This is a group project for COMP9444. I trained a fashion classification model using ResNet18 to classify items into four categories: shoes, clothing, accessories, and bags. The model achieved high accuracy on the test set.

## Files
- `process_lat_data.py`: Script to process the dataset and generate `fashion_dataset.json`.
- `train_model.py`: Script to train the ResNet18 model.
- `test_model.py`: Script to test the model on new images.
- `fashion_dataset.json`: Processed dataset with image filenames and labels.

## Dataset and Model
The dataset and trained model files are too large for GitHub. Download them from the links below:
- [LAT.zip](https://1drv.ms/u/c/b82bee97bf2cbc97/EXAnIefyHDdKkICyWkXQWCwBC04x8-RmxcRE_g1yXcpuPA?e=6n23kW) - LAT dataset
- [AAT.zip](https://1drv.ms/u/c/b82bee97bf2cbc97/EYfLrQV2hM9Em73MpfoOk7kBCJxP_4cHEsul0Vcpteax7A?e=zUAjFb) - AAT dataset
- [fashion_classifier.pth](https://1drv.ms/u/c/b82bee97bf2cbc97/EU84bStu4oBPlaNC2a6-HfcBYhJb7jhXu8CAyg-FxQXezw?e=78KB30) - Trained model weights

## Usage
1. Download the dataset and model files from the links above.
2. Extract `LAT.zip` and `AAT.zip` into the `Datasets/` directory.
3. Run `process_lat_data.py` to generate the dataset.
4. Run `train_model.py` to train the model (or use the pre-trained `fashion_classifier.pth`).
5. Run `test_model.py` to test the model on new images.
