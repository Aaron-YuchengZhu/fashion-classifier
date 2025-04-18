import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import os
import pickle
import pandas as pd

# Setup
base_dir = '/Users/i/Downloads/COMP9444/Group Project/Datasets'
plot_dir = os.path.join(base_dir, 'plot')
docs_dir = os.path.join(base_dir, 'docs')
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(docs_dir, exist_ok=True)
categories = ['Accessories', 'Bags', 'Clothings', 'Shoes']

# Load results
results = pickle.load(open(os.path.join(docs_dir, 'results.pkl'), 'rb'))

# Plot settings
model_names = ['mobilenet_v2', 'resnet18', 'resnet50', 'efficientnet_b0']
model_labels = ['MobileNetV2', 'ResNet18', 'ResNet50', 'EfficientNet-B0']
colors = ['blue', 'orange', 'green', 'red']

# 1. PR Curve
plt.figure(figsize=(10, 8))
for i, (model_name, label) in enumerate(zip(model_names, model_labels)):
    true = results[model_name]['true']
    probs = results[model_name]['probs']
    for j, cat in enumerate(categories):
        true_binary = [1 if t == cat else 0 for t in true]
        prob = [p[j] for p in probs]
        precision, recall, _ = precision_recall_curve(true_binary, prob)
        ap = average_precision_score(true_binary, prob)
        plt.plot(recall, precision, label=f'{label} - {cat} (AP={ap:.3f})', color=colors[i], alpha=0.7)
plt.title('Precision-Recall Curves (Gaussian Noise)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left', fontsize=8, ncol=2)
plt.grid(True)
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.savefig(os.path.join(plot_dir, 'comparison_pr_noise.png'), bbox_inches='tight')
plt.close()

# 2. Accuracy Bar Plot
plt.figure(figsize=(8, 6))
accuracies = [90.40, 94.20, 88.40, 95.40]
x = np.arange(len(model_names))
bars = plt.bar(x, accuracies, color='skyblue')
plt.xticks(x, model_labels, rotation=45)
plt.title('Model Comparison: Accuracy (Gaussian Noise)')
plt.ylabel('Accuracy (%)')
plt.ylim(85, 100)  # Optimize for visibility
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, acc + 0.5, f'{acc:.2f}%', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'comparison_accuracy_noise.png'), bbox_inches='tight')
plt.close()

# 3. mAP Bar Plot
plt.figure(figsize=(8, 6))
maps = [0.979, 0.989, 0.983, 0.993]
x = np.arange(len(model_names))
bars = plt.bar(x, maps, color='lightcoral')
plt.xticks(x, model_labels, rotation=45)
plt.title('Model Comparison: mAP (Gaussian Noise)')
plt.ylabel('mAP')
plt.ylim(0.97, 1.00)  # Zoom in for mAP differences
for bar, m in zip(bars, maps):
    plt.text(bar.get_x() + bar.get_width()/2, m + 0.001, f'{m:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'comparison_map_noise.png'), bbox_inches='tight')
plt.close()

# 4. Confusion Matrices (3 models)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, model_name in enumerate(['mobilenet_v2', 'resnet18', 'efficientnet_b0']):
    cm = results[model_name]['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories, cbar=False, ax=axes[i])
    axes[i].set_title(f'{model_labels[i]} (Gaussian Noise)')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'comparison_cm_noise.png'), bbox_inches='tight')
plt.close()

# 5. Heatmap (Accuracy, F1, mAP)
data = {
    'Model': ['MobileNetV2', 'ResNet18', 'ResNet50', 'EfficientNet-B0'],
    'Accuracy': [90.40, 94.20, 88.40, 95.40],
    'F1': [0.904, 0.942, 0.884, 0.954],
    'mAP': [0.979, 0.989, 0.983, 0.993]
}
df = pd.DataFrame(data).set_index('Model')
plt.figure(figsize=(8, 6))
sns.heatmap(df, annot=True, fmt='.3f', cmap='Blues')
plt.title('Model Performance with Gaussian Noise')
plt.savefig(os.path.join(plot_dir, 'comparison_heatmap_noise.png'), bbox_inches='tight')
plt.close()