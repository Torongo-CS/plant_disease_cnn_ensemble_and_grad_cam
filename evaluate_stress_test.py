import os
import sys
import json
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Stress Test Evaluator] Using device: {device}")

# 1. Load trained classes mapping
with open("classes.json", "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
num_classes = len(class_names)

# 2. Map Stress_test folder names to dataset class names
folder_to_class = {
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold"
}

stress_dir = "Stress_test"
stress_images = []
stress_labels = []

valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

for folder, target_cls in folder_to_class.items():
    folder_path = os.path.join(stress_dir, folder)
    if os.path.exists(folder_path):
        target_idx = class_to_idx[target_cls]
        count = 0
        for fname in os.listdir(folder_path):
            ext = os.path.splitext(fname)[1]
            if ext in valid_exts:
                img_p = os.path.join(folder_path, fname)
                stress_images.append(img_p)
                stress_labels.append(target_idx)
                count += 1
        print(f"[Dataset] Folder '{folder}' -> '{target_cls}' (Idx {target_idx}): {count} images")
    else:
        print(f"[Warning] Folder missing: {folder_path}")

print(f"[Stress Test] Total images collected: {len(stress_images)}")

# Transform for test images (Standard ImageNet normalization)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class StressDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

dataset = StressDataset(stress_images, stress_labels, test_transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

# Model loader factory
def load_trained_model(model_name):
    if model_name == "EfficientNet_B0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "ResNet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "DenseNet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name == "MobileNetV3_Large":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif model_name == "ShuffleNetV2":
        model = models.shufflenet_v2_x1_0(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pth_path = os.path.join("kaggle_output", "models", model_name, "best.pth")
    if not os.path.exists(pth_path):
        pth_path = os.path.join("pretrained_dataset", "models", model_name, "best.pth")
    
    print(f"Loading {model_name} weights from: {pth_path}")
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 3. Collect probability predictions for all 5 models
all_models = ["DenseNet121", "EfficientNet_B0", "MobileNetV3_Large", "ResNet50", "ShuffleNetV2"]
model_probs = {}

for m in all_models:
    model = load_trained_model(m)
    probs_list = []
    
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)
            
    model_probs[m] = np.concatenate(probs_list, axis=0)

y_true = np.array(stress_labels)

# Winning ensemble model: DenseNet121 + EfficientNet_B0 + MobileNetV3_Large (Equal soft-voting)
winning_models = ["DenseNet121", "EfficientNet_B0", "MobileNetV3_Large"]
winning_probs = np.mean([model_probs[m] for m in winning_models], axis=0)

# Full 5-Model Ensemble
full_5_probs = np.mean([model_probs[m] for m in all_models], axis=0)

eval_setups = {
    "DenseNet121": model_probs["DenseNet121"],
    "EfficientNet_B0": model_probs["EfficientNet_B0"],
    "MobileNetV3_Large": model_probs["MobileNetV3_Large"],
    "ResNet50": model_probs["ResNet50"],
    "ShuffleNetV2": model_probs["ShuffleNetV2"],
    "Winning_Ensemble (3-Model)": winning_probs,
    "Full_5Model_Ensemble": full_5_probs
}

# Output directory for results
out_dir = "stress_test_results"
os.makedirs(out_dir, exist_ok=True)

metrics_summary = {}
summary_table_data = []

# Academic evaluation metrics loop
for name, probs in eval_setups.items():
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, preds, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, preds, average="weighted", zero_division=0)
    
    # Top-3 Accuracy
    top3_correct = 0
    for idx_i in range(len(y_true)):
        top3_preds = np.argsort(probs[idx_i])[-3:]
        if y_true[idx_i] in top3_preds:
            top3_correct += 1
    top3_acc = top3_correct / len(y_true)

    metrics_summary[name] = {
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "top3_accuracy": float(top3_acc)
    }

    summary_table_data.append({
        "Model Setup": name,
        "Accuracy (%)": round(acc * 100, 2),
        "Macro Precision (%)": round(p_macro * 100, 2),
        "Macro Recall (%)": round(r_macro * 100, 2),
        "Macro F1-Score (%)": round(f1_macro * 100, 2),
        "Weighted F1 (%)": round(f1_weighted * 100, 2),
        "Top-3 Accuracy (%)": round(top3_acc * 100, 2)
    })

# Save JSON metrics summary
with open(os.path.join(out_dir, "stress_test_metrics_summary.json"), "w") as f:
    json.dump(metrics_summary, f, indent=4)

# Save CSV summary table
df_summary = pd.DataFrame(summary_table_data)
df_summary.to_csv(os.path.join(out_dir, "stress_test_metrics_table.csv"), index=False)

print("\n==========================================")
print(" STRESS TEST ACADEMIC EVALUATION RESULTS")
print("==========================================")
print(df_summary.to_string(index=False))

# Plot comparative confusion matrices
fig, axes = plt.subplots(2, 4, figsize=(22, 11))
axes = axes.flatten()

unique_present_classes = sorted(list(set(y_true)))
present_class_labels = [class_names[c] for c in unique_present_classes]

for idx_i, (name, probs) in enumerate(eval_setups.items()):
    preds = np.argmax(probs, axis=1)
    cm = confusion_matrix(y_true, preds, labels=unique_present_classes)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[idx_i],
                xticklabels=[c.split('___')[-1][:12] for c in present_class_labels],
                yticklabels=[c.split('___')[-1][:12] for c in present_class_labels],
                cbar=False)
    axes[idx_i].set_title(f"{name}\nAcc: {metrics_summary[name]['accuracy']*100:.2f}% | F1: {metrics_summary[name]['f1_macro']*100:.2f}%", fontsize=10, fontweight='bold')
    axes[idx_i].set_xlabel("Predicted")
    axes[idx_i].set_ylabel("True")

# Hide extra subplot
axes[7].axis('off')
plt.suptitle("Stress Test Evaluation - Normalized Confusion Matrices", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "stress_test_confusion_matrices.png"), dpi=150)
plt.close()

# Plot Macro F1 comparison bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(df_summary["Model Setup"], df_summary["Macro F1-Score (%)"], color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#FF9F43'])
plt.ylabel("Macro F1-Score (%)", fontsize=12, fontweight='bold')
plt.title("Out-of-Distribution Stress Test Performance Comparison", fontsize=14, fontweight='bold')
plt.xticks(rotation=25, ha='right', fontsize=10)
plt.ylim(0, 105)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "stress_test_f1_comparison.png"), dpi=150)
plt.close()

print(f"\nSaved stress test artifacts to '{out_dir}/'")
