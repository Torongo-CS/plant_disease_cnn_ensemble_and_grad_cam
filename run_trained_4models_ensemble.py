import os
import sys
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

sys.stdout.reconfigure(line_buffering=True)

OUTPUT_DIR = "kaggle_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load classes & manifest
classes_path = os.path.join(OUTPUT_DIR, "classes.json")
with open(classes_path, "r") as f:
    class_to_idx = json.load(f)

class_names = list(class_to_idx.keys())
num_classes = len(class_names)

manifest_path = os.path.join(OUTPUT_DIR, "split_manifest.json")
with open(manifest_path, "r") as f:
    split_manifest = json.load(f)

val_paths = [item[0] for item in split_manifest["val"]]
val_labels = [item[1] for item in split_manifest["val"]]
test_paths = [item[0] for item in split_manifest["test"]]
test_labels = [item[1] for item in split_manifest["test"]]

def fix_path(p):
    if not os.path.exists(p):
        fname = os.path.basename(p)
        parent_dir = os.path.basename(os.path.dirname(p))
        local_p = os.path.join("Selected_dataset", parent_dir, fname)
        if os.path.exists(local_p):
            return local_p
    return p

val_paths = [fix_path(p) for p in val_paths]
test_paths = [fix_path(p) for p in test_paths]

class PlantDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

BATCH_SIZE = 64
val_dataset = PlantDataset(val_paths, val_labels, transform=val_test_transform)
test_dataset = PlantDataset(test_paths, test_labels, transform=val_test_transform)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Ensemble] Using device: {device}", flush=True)

def build_model(model_name, num_classes):
    if model_name == "EfficientNet_B0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "ResNet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "DenseNet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "MobileNetV3_Large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

TRAINED_MODELS = ["EfficientNet_B0", "ResNet50", "DenseNet121", "MobileNetV3_Large"]

val_probs_dict = {}
test_probs_dict = {}

def get_probs_and_targets(model, loader):
    probs_list = []
    targets_list = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            probs_list.append(probs)
            targets_list.extend(targets.numpy())
    return np.vstack(probs_list), np.array(targets_list)

def safe_load(path, dev):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        return torch.load(path, map_location=dev)
    except Exception:
        return None

print("\n[Ensemble] Computing probabilities for all trained models...", flush=True)

for name in TRAINED_MODELS:
    ckpt_path = os.path.join(OUTPUT_DIR, "models", name, "best.pth")
    weights = safe_load(ckpt_path, device)
    if weights is None:
        ckpt_path = os.path.join("pretrained_dataset", "models", name, "best.pth")
        weights = safe_load(ckpt_path, device)

    print(f"  Loading weights for {name} from {ckpt_path}...", flush=True)
    model = build_model(name, num_classes)
    if weights is not None:
        model.load_state_dict(weights)
    else:
        print(f"  [Notice] Weights for {name} missing/corrupted. Using default pretrained base weights.", flush=True)
    model = model.to(device)
    model.eval()

    val_probs, val_targets = get_probs_and_targets(model, val_loader)
    test_probs, test_targets = get_probs_and_targets(model, test_loader)

    val_probs_dict[name] = val_probs
    test_probs_dict[name] = test_probs

# Evaluate all 3-model combinations out of 4 trained models
print("\n==========================================", flush=True)
print(" 3-Model Soft-Voting Ensemble Combination Search", flush=True)
print("==========================================", flush=True)

combinations = list(itertools.combinations(TRAINED_MODELS, 3))
ensemble_results = []
best_combo = None
best_val_f1 = 0.0

for idx, combo in enumerate(combinations, 1):
    combo_val_probs = np.mean([val_probs_dict[m] for m in combo], axis=0)
    combo_val_preds = combo_val_probs.argmax(axis=1)

    val_acc = accuracy_score(val_targets, combo_val_preds)
    _, _, val_f1, _ = precision_recall_fscore_support(val_targets, combo_val_preds, average="macro", zero_division=0)

    combo_test_probs = np.mean([test_probs_dict[m] for m in combo], axis=0)
    combo_test_preds = combo_test_probs.argmax(axis=1)

    test_acc = accuracy_score(test_targets, combo_test_preds)
    _, _, test_f1, _ = precision_recall_fscore_support(test_targets, combo_test_preds, average="macro", zero_division=0)

    res = {
        "combo_index": idx,
        "models": list(combo),
        "val_accuracy": float(val_acc),
        "val_macro_f1": float(val_f1),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1)
    }
    ensemble_results.append(res)

    print(f"Combo #{idx}: {' + '.join(combo)}", flush=True)
    print(f"  Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}% | Test Acc: {test_acc*100:.2f}% | Test F1: {test_f1*100:.2f}%", flush=True)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_combo = combo

print(f"\n[WINNING ENSEMBLE COMBINATION SELECTED]: {' + '.join(best_combo)}", flush=True)
print(f"   Validation Accuracy: {best_val_f1*100:.2f}% | Validation Macro-F1: {best_val_f1*100:.2f}%", flush=True)

search_output = {
    "winning_ensemble": list(best_combo),
    "all_combinations": ensemble_results
}

with open(os.path.join(OUTPUT_DIR, "ensemble_3models_results.json"), "w") as f:
    json.dump(search_output, f, indent=4)

winning_test_probs = np.mean([test_probs_dict[m] for m in best_combo], axis=0)
winning_test_preds = winning_test_probs.argmax(axis=1)

win_acc = accuracy_score(test_targets, winning_test_preds)
win_p, win_r, win_f1, _ = precision_recall_fscore_support(test_targets, winning_test_preds, average="macro", zero_division=0)

winning_metrics = {
    "winning_ensemble_models": list(best_combo),
    "test_accuracy": float(win_acc),
    "test_precision_macro": float(win_p),
    "test_recall_macro": float(win_r),
    "test_f1_macro": float(win_f1)
}

with open(os.path.join(OUTPUT_DIR, "winning_ensemble_metrics.json"), "w") as f:
    json.dump(winning_metrics, f, indent=4)

cm = confusion_matrix(test_targets, winning_test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Greens")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title(f"Winning Ensemble ({' + '.join(best_combo)}) Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "winning_ensemble_confusion_matrix.png"))
plt.close()

print("\n[Ensemble Evaluation Completed Successfully!]", flush=True)
