import os
import json
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Paths
LOCAL_MODEL_DIR = r"c:\Users\toron\OneDrive\Desktop\ML_FRESH\kaggle_output"
REAL_WORLD_DIR = r"c:\Users\toron\OneDrive\Desktop\ML_FRESH\real_world_test_dataset"
OUTPUT_EVAL_DIR = r"c:\Users\toron\OneDrive\Desktop\ML_FRESH\real_world_eval_results"
os.makedirs(OUTPUT_EVAL_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Hardware] Using device for field evaluation: {device}")

# Load Classes
classes_file = os.path.join(LOCAL_MODEL_DIR, "classes.json")
if not os.path.exists(classes_file):
    print(f"Error: {classes_file} not found. Waiting for Kaggle outputs to finish downloading.")
    exit(1)

with open(classes_file, "r") as f:
    class_to_idx = json.load(f)

class_names = sorted(list(class_to_idx.keys()), key=lambda k: class_to_idx[k])
num_classes = len(class_names)
print(f"[Classes] Found {num_classes} classes: {class_names}")

# Collect Real-World Field Test Images
image_paths = []
labels = []

valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
for cls in class_names:
    cls_dir = os.path.join(REAL_WORLD_DIR, cls)
    if os.path.exists(cls_dir):
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1] in valid_exts:
                image_paths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_idx[cls])

print(f"[Field Dataset] Total real-world images collected: {len(image_paths)}")

if len(image_paths) == 0:
    print("No images found in real_world_test_dataset. Run download_real_world_dataset.py first!")
    exit(1)

# Transform
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class RealWorldDataset(Dataset):
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

dataset = RealWorldDataset(image_paths, labels, transform=eval_transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

# Helper to Build Model Architecture
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
    elif model_name == "ShuffleNetV2":
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

# Find trained models
model_dirs = glob.glob(os.path.join(LOCAL_MODEL_DIR, "models", "*"))
trained_models = []

for md in model_dirs:
    m_name = os.path.basename(md)
    ckpt_path = os.path.join(md, "best.pth")
    if os.path.exists(ckpt_path):
        trained_models.append(m_name)

print(f"[Trained Models Found]: {trained_models}")

model_probs_dict = {}

# Evaluate each model on real-world dataset
for name in trained_models:
    ckpt_path = os.path.join(LOCAL_MODEL_DIR, "models", name, "best.pth")
    model = build_model(name, num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    probs_list = []
    targets_list = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            probs_list.append(probs)
            targets_list.extend(targets.numpy())

    probs_arr = np.vstack(probs_list)
    targets_arr = np.array(targets_list)
    model_probs_dict[name] = probs_arr

    preds_arr = probs_arr.argmax(axis=1)
    acc = accuracy_score(targets_arr, preds_arr)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(targets_arr, preds_arr, average="macro", zero_division=0)

    print(f"\n--- [Real-World Field Evaluation] {name} ---")
    print(f"  Field Accuracy: {acc:.4f} | Field Macro-F1: {f1_macro:.4f}")

    metrics = {
        "model_name": name,
        "field_accuracy": acc,
        "field_precision_macro": p_macro,
        "field_recall_macro": r_macro,
        "field_f1_macro": f1_macro,
        "total_field_images": len(targets_arr)
    }

    out_dir = os.path.join(OUTPUT_EVAL_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "real_world_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Confusion Matrix
    cm = confusion_matrix(targets_arr, preds_arr)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Oranges")
    plt.ylabel("Actual Field Label")
    plt.xlabel("Predicted Label")
    plt.title(f"{name} Real-World Field Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "real_world_confusion_matrix.png"))
    plt.close()

# Ensemble Evaluation (if multiple models available)
if len(trained_models) >= 2:
    print(f"\n==========================================")
    print(" Real-World Soft-Voting Ensemble Evaluation")
    print("==========================================")
    
    ensemble_probs = np.mean([model_probs_dict[m] for m in trained_models], axis=0)
    ensemble_preds = ensemble_probs.argmax(axis=1)
    
    ens_acc = accuracy_score(targets_arr, ensemble_preds)
    ens_p, ens_r, ens_f1, _ = precision_recall_fscore_support(targets_arr, ensemble_preds, average="macro", zero_division=0)
    
    print(f"[Ensemble ({' + '.join(trained_models)})]: Field Accuracy: {ens_acc:.4f} | Field Macro-F1: {ens_f1:.4f}")
    
    ens_metrics = {
        "models": trained_models,
        "field_accuracy": ens_acc,
        "field_precision_macro": ens_p,
        "field_recall_macro": ens_r,
        "field_f1_macro": ens_f1
    }
    with open(os.path.join(OUTPUT_EVAL_DIR, "ensemble_real_world_metrics.json"), "w") as f:
        json.dump(ens_metrics, f, indent=4)

    cm_ens = confusion_matrix(targets_arr, ensemble_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_ens, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Purples")
    plt.ylabel("Actual Field Label")
    plt.xlabel("Predicted Label")
    plt.title("Ensemble Real-World Field Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_EVAL_DIR, "ensemble_real_world_confusion_matrix.png"))
    plt.close()

print("\n==========================================")
print(" Real-World Evaluation Complete!")
print(f" Results saved to: {OUTPUT_EVAL_DIR}")
print("==========================================")
