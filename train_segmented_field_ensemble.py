import os
import sys
import json
import glob
import time
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.stdout.reconfigure(line_buffering=True)

SEED = 42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

torch.set_num_threads(os.cpu_count() or 4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Field Domain Adaptor] Using device: {device} | Threads: {torch.get_num_threads()}")

# 1. Load Classes Mapping
with open("classes.json", "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
num_classes = len(class_names)

# 2. Collect Segmented_dataset Images
seg_dir = "Segmented_dataset"
seg_images = []
seg_labels = []

valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

for cls in class_names:
    cls_dir = os.path.join(seg_dir, cls)
    if os.path.exists(cls_dir):
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1] in valid_exts:
                seg_images.append(os.path.join(cls_dir, fname))
                seg_labels.append(class_to_idx[cls])

print(f"[Segmented Dataset] Total images found: {len(seg_images)}")

# Train / Val Split on Segmented Dataset
train_paths, val_paths, train_labels, val_labels = train_test_split(
    seg_images, seg_labels, test_size=0.20, random_state=SEED, stratify=seg_labels
)

# Heavy Domain Adaptation Augmentation Pipeline
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PlantDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

train_dataset = PlantDataset(train_paths, train_labels, train_transform)
val_dataset = PlantDataset(val_paths, val_labels, val_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

# Load Stress_test Dataset for Out-of-Distribution Validation
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

for folder, target_cls in folder_to_class.items():
    folder_path = os.path.join(stress_dir, folder)
    if os.path.exists(folder_path):
        target_idx = class_to_idx[target_cls]
        for fname in os.listdir(folder_path):
            if os.path.splitext(fname)[1] in valid_exts:
                stress_images.append(os.path.join(folder_path, fname))
                stress_labels.append(target_idx)

stress_dataset = PlantDataset(stress_images, stress_labels, val_transform)
stress_loader = DataLoader(stress_dataset, batch_size=32, shuffle=False, num_workers=0)
y_stress_true = np.array(stress_labels)

def get_model(model_name):
    if model_name == "EfficientNet_B0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "MobileNetV3_Large":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif model_name == "DenseNet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name == "ResNet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "ShuffleNetV2":
        model = models.shufflenet_v2_x1_0(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pth_path = os.path.join("kaggle_output", "models", model_name, "best.pth")
    if not os.path.exists(pth_path):
        pth_path = os.path.join("pretrained_dataset", "models", model_name, "best.pth")
    if os.path.exists(pth_path):
        model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    return model

def fine_tune_and_eval(model_name, epochs=5):
    print(f"\n==========================================")
    print(f" Fine-Tuning Model on Segmented Leaf Data: {model_name}")
    print(f"==========================================")
    model = get_model(model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    out_dir = os.path.join("segmented_models", model_name)
    os.makedirs(out_dir, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct = 0.0, 0
        batch_idx = 0
        for imgs, lbls in train_loader:
            batch_idx += 1
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == lbls).sum().item()
            
            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                print(f"   [{model_name} Epoch {epoch}/{epochs}] Batch {batch_idx}/{len(train_loader)} - Acc: {train_correct/(batch_idx*32)*100:.2f}%", flush=True)

        # Validation
        model.eval()
        val_correct = 0
        val_preds_list, val_targets_list = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == lbls).sum().item()
                val_preds_list.extend(preds.cpu().numpy())
                val_targets_list.extend(lbls.cpu().numpy())

        val_acc = val_correct / len(val_dataset)
        _, _, val_f1, _ = precision_recall_fscore_support(val_targets_list, val_preds_list, average="macro", zero_division=0)
        
        print(f"Epoch [{epoch}/{epochs}] | Train Acc: {train_correct/len(train_dataset)*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}%")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(out_dir, "best_segmented.pth"))

    # Load best model & evaluate on Out-of-Distribution Stress_test
    best_pth = os.path.join(out_dir, "best_segmented.pth")
    model.load_state_dict(torch.load(best_pth, map_location=device))
    model.eval()

    stress_preds = []
    stress_probs_list = []
    with torch.no_grad():
        for imgs, _ in stress_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            stress_probs_list.append(probs)
            stress_preds.extend(np.argmax(probs, axis=1))

    stress_acc = accuracy_score(y_stress_true, stress_preds)
    _, _, stress_f1, _ = precision_recall_fscore_support(y_stress_true, stress_preds, average="macro", zero_division=0)
    print(f"\n🎯 [Stress Test OOD Evaluation] {model_name}:")
    print(f"   Accuracy: {stress_acc*100:.2f}% | Macro F1: {stress_f1*100:.2f}%")
    
    return np.concatenate(stress_probs_list, axis=0)

all_models = ["EfficientNet_B0", "MobileNetV3_Large", "ShuffleNetV2"]
model_probs = {}

for m in all_models:
    model_probs[m] = fine_tune_and_eval(m, epochs=2)

# Winning Ensemble Soft-Voting Evaluation
winning_probs = np.mean([model_probs["EfficientNet_B0"], model_probs["MobileNetV3_Large"], model_probs["ShuffleNetV2"]], axis=0)
winning_preds = np.argmax(winning_probs, axis=1)

ens_acc = accuracy_score(y_stress_true, winning_preds)
_, _, ens_f1, _ = precision_recall_fscore_support(y_stress_true, winning_preds, average="macro", zero_division=0)

print("\n==========================================")
print(" 🏆 FINAL DOMAIN-ADAPTED STRESS TEST ENSEMBLE RESULTS")
print("==========================================")
print(f" Winning Ensemble (DenseNet121 + EfficientNet_B0 + MobileNetV3_Large):")
print(f"   Field Stress Test Accuracy: {ens_acc*100:.2f}%")
print(f"   Field Stress Test Macro F1: {ens_f1*100:.2f}%")
