import os
import sys
import json
import glob
import time
import shutil
import gc
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# UTF-8 stdout
sys.stdout.reconfigure(line_buffering=True)

SEED = 42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)

OUTPUT_DIR = "kaggle_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load classes
classes_path = os.path.join(OUTPUT_DIR, "classes.json")
with open(classes_path, "r") as f:
    class_to_idx = json.load(f)

class_names = list(class_to_idx.keys())
num_classes = len(class_names)
print(f"[Local Trainer] Loaded {num_classes} classes: {class_names}", flush=True)

# Load Split Manifest
manifest_path = os.path.join(OUTPUT_DIR, "split_manifest.json")
with open(manifest_path, "r") as f:
    split_manifest = json.load(f)

train_paths = [item[0] for item in split_manifest["train"]]
train_labels = [item[1] for item in split_manifest["train"]]
val_paths = [item[0] for item in split_manifest["val"]]
val_labels = [item[1] for item in split_manifest["val"]]
test_paths = [item[0] for item in split_manifest["test"]]
test_labels = [item[1] for item in split_manifest["test"]]

# Adjust local dataset relative paths if needed
def fix_path(p):
    if not os.path.exists(p):
        fname = os.path.basename(p)
        parent_dir = os.path.basename(os.path.dirname(p))
        local_p = os.path.join("Selected_dataset", parent_dir, fname)
        if os.path.exists(local_p):
            return local_p
    return p

train_paths = [fix_path(p) for p in train_paths]
val_paths = [fix_path(p) for p in val_paths]
test_paths = [fix_path(p) for p in test_paths]

print(f"[Local Split] Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}", flush=True)

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

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.20, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

BATCH_SIZE = 64
train_dataset = PlantDataset(train_paths, train_labels, transform=train_transform)
val_dataset = PlantDataset(val_paths, val_labels, transform=val_test_transform)
test_dataset = PlantDataset(test_paths, test_labels, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Hardware] Using device: {device}", flush=True)

def build_model(model_name, num_classes):
    if model_name == "EfficientNet_B0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "DenseNet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "MobileNetV3_Large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "ShuffleNetV2":
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

MODEL_NAMES = ["EfficientNet_B0", "ResNet50", "DenseNet121", "MobileNetV3_Large", "ShuffleNetV2"]
EPOCHS = 30
LR = 0.0005
WEIGHT_DECAY = 0.0001
criterion = nn.CrossEntropyLoss()

def train_and_eval_shufflenet():
    name = "ShuffleNetV2"
    model_dir = os.path.join(OUTPUT_DIR, "models", name)
    os.makedirs(model_dir, exist_ok=True)
    best_ckpt_path = os.path.join(model_dir, "best.pth")
    metrics_path = os.path.join(model_dir, "metrics.json")
    last_ckpt_path = os.path.join(model_dir, "last.pth")

    model = build_model(name, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    start_epoch = 1
    best_val_macro_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []}

    if os.path.exists(last_ckpt_path) and os.path.getsize(last_ckpt_path) > 0:
        print(f"[Resume] Loading last.pth for {name}...", flush=True)
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_macro_f1 = ckpt.get("best_val_macro_f1", 0.0)
        history = ckpt.get("history", history)

    if not (os.path.exists(best_ckpt_path) and os.path.exists(metrics_path) and os.path.getsize(best_ckpt_path) > 0):
        print(f"\n==========================================", flush=True)
        print(f" Starting Local Training for {name} (Epochs {start_epoch}-{EPOCHS})", flush=True)
        print(f"==========================================", flush=True)

        for epoch in range(start_epoch, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            all_preds, all_targets = [], []

            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())

            train_loss = running_loss / len(train_loader.dataset)
            train_acc = accuracy_score(all_targets, all_preds)
            _, _, train_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)

            model.eval()
            val_running_loss = 0.0
            val_preds, val_targets = [], []
            with torch.no_grad():
                for images, targets in val_loader:
                    images, targets = images.to(device), targets.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    val_running_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(targets.cpu().numpy())

            val_loss = val_running_loss / len(val_loader.dataset)
            val_acc = accuracy_score(val_targets, val_preds)
            _, _, val_f1, _ = precision_recall_fscore_support(val_targets, val_preds, average="macro", zero_division=0)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["train_f1"].append(train_f1)
            history["val_f1"].append(val_f1)

            print(f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}", flush=True)

            with open(os.path.join(model_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=4)

            if val_f1 >= best_val_macro_f1:
                best_val_macro_f1 = val_f1
                torch.save(model.state_dict(), best_ckpt_path)

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_macro_f1": best_val_macro_f1,
                "history": history
            }, last_ckpt_path)

    if os.path.exists(best_ckpt_path) and os.path.getsize(best_ckpt_path) > 0:
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    else:
        torch.save(model.state_dict(), best_ckpt_path)
        
    torch.save(model.state_dict(), os.path.join(model_dir, "final.pth"))
    model.eval()

    def get_probs_and_targets(loader):
        probs_list, targets_list = [], []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                probs_list.append(probs)
                targets_list.extend(targets.numpy())
        return np.vstack(probs_list), np.array(targets_list)

    val_probs, val_targets = get_probs_and_targets(val_loader)
    test_probs, test_targets = get_probs_and_targets(test_loader)
    test_preds = test_probs.argmax(axis=1)

    acc = accuracy_score(test_targets, test_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(test_targets, test_preds, average="macro", zero_division=0)

    metrics = {
        "model_name": name,
        "test_accuracy": acc,
        "test_precision_macro": p_macro,
        "test_recall_macro": r_macro,
        "test_f1_macro": f1_macro,
        "param_count": sum(p.numel() for p in model.parameters())
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    cm = confusion_matrix(test_targets, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))
    plt.close()

    report = classification_report(test_targets, test_preds, target_names=class_names, digits=4)
    with open(os.path.join(model_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Plot curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"{name} Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title(f"{name} Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "training_curves.png"))
    plt.close()

    # Sync to pretrained_dataset
    dst_dir = os.path.join("pretrained_dataset", "models", name)
    os.makedirs(dst_dir, exist_ok=True)
    for f in os.listdir(model_dir):
        shutil.copy2(os.path.join(model_dir, f), os.path.join(dst_dir, f))

    print(f"[Completed {name}] Test Acc: {acc:.4f} | Test Macro-F1: {f1_macro:.4f}", flush=True)

if __name__ == "__main__":
    train_and_eval_shufflenet()
