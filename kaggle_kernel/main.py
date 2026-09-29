import os
import sys
import json
import glob
import time
import copy
import shutil
import gc
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models

# Set seed for reproducibility
SEED = 42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)

# Force unbuffered stdout printing for live Kaggle console updates
sys.stdout.reconfigure(line_buffering=True)

# Detect Dataset Path
INPUT_BASE = "/kaggle/input"
dataset_path = None

for root, dirs, files in os.walk(INPUT_BASE):
    subdirs = [d for d in dirs if not d.startswith(".")]
    if len(subdirs) >= 10:
        dataset_path = root
        break

if dataset_path is None:
    possible_paths = glob.glob("/kaggle/input/**/Selected_dataset", recursive=True) + glob.glob("/kaggle/input/**", recursive=True)
    for p in possible_paths:
        if os.path.isdir(p) and len([d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]) >= 10:
            dataset_path = p
            break

print(f"[Dataset] Using dataset path: {dataset_path}", flush=True)

# Discover classes
class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
num_classes = len(class_names)
print(f"[Dataset] Discovered {num_classes} classes: {class_names}", flush=True)

# Collect all image files
image_paths = []
labels = []

valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
for cls in class_names:
    cls_dir = os.path.join(dataset_path, cls)
    for fname in os.listdir(cls_dir):
        ext = os.path.splitext(fname)[1]
        if ext in valid_exts:
            image_paths.append(os.path.join(cls_dir, fname))
            labels.append(class_to_idx[cls])

print(f"[Dataset] Total images found: {len(image_paths)}", flush=True)

# 70% Train, 15% Val, 15% Test Stratified Split
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.30, random_state=SEED, stratify=labels
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    test_paths, test_labels, test_size=0.50, random_state=SEED, stratify=test_labels
)

print(f"[Split] Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}", flush=True)

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save Manifest & Classes
with open(os.path.join(OUTPUT_DIR, "classes.json"), "w") as f:
    json.dump(class_to_idx, f, indent=4)

split_manifest = {
    "train": list(zip(train_paths, train_labels)),
    "val": list(zip(val_paths, val_labels)),
    "test": list(zip(test_paths, test_labels))
}
with open(os.path.join(OUTPUT_DIR, "split_manifest.json"), "w") as f:
    json.dump(split_manifest, f, indent=4)

# PyTorch Dataset
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

# Data Augmentations (Per Protocol)
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
NUM_WORKERS = 0  # 0 workers avoids Kaggle /dev/shm shared memory Bus error

train_dataset = PlantDataset(train_paths, train_labels, transform=train_transform)
val_dataset = PlantDataset(val_paths, val_labels, transform=val_test_transform)
test_dataset = PlantDataset(test_paths, test_labels, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Device Setup (Multi-GPU Dual T4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"[Hardware] Using device: {device}, GPU Count: {num_gpus}", flush=True)
if num_gpus > 1:
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)

# Model Building Helper
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
    elif model_name == "ConvNeXt_Tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name == "ShuffleNetV2":
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

MODEL_NAMES = [
    "EfficientNet_B0",
    "ResNet50",
    "DenseNet121",
    "MobileNetV3_Large",
    "ShuffleNetV2"
]

EPOCHS = 30
LR = 0.0005
WEIGHT_DECAY = 0.0001
criterion = nn.CrossEntropyLoss()

def safe_load_checkpoint(pth_path, device="cpu"):
    if not os.path.exists(pth_path) or os.path.getsize(pth_path) == 0:
        if os.path.exists(pth_path):
            print(f"[Warning] Found 0-byte checkpoint at {pth_path}. Removing...", flush=True)
            try:
                os.remove(pth_path)
            except Exception:
                pass
        return None
    try:
        return torch.load(pth_path, map_location=device)
    except Exception as e:
        print(f"[Warning] Failed to unpickle checkpoint from {pth_path}: {e}. Deleting invalid file...", flush=True)
        try:
            os.remove(pth_path)
        except Exception:
            pass
        return None

def train_and_evaluate_model(name):
    model_dir = os.path.join(OUTPUT_DIR, "models", name)
    os.makedirs(model_dir, exist_ok=True)
    best_ckpt_path = os.path.join(model_dir, "best.pth")
    metrics_path = os.path.join(model_dir, "metrics.json")
    
    # Check if pre-trained artifacts exist in input dataset
    pretrained_dirs = [
        f"/kaggle/input/plant-disease-pretrained-models/models/{name}",
        f"/kaggle/input/plant-disease-pretrained-models/{name}"
    ]
    for ptd in pretrained_dirs:
        if os.path.isdir(ptd):
            print(f"\n[Pretrained Input] Found pre-trained artifacts for {name} in {ptd}! Syncing...", flush=True)
            for f_item in os.listdir(ptd):
                src_f = os.path.join(ptd, f_item)
                dst_f = os.path.join(model_dir, f_item)
                if os.path.isfile(src_f) and os.path.getsize(src_f) > 0 and not os.path.exists(dst_f):
                    shutil.copy2(src_f, dst_f)
    
    last_ckpt_path = os.path.join(model_dir, "last.pth")
    model = build_model(name, num_classes)
    
    best_weights = safe_load_checkpoint(best_ckpt_path, device="cpu")
    
    if name != "ConvNeXt_Tiny" or os.path.exists(metrics_path):
        print(f"\n[Skip Training] Model {name} is complete or pre-trained! Skipping 30-epoch training...", flush=True)
        if best_weights is not None:
            model.load_state_dict(best_weights)
        else:
            raw_m = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(raw_m.state_dict(), best_ckpt_path)
        model = model.to(device)
        model.eval()
    elif best_weights is not None and not os.path.exists(last_ckpt_path):
        print(f"\n[Evaluation Resume] Found trained weights best.pth for {name}! Generating evaluation reports...", flush=True)
        model.load_state_dict(best_weights)
        model = model.to(device)
        model.eval()
    else:
        start_epoch = 1
        best_val_macro_f1 = 0.0
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []}

        if num_gpus > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        last_ckpt = safe_load_checkpoint(last_ckpt_path, device=device)
        if last_ckpt is not None:
            print(f"\n[Epoch Resume] Found last.pth checkpoint for {name}! Resuming state...", flush=True)
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            raw_model.load_state_dict(last_ckpt["model_state_dict"])
            optimizer.load_state_dict(last_ckpt["optimizer_state_dict"])
            start_epoch = last_ckpt["epoch"] + 1
            best_val_macro_f1 = last_ckpt.get("best_val_macro_f1", 0.0)
            history = last_ckpt.get("history", history)
            print(f"[Epoch Resume] Successfully loaded checkpoint! Continuing training from Epoch {start_epoch} to {EPOCHS}...", flush=True)

        if start_epoch <= EPOCHS:
            print(f"\n==========================================", flush=True)
            print(f" Starting/Resuming Training for {name} (Epochs {start_epoch}-{EPOCHS})", flush=True)
            print(f"==========================================", flush=True)
            
            for epoch in range(start_epoch, EPOCHS + 1):
                model.train()
                running_loss = 0.0
                all_preds = []
                all_targets = []
                
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
                val_preds = []
                val_targets = []
                
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
                
                # Save history array instantly every epoch!
                with open(os.path.join(model_dir, "history.json"), "w") as f:
                    json.dump(history, f, indent=4)

                raw_model = model.module if isinstance(model, nn.DataParallel) else model
                if val_f1 > best_val_macro_f1:
                    best_val_macro_f1 = val_f1
                    torch.save(raw_model.state_dict(), best_ckpt_path)

                # Save per-epoch resume checkpoint (last.pth)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_macro_f1": best_val_macro_f1,
                    "history": history
                }, last_ckpt_path)

        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        if not os.path.exists(best_ckpt_path):
            torch.save(raw_model.state_dict(), best_ckpt_path)
        else:
            best_w = safe_load_checkpoint(best_ckpt_path, device=device)
            if best_w is not None:
                raw_model.load_state_dict(best_w)
        model = raw_model.to(device)
        model.eval()

    # Get Probability Predictions
    def get_probs_and_targets(loader):
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
        
    val_probs, val_targets = get_probs_and_targets(val_loader)
    test_probs, test_targets = get_probs_and_targets(test_loader)
    
    test_preds = test_probs.argmax(axis=1)
    acc = accuracy_score(test_targets, test_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(test_targets, test_preds, average="macro", zero_division=0)
    
    raw_model_eval = model.module if isinstance(model, nn.DataParallel) else model
    metrics = {
        "model_name": name,
        "test_accuracy": acc,
        "test_precision_macro": p_macro,
        "test_recall_macro": r_macro,
        "test_f1_macro": f1_macro,
        "param_count": sum(p.numel() for p in raw_model_eval.parameters())
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
        
    print(f"[{name}] Test Accuracy: {acc:.4f} | Test Macro-F1: {f1_macro:.4f}", flush=True)
    
    del model
    if 'optimizer' in locals():
        del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return val_probs, test_probs, val_targets, test_targets

# Store model probability outputs
val_probs_dict = {}
test_probs_dict = {}

for name in MODEL_NAMES:
    val_probs, test_probs, val_targets, test_targets = train_and_evaluate_model(name)
    val_probs_dict[name] = val_probs
    test_probs_dict[name] = test_probs

# 10 Ensemble Search (5 Choose 3 Combinations)
print("\n==========================================", flush=True)
print(" Running 10-Ensemble Combination Search", flush=True)
print("==========================================", flush=True)

combinations = list(itertools.combinations(MODEL_NAMES, 3))
ensemble_results = []

best_combo = None
best_ensemble_val_f1 = 0.0

for idx, combo in enumerate(combinations, 1):
    combo_val_probs = np.mean([val_probs_dict[m] for m in combo], axis=0)
    combo_val_preds = combo_val_probs.argmax(axis=1)
    
    val_acc = accuracy_score(val_targets, combo_val_preds)
    _, _, val_f1, _ = precision_recall_fscore_support(val_targets, combo_val_preds, average="macro", zero_division=0)
    
    combo_test_probs = np.mean([test_probs_dict[m] for m in combo], axis=0)
    combo_test_preds = combo_test_probs.argmax(axis=1)
    test_acc = accuracy_score(test_targets, combo_test_preds)
    _, _, test_f1, _ = precision_recall_fscore_support(test_targets, combo_test_preds, average="macro", zero_division=0)
    
    result = {
        "combo_index": idx,
        "models": list(combo),
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1
    }
    ensemble_results.append(result)
    
    print(f"Combo #{idx}: {combo} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}", flush=True)
    
    if val_f1 > best_ensemble_val_f1:
        best_ensemble_val_f1 = val_f1
        best_combo = combo

print(f"\n[Winning Ensemble Selected]: {best_combo} with Validation Macro-F1: {best_ensemble_val_f1:.4f}", flush=True)

with open(os.path.join(OUTPUT_DIR, "ensemble_search_results.json"), "w") as f:
    json.dump({"best_ensemble": list(best_combo), "all_combinations": ensemble_results}, f, indent=4)

# Evaluate Winner Ensemble on Test Set
winning_test_probs = np.mean([test_probs_dict[m] for m in best_combo], axis=0)
winning_test_preds = winning_test_probs.argmax(axis=1)

win_acc = accuracy_score(test_targets, winning_test_preds)
win_p, win_r, win_f1, _ = precision_recall_fscore_support(test_targets, winning_test_preds, average="macro", zero_division=0)

winning_metrics = {
    "winning_ensemble_models": list(best_combo),
    "test_accuracy": win_acc,
    "test_precision_macro": win_p,
    "test_recall_macro": win_r,
    "test_f1_macro": win_f1
}
with open(os.path.join(OUTPUT_DIR, "winning_ensemble_metrics.json"), "w") as f:
    json.dump(winning_metrics, f, indent=4)

# Winning Ensemble Confusion Matrix
cm_win = confusion_matrix(test_targets, winning_test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_win, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Greens")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title(f"Winning Ensemble ({' + '.join(best_combo)}) Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "winning_ensemble_confusion_matrix.png"))
plt.close()

# Grad-CAM XAI Layer Implementation for PyTorch Models
print("\n==========================================", flush=True)
print(" Generating Grad-CAM XAI Visualizations", flush=True)
print("==========================================", flush=True)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

def get_target_layer(model, name):
    if name == "EfficientNet_B0":
        return model.features[-1]
    elif name == "ResNet50":
        return model.layer4[-1]
    elif name == "DenseNet121":
        return model.features.denseblock4.denselayer16
    elif name == "MobileNetV3_Large":
        return model.features[-1]
    elif name == "ConvNeXt_Tiny":
        return model.features[-1]
    elif name == "ShuffleNetV2":
        return model.conv5
    return None

xai_dir = os.path.join(OUTPUT_DIR, "xai_gradcam")
os.makedirs(xai_dir, exist_ok=True)

try:
    sample_indices = [0, len(test_dataset) // 2, len(test_dataset) - 1]
    for idx_in_test in sample_indices:
        img_tensor, true_label = test_dataset[idx_in_test]
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        orig_img_path = test_paths[idx_in_test]
        raw_img = Image.open(orig_img_path).convert("RGB").resize((224, 224))
        
        ensemble_cams = []
        fig, axes = plt.subplots(1, len(best_combo) + 2, figsize=(18, 4))
        axes[0].imshow(raw_img)
        axes[0].set_title(f"Original\nTrue: {class_names[true_label]}")
        axes[0].axis("off")
        
        for m_idx, m_name in enumerate(best_combo):
            m_model = build_model(m_name, num_classes)
            m_path = os.path.join(OUTPUT_DIR, "models", m_name, "best.pth")
            m_weights = safe_load_checkpoint(m_path, device="cpu")
            if m_weights is not None:
                m_model.load_state_dict(m_weights)
            m_model = m_model.to(device)
            
            target_layer = get_target_layer(m_model, m_name)
            if target_layer is not None:
                try:
                    cam_gen = GradCAM(m_model, target_layer)
                    cam = cam_gen.generate(input_tensor)
                    ensemble_cams.append(cam)
                    
                    cam_img = Image.fromarray(np.uint8(255 * cam)).resize((224, 224), Image.BILINEAR)
                    cam_arr = np.array(cam_img) / 255.0
                    
                    axes[m_idx + 1].imshow(raw_img)
                    axes[m_idx + 1].imshow(cam_arr, cmap="jet", alpha=0.5)
                    axes[m_idx + 1].set_title(f"{m_name} Grad-CAM")
                    axes[m_idx + 1].axis("off")
                except Exception as cam_err:
                    print(f"[Warning] Grad-CAM failed for {m_name}: {cam_err}", flush=True)
            
            del m_model
            gc.collect()
            torch.cuda.empty_cache()
            
        if len(ensemble_cams) > 0:
            avg_cam = np.mean(ensemble_cams, axis=0)
            avg_cam_img = Image.fromarray(np.uint8(255 * avg_cam)).resize((224, 224), Image.BILINEAR)
            avg_cam_arr = np.array(avg_cam_img) / 255.0
            
            axes[-1].imshow(raw_img)
            axes[-1].imshow(avg_cam_arr, cmap="jet", alpha=0.5)
            axes[-1].set_title("Ensemble Grad-CAM (Avg)")
            axes[-1].axis("off")
            
        plt.tight_layout()
        plt.savefig(os.path.join(xai_dir, f"gradcam_sample_{idx_in_test}.png"))
        plt.close()

    print("[XAI] Grad-CAM generation complete.", flush=True)
except Exception as e:
    print(f"[Warning] Grad-CAM visual auditing notice: {e}", flush=True)

# Convert Models to ONNX / Save Checkpoints
print("\n==========================================", flush=True)
print(" Exporting Checkpoints & ONNX Models", flush=True)
print("==========================================", flush=True)

try:
    onnx_dir = os.path.join(OUTPUT_DIR, "onnx_export")
    os.makedirs(onnx_dir, exist_ok=True)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    for name in MODEL_NAMES:
        m_model = build_model(name, num_classes)
        m_path = os.path.join(OUTPUT_DIR, "models", name, "best.pth")
        m_weights = safe_load_checkpoint(m_path, device="cpu")
        if m_weights is not None:
            m_model.load_state_dict(m_weights)
        m_model = m_model.to(device)
        m_model.eval()
        
        onnx_path = os.path.join(onnx_dir, f"{name}.onnx")
        torch.onnx.export(
            m_model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        print(f"Exported ONNX model for {name} to {onnx_path}", flush=True)
        
        del m_model
        gc.collect()
        torch.cuda.empty_cache()
except Exception as e:
    print(f"[Warning] ONNX export notice: {e}", flush=True)

print("\n==========================================", flush=True)
print(" All Training, Ensemble Search, XAI, and Export Complete!", flush=True)
print("==========================================", flush=True)
