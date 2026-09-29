import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Segmented Stress Eval] Using device: {device}")

# Load class mapping
with open("classes.json", "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

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
        for fname in os.listdir(folder_path):
            if os.path.splitext(fname)[1] in valid_exts:
                stress_images.append(os.path.join(folder_path, fname))
                stress_labels.append(target_idx)

print(f"[Dataset] Total stress test images: {len(stress_images)}")

# Standard PyTorch normalization
norm_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Leaf Segmentation & Background Removal Pipeline
def preprocess_leaf_roi(img_path):
    img_np = cv2.imread(img_path)
    if img_np is None:
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)[:, :, ::-1]

    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)

    # Green & yellow disease leaf tissue HSV bounds
    lower1 = np.array([20, 20, 20])
    upper1 = np.array([95, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)

    # Brown / necrotic lesion tissue HSV bounds
    lower2 = np.array([10, 30, 20])
    upper2 = np.array([25, 255, 200])
    mask2 = cv2.inRange(hsv, lower2, upper2)

    combined_mask = cv2.bitwise_or(mask1, mask2)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

    # Find largest contour bounding box
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w > 20 and h > 20:
            # Crop to ROI with 5% padding
            pad_w = int(w * 0.05)
            pad_h = int(h * 0.05)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(img_rgb.shape[1], x + w + pad_w)
            y2 = min(img_rgb.shape[0], y + h + pad_h)
            
            img_rgb = img_rgb[y1:y2, x1:x2]
            clean_mask = clean_mask[y1:y2, x1:x2]

    # Apply mask (zero out background)
    mask_3d = np.repeat(clean_mask[:, :, np.newaxis], 3, axis=2) / 255.0
    segmented_rgb = (img_rgb * mask_3d).astype(np.uint8)

    pil_img = Image.fromarray(segmented_rgb)
    return norm_transform(pil_img)

class SegmentedStressDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        tensor = preprocess_leaf_roi(self.paths[idx])
        return tensor, self.labels[idx]

def load_trained_model(model_name):
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

    pth_path = os.path.join("kaggle_output", "models", model_name, "best.pth")
    if not os.path.exists(pth_path):
        pth_path = os.path.join("pretrained_dataset", "models", model_name, "best.pth")
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()
    return model

print("\n--- Evaluating Segmented Leaf ROI Pipeline on Stress Test Dataset ---")
ds = SegmentedStressDataset(stress_images, stress_labels)
loader = DataLoader(ds, batch_size=32, shuffle=False)

all_models = ["DenseNet121", "EfficientNet_B0", "MobileNetV3_Large", "ResNet50", "ShuffleNetV2"]
model_probs = {}

for m in all_models:
    print(f"Evaluating model on segmented leaf ROI: {m}...")
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

# Evaluate individual & ensemble models
winning_models = ["DenseNet121", "EfficientNet_B0", "MobileNetV3_Large"]
winning_probs = np.mean([model_probs[m] for m in winning_models], axis=0)
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

print("\n==========================================")
print(" SEGMENTED LEAF ROI STRESS TEST RESULTS")
print("==========================================")
for name, probs in eval_setups.items():
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, preds, average="macro", zero_division=0)
    print(f"{name:30s} | Accuracy: {acc*100:6.2f}% | Macro F1: {f1_macro*100:6.2f}%")
