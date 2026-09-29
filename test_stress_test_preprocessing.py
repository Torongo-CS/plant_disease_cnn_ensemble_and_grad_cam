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
print(f"[OOD Enhancement] Using device: {device}")

# Load classes
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

# Preprocessing Pipeline 1: Standard Resize
standard_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Preprocessing Pipeline 2: Green-Channel Bounding Box Leaf Crop
def leaf_crop_transform(pil_img):
    img_np = np.array(pil_img)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    # Convert to HSV and isolate green/yellow plant tissue
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    # Range for green and yellowish diseased leaf areas
    lower_plant = np.array([20, 30, 30])
    upper_plant = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_plant, upper_plant)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find largest contour (the leaf)
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w > 30 and h > 30:
            crop = img_np[y:y+h, x:x+w]
            pil_img = Image.fromarray(crop)

    return standard_transform(pil_img)

# Preprocessing Pipeline 3: Foreground Masking (Zero out non-leaf background)
def leaf_masked_transform(pil_img):
    img_np = np.array(pil_img)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lower_plant = np.array([15, 25, 25])
    upper_plant = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_plant, upper_plant)
    
    # Apply morphological operations
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2) / 255.0
    
    masked_img = (img_np * mask_3d).astype(np.uint8)
    return standard_transform(Image.fromarray(masked_img))

class CustomDataset(Dataset):
    def __init__(self, paths, labels, transform_fn):
        self.paths = paths
        self.labels = labels
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_pil = Image.open(self.paths[idx]).convert("RGB")
        tensor = self.transform_fn(img_pil)
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

print("\n--- Testing OOD Preprocessing Techniques ---")

pipelines = {
    "1. Standard Preprocessing": standard_transform,
    "2. Leaf Bounding-Box Crop": leaf_crop_transform,
    "3. Background Mask Isolation": leaf_masked_transform
}

models_to_eval = ["DenseNet121", "EfficientNet_B0", "MobileNetV3_Large", "ShuffleNetV2"]
loaded_models = {m: load_trained_model(m) for m in models_to_eval}
y_true = np.array(stress_labels)

for pipe_name, transform_fn in pipelines.items():
    print(f"\nEvaluating: {pipe_name}...")
    ds = CustomDataset(stress_images, stress_labels, transform_fn)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    probs_dict = {}
    for m_name, model in loaded_models.items():
        probs_list = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)
        probs_dict[m_name] = np.concatenate(probs_list, axis=0)

    # Winning ensemble
    ens_probs = np.mean([probs_dict["DenseNet121"], probs_dict["EfficientNet_B0"], probs_dict["MobileNetV3_Large"]], axis=0)
    preds = np.argmax(ens_probs, axis=1)
    acc = accuracy_score(y_true, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, preds, average="macro", zero_division=0)
    
    print(f"  ==> {pipe_name} Ensemble Results:")
    print(f"      Accuracy: {acc*100:.2f}% | Macro F1: {f1_macro*100:.2f}%")
