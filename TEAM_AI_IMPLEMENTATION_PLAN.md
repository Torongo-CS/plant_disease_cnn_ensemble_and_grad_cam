# 📘 Team Implementation Blueprint: Domain-Adapted Deep CNN Ensemble for Plant Disease Diagnosis

**Target Audience**: Project Teammates & AI Coding Assistants  
**Project Goal**: Train 5 deep CNN models, perform domain adaptation to eliminate background bias, and evaluate a winning soft-voting ensemble on out-of-distribution real-world field stress test images.

---

## 📑 Executive Summary of the 3-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Primary Baseline Training                                      │
│ • Dataset: Selected_dataset (Unsegmented Lab Benchmark)                 │
│ • Models: DenseNet121, EfficientNet_B0, MobileNetV3_Large,              │
│           ResNet50, ShuffleNetV2                                        │
│ • EXACT EPOCHS: 30 Epochs per model                                     │
│ • Output: Primary weights (best.pth)                                    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Domain Adaptation & Segmented Fine-Tuning                      │
│ • Dataset: Segmented_dataset (20,312 Masked Leaf Images)                │
│ • Warm-Start Initializer: Stage 1 best.pth weights                      │
│ • Augmentations: ColorJitter + RandomResizedCrop +                      │
│                  RandomRotation + RandomErasing                         │
│ • EXACT EPOCHS: 5 Fine-Tuning Epochs per model                          │
│ • Output: Domain-Adapted Weights (best_segmented.pth)                   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Inference-Time Leaf Cropping & Field Stress Testing            │
│ • Preprocessing: Automatic HSV Leaf ROI Bounding-Box Background Removal │
│ • Dataset: Stress_test (868 Out-of-Distribution Field Images)           │
│ • Ensemble: Soft-Voting (DenseNet121 + EfficientNet_B0 + MobileNetV3)   │
│ • EXACT EPOCHS: 0 Epochs (Inference Only)                               │
│ • Output: Publication Tables, Confusion Matrices, CSVs, JSONs           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

> [!IMPORTANT]
> **NOTE FOR TEAMMATES & AI ASSISTANTS**: 
> **Stage 1 (30 Epochs) is ALREADY 100% COMPLETED!** All 30-epoch pre-trained weights (`best.pth`) are already saved and available in `kaggle_output/models/` and `pretrained_dataset/models/`. 
> **DO NOT re-train for 30 epochs!** Teammates only need to run **Stage 2 (5 fine-tuning epochs)** and **Stage 3 (Inference on Stress_test)**.

## 🛠️ Stage 1: Primary Baseline Training (ALREADY COMPLETED - SKIP THIS STEP)

### Specifications:
* **Input Dataset**: [`Selected_dataset`](file:///c:/Users/toron/OneDrive/Desktop/ML_FRESH/Selected_dataset) (13 Plant Disease & Healthy Classes).
* **Train / Val / Test Split**: 70% Train, 15% Val, 15% Test (Stratified, Seed=42).
* **Architectures**:
  1. `DenseNet121`
  2. `EfficientNet_B0`
  3. `MobileNetV3_Large`
  4. `ResNet50`
  5. `ShuffleNetV2`
* **Exact Epoch Count**: **30 Epochs per model**.
* **Optimizer**: AdamW (`lr=1e-4`, `weight_decay=1e-4`).
* **Saved Artifact**: `kaggle_output/models/{ModelName}/best.pth`.

---

## 🌿 Stage 2: Domain Adaptation Fine-Tuning

### Specifications:
* **Input Dataset**: [`Segmented_dataset`](file:///c:/Users/toron/OneDrive/Desktop/ML_FRESH/Segmented_dataset) (20,312 images with background pixels masked to zero).
* **Warm-Start Initializer**: Load `best.pth` from Stage 1.
* **Exact Epoch Count**: **5 Fine-Tuning Epochs per model**.
* **Domain Augmentation Pipeline**:
  ```python
  transforms.Compose([
      transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(30),
      transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
  ])
  ```
* **Saved Artifact**: `segmented_models/{ModelName}/best_segmented.pth`.

---

## 🔬 Stage 3: Inference-Time Leaf Cropping & Stress Testing

### Specifications:
* **Input Dataset**: [`Stress_test`](file:///c:/Users/toron/OneDrive/Desktop/ML_FRESH/Stress_test) (868 out-of-distribution real-world field photographs across 10 disease categories).
* **Inference-Time Leaf Cropping Preprocessing**:
  Before passing any field photo to the models, automatically segment the leaf ROI and crop out non-leaf background clutter (soil, weeds, shadows).
* **Exact Epoch Count**: **0 Epochs (Inference Only)**.
* **Winning Soft-Voting Ensemble Combination**:
  $$\text{Winning Ensemble} = \text{DenseNet121} + \text{EfficientNet\_B0} + \text{MobileNetV3\_Large}$$
  $$\hat{y} = \arg\max_k \left( \frac{P_{\text{DenseNet}}(C_k) + P_{\text{EfficientNet}}(C_k) + P_{\text{MobileNet}}(C_k)}{3} \right)$$

---

## 🐍 Reference Python Implementation Code for AI Assistants

AI assistants can directly execute the following standalone reference code:

### 1. Automatic Leaf ROI Segmentation & Bounding-Box Cropping Function
```python
import cv2
import numpy as np
from PIL import Image

def preprocess_and_crop_leaf_roi(img_path):
    """
    Automated OpenCV HSV Thresholding & Contour Bounding-Box Cropping.
    Segments leaf tissue and removes background soil/clutter.
    """
    img_np = cv2.imread(img_path)
    if img_np is None:
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)[:, :, ::-1]

    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)

    # HSV bounds for healthy green and diseased yellow/brown tissue
    lower_green = np.array([20, 20, 20])
    upper_green = np.array([95, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    lower_brown = np.array([10, 30, 20])
    upper_brown = np.array([25, 255, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    combined_mask = cv2.bitwise_or(mask_green, mask_brown)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

    # Bounding Box Cropping around largest leaf contour
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w > 20 and h > 20:
            pad_w = int(w * 0.05)
            pad_h = int(h * 0.05)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(img_rgb.shape[1], x + w + pad_w)
            y2 = min(img_rgb.shape[0], y + h + pad_h)
            img_rgb = img_rgb[y1:y2, x1:x2]
            clean_mask = clean_mask[y1:y2, x1:x2]

    # Zero-out non-leaf background pixels
    mask_3d = np.repeat(clean_mask[:, :, np.newaxis], 3, axis=2) / 255.0
    segmented_rgb = (img_rgb * mask_3d).astype(np.uint8)

    return Image.fromarray(segmented_rgb)
```

### 2. Triplet Soft-Voting Ensemble Prediction Function
```python
import torch
import torch.nn.functional as F

def predict_winning_ensemble(models_dict, input_tensor):
    """
    Computes Soft-Voting Ensemble Probability Averaging across:
    DenseNet121 + EfficientNet_B0 + MobileNetV3_Large
    """
    winning_models = ["DenseNet121", "EfficientNet_B0", "MobileNetV3_Large"]
    probs_list = []

    with torch.no_grad():
        for m_name in winning_models:
            model = models_dict[m_name]
            model.eval()
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs)

    # Equal weight soft-voting ensemble average
    ensemble_probs = torch.stack(probs_list, dim=0).mean(dim=0)
    pred_class = torch.argmax(ensemble_probs, dim=1)
    return pred_class, ensemble_probs
```

---

## 📊 Summary Checklist for Teammates

| Task | Target Dataset | Epoch Count | Priority | Status |
| :--- | :--- | :---: | :---: | :---: |
| 1. Primary Pre-training (5 Models) | `Selected_dataset` | **30 Epochs** | High | **ALREADY COMPLETED (DO NOT RE-RUN)** |
| 2. 10-Combo Soft-Voting Search | Validation Split | **0 Epochs** | High | Completed (`DenseNet` + `EfficientNet` + `MobileNetV3`) |
| 3. Grad-CAM Visualization | Sample Test Images | **0 Epochs** | Medium | Completed (Saved in `gradcam_results/`) |
| 4. Segmented Fine-Tuning | `Segmented_dataset` | **5 Epochs** | High | Ready for execution |
| 5. Leaf ROI Background Cropping | `Stress_test` | **0 Epochs** | High | Ready for execution |
