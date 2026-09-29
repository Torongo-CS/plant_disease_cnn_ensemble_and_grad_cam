import os
import sys
import json
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Grad-CAM] Using device: {device}")

# 1. Load Classes
with open("classes.json", "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
num_classes = len(class_names)

# Image preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to instantiate model architecture
def get_model(model_name):
    if model_name == "EfficientNet_B0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        target_layer = model.features[-1]
    elif model_name == "ResNet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        target_layer = model.layer4[-1]
    elif model_name == "DenseNet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        target_layer = model.features.norm5
    elif model_name == "MobileNetV3_Large":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        target_layer = model.features[-1]
    elif model_name == "ShuffleNetV2":
        model = models.shufflenet_v2_x1_0(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        target_layer = model.conv5
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model, target_layer

# GradCAM Implementation Class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.to(device)
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        self.activations = output

    def generate(self, input_tensor, target_class=None):
        input_tensor = input_tensor.to(device)
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        target = output[0, target_class]
        
        # Compute gradients of output w.r.t target layer activations using autograd.grad
        grads = torch.autograd.grad(target, self.activations, retain_graph=True)[0]

        gradients = grads.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, target_class

def overlay_heatmap(img_pil, cam_map):
    img_np = np.array(img_pil.resize((224, 224)))
    cam_resized = cv2.resize(cam_map, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed = heatmap * 0.4 + img_np * 0.6
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    return img_np, heatmap, superimposed

def main():
    out_dir = "gradcam_results"
    os.makedirs(out_dir, exist_ok=True)

    models_to_test = ["DenseNet121", "EfficientNet_B0", "MobileNetV3_Large", "ResNet50", "ShuffleNetV2"]
    
    # Collect sample test images (1 per class)
    sample_images = {}
    for cls in class_names:
        cls_dir = os.path.join("Selected_dataset", cls)
        if os.path.exists(cls_dir):
            files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.png', '.JPG'))]
            if files:
                sample_images[cls] = files[0]

    print(f"[Grad-CAM] Found {len(sample_images)} sample images for visualization.")

    for model_name in models_to_test:
        weights_path = os.path.join("pretrained_dataset", "models", model_name, "best.pth")
        if not os.path.exists(weights_path):
            weights_path = os.path.join("kaggle_output", "models", model_name, "best.pth")

        if not os.path.exists(weights_path):
            print(f"[Grad-CAM] Weights for {model_name} not found, skipping...")
            continue

        print(f"\n[Grad-CAM] Generating heatmaps for model: {model_name}...")
        model, target_layer = get_model(model_name)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        grad_cam = GradCAM(model, target_layer)

        model_out_dir = os.path.join(out_dir, model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        fig, axes = plt.subplots(len(sample_images), 3, figsize=(10, 3 * len(sample_images)))
        fig.suptitle(f"Grad-CAM Heatmaps - {model_name}", fontsize=14, fontweight='bold')

        for idx, (cls, img_path) in enumerate(sample_images.items()):
            img_pil = Image.open(img_path).convert("RGB")
            input_tensor = transform(img_pil).unsqueeze(0)

            cam_map, pred_class_idx = grad_cam.generate(input_tensor)
            pred_class_name = idx_to_class[pred_class_idx]

            orig_np, heatmap, overlay = overlay_heatmap(img_pil, cam_map)

            # Single row visualization save
            cv2.imwrite(os.path.join(model_out_dir, f"{cls}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            axes[idx, 0].imshow(orig_np)
            axes[idx, 0].set_title(f"Original: {cls[:20]}", fontsize=8)
            axes[idx, 0].axis("off")

            axes[idx, 1].imshow(heatmap)
            axes[idx, 1].set_title("Grad-CAM Heatmap", fontsize=8)
            axes[idx, 1].axis("off")

            axes[idx, 2].imshow(overlay)
            axes[idx, 2].set_title(f"Pred: {pred_class_name[:20]}", fontsize=8)
            axes[idx, 2].axis("off")

        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"{model_name}_gradcam_summary.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  --> Saved summary plot: {plot_path}")

    print("\nGrad-CAM generation finished successfully!")

if __name__ == "__main__":
    main()
