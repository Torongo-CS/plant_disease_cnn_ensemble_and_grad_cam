import os
import time
import urllib.request
from PIL import Image
from io import BytesIO

# Direct open-access image URLs from Wikimedia Commons
DIRECT_IMAGE_URLS = {
    "Tomato___Late_blight": [
        "https://upload.wikimedia.org/wikipedia/commons/0/04/Tomato_late_blight_leaf_curl_2_%285816171413%29.jpg"
    ],
    "Potato___Late_blight": [
        "https://upload.wikimedia.org/wikipedia/commons/1/1b/Phytophthora_infestans_potato_leaf_2.jpg"
    ],
    "Potato___Early_blight": [
        "https://upload.wikimedia.org/wikipedia/commons/7/7b/Alternaria_solani_potato_leaf.jpg"
    ],
    "Potato___healthy": [
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Solanum_tuberosum_leaf.jpg"
    ],
    "Tomato___Early_blight": [
        "https://upload.wikimedia.org/wikipedia/commons/6/6f/Tomato_early_blight_leaf.jpg"
    ],
    "Tomato___healthy": [
        "https://upload.wikimedia.org/wikipedia/commons/e/e9/Tomato_leaf.jpg"
    ],
    "Tomato___Bacterial_spot": [
        "https://upload.wikimedia.org/wikipedia/commons/5/52/Bacterial_spot_tomato.jpg"
    ],
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": [
        "https://upload.wikimedia.org/wikipedia/commons/3/30/TYLCV_tomato.jpg"
    ]
}

OUTPUT_BASE = r"c:\Users\toron\OneDrive\Desktop\ML_FRESH\real_world_test_dataset"
os.makedirs(OUTPUT_BASE, exist_ok=True)

# Wikimedia requires a descriptive User-Agent with contact email
headers = {
    "User-Agent": "PlantDiseaseResearchBot/1.0 (educational_research@example.com)"
}

print("==================================================")
print(" Downloading Direct Real-World Field Leaf Images (Polite Crawler)")
print("==================================================")

total_saved = 0

for cls_name, urls in DIRECT_IMAGE_URLS.items():
    cls_dir = os.path.join(OUTPUT_BASE, cls_name)
    os.makedirs(cls_dir, exist_ok=True)
    
    for idx, url in enumerate(urls, 1):
        save_path = os.path.join(cls_dir, f"field_direct_{idx}.jpg")
        try:
            req = urllib.request.Request(url, headers=headers)
            time.sleep(1.5)  # Respect Wikimedia rate limits
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                img = Image.open(BytesIO(data)).convert("RGB")
                img.save(save_path, "JPEG")
                total_saved += 1
                print(f"  [OK] Saved {cls_name} image #{idx}")
        except Exception as e:
            print(f"  [ERROR] Failed {cls_name}: {e}")

print("\n==================================================")
print(f" Download Complete! Total Direct Field Images Saved: {total_saved}")
print("==================================================")
