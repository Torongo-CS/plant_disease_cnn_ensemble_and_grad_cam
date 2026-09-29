import os
import re
import html
import requests
from PIL import Image
from io import BytesIO

REMAINING_CLASSES = {
    "Potato___Early_blight": [
        "potato early blight",
        "early blight on potato",
        "alternaria solani potato"
    ],
    "Tomato___Bacterial_spot": [
        "tomato bacterial spot",
        "bacterial spot tomato leaf",
        "xanthomonas tomato"
    ],
    "Tomato___Late_blight": [
        "tomato late blight",
        "late blight on tomato",
        "phytophthora infestans tomato"
    ],
    "Tomato___Target_Spot": [
        "tomato target spot",
        "target spot tomato leaf",
        "corynespora tomato"
    ]
}

OUTPUT_BASE = r"c:\Users\toron\OneDrive\Desktop\ML_FRESH\real_world_test_dataset"
os.makedirs(OUTPUT_BASE, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}

def fetch_real_image_urls(query, max_urls=30):
    url = f"https://www.bing.com/images/async?q={requests.utils.quote(query)}&first=1&count=40&relpath=4"
    image_urls = []
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            matches = re.findall(r'murl&quot;:&quot;(http[s]?://[^&]+?)&quot;', resp.text, re.IGNORECASE)
            for m in matches:
                clean_url = html.unescape(m)
                if clean_url not in image_urls and not clean_url.endswith(('.svg', '.gif')):
                    image_urls.append(clean_url)
                if len(image_urls) >= max_urls:
                    break
    except Exception as e:
        print(f"Error fetching images for '{query}': {e}")
    return image_urls

print("Fetching images for remaining 4 classes...")

for cls_name, queries in REMAINING_CLASSES.items():
    cls_dir = os.path.join(OUTPUT_BASE, cls_name)
    os.makedirs(cls_dir, exist_ok=True)
    
    existing = len([f for f in os.listdir(cls_dir) if f.startswith('real_web_')])
    target_count = 15
    
    print(f"\nSearching for {cls_name}...")
    downloaded = existing
    
    for q in queries:
        if downloaded >= target_count:
            break
        urls = fetch_real_image_urls(q, max_urls=25)
        for img_url in urls:
            if downloaded >= target_count:
                break
            try:
                r = requests.get(img_url, headers=headers, timeout=5)
                if r.status_code == 200:
                    img = Image.open(BytesIO(r.content)).convert("RGB")
                    if img.width >= 100 and img.height >= 100:
                        save_path = os.path.join(cls_dir, f"real_web_{downloaded + 1}.jpg")
                        img.save(save_path, "JPEG")
                        downloaded += 1
                        print(f"  [OK] Saved real web image #{downloaded} for {cls_name}")
            except Exception:
                continue

    print(f"[{cls_name}] Total real web images saved: {downloaded}")

print("\nDone filling remaining classes!")
