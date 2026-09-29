import os
import sys
import io

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

kernel_ref = "torongocs/plant-disease-cnn-ensemble-training"
download_path = r"c:\Users\toron\OneDrive\Desktop\ML_FRESH\kaggle_output"

print(f"Downloading output files for kernel: {kernel_ref}...")
try:
    api.kernels_output_cli(kernel_ref, path=download_path, force=True)
    print("Download completed successfully!")
except Exception as e:
    print(f"Download completed with notice: {e}")
