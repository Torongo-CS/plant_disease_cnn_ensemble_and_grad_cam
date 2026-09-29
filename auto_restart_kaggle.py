import os
import sys
import json
import time
import shutil
import io
import subprocess

# Set UTF-8 encoding for Windows stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

KERNEL_NAME = "torongocs/plant-disease-cnn-ensemble-training"
DATASET_NAME = "torongocs/plant-disease-pretrained-models"
ALL_MODELS = ["EfficientNet_B0", "ResNet50", "DenseNet121", "MobileNetV3_Large", "ShuffleNetV2"]

def log_msg(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def sync_completed_models_to_pretrained_dataset():
    updated = False
    output_models_dir = os.path.join("kaggle_output", "models")
    pretrained_models_dir = os.path.join("pretrained_dataset", "models")
    os.makedirs(pretrained_models_dir, exist_ok=True)

    if not os.path.exists(output_models_dir):
        return False

    for m in ALL_MODELS:
        src = os.path.join(output_models_dir, m)
        dst = os.path.join(pretrained_models_dir, m)
        if os.path.exists(src):
            for fname in os.listdir(src):
                s_file = os.path.join(src, fname)
                d_file = os.path.join(dst, fname)
                if os.path.isfile(s_file):
                    if not os.path.exists(d_file) or os.path.getmtime(s_file) > os.path.getmtime(d_file):
                        os.makedirs(dst, exist_ok=True)
                        shutil.copy2(s_file, d_file)
                        log_msg(f"Checkpoint file updated for {m}: {fname}")
                        updated = True

    if updated:
        log_msg("Uploading updated pretrained_dataset with per-epoch checkpoints to Kaggle...")
        try:
            res = subprocess.run(
                [sys.executable, "-m", "kaggle", "datasets", "version", "-p", "pretrained_dataset", "-m", "Auto-checkpoint sync", "-r", "zip"],
                capture_output=True, text=True, timeout=120
            )
            log_msg(f"Dataset upload output: {res.stdout.strip()}")
            return True
        except Exception as e:
            log_msg(f"Dataset update error: {e}")
    return False

def get_kernel_status():
    try:
        res = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "status", KERNEL_NAME],
            capture_output=True, text=True, timeout=30
        )
        out = res.stdout.strip()
        if "KernelWorkerStatus.RUNNING" in out:
            return "RUNNING"
        elif "KernelWorkerStatus.QUEUED" in out:
            return "QUEUED"
        elif "KernelWorkerStatus.COMPLETE" in out:
            return "COMPLETE"
        elif "KernelWorkerStatus.FAILED" in out:
            return "FAILED"
        elif "KernelWorkerStatus.CANCELLED" in out:
            return "CANCELLED"
        elif "KernelWorkerStatus.ERROR" in out:
            return "ERROR"
        else:
            return out
    except Exception as e:
        log_msg(f"Status check exception: {e}")
        return "UNKNOWN"

def download_outputs():
    try:
        subprocess.run(
            [sys.executable, "download_kaggle_outputs.py"],
            capture_output=True, text=True, timeout=60
        )
    except Exception as e:
        pass

def restart_kernel():
    log_msg("Auto-restarting Kaggle kernel with latest checkpoints...")
    try:
        res = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "push", "-p", "kaggle_kernel"],
            capture_output=True, text=True, timeout=60
        )
        log_msg(f"Kernel push response: {res.stdout.strip()}")
    except Exception as e:
        log_msg(f"Kernel push error: {e}")

def run_automation_loop(sleep_sec=60):
    log_msg("=== Starting Kaggle Auto-Restart & Checkpoint Sync Daemon ===")
    iteration = 0
    
    while True:
        iteration += 1
        status = get_kernel_status()
        log_msg(f"Iteration {iteration} | Kernel Status: {status}")
        
        # 1. Download any available output files
        download_outputs()
        
        # 2. Sync any newly completed models to Kaggle dataset
        dataset_updated = sync_completed_models_to_pretrained_dataset()
        
        # 3. Check if all 5 models are finished locally
        output_models_dir = os.path.join("kaggle_output", "models")
        completed_models = []
        if os.path.exists(output_models_dir):
            for m in ALL_MODELS:
                mf = os.path.join(output_models_dir, m, "metrics.json")
                if os.path.exists(mf):
                    completed_models.append(m)
                
        log_msg(f"Completed Models ({len(completed_models)}/5): {completed_models}")
        
        if len(completed_models) == 5:
            log_msg("🎉 ALL 5 MODELS ARE FULLY TRAINED & EVALUATED! Stopping Daemon.")
            break

        # 4. Handle Stopped / Failed / Error status by auto-restarting
        if status in ["FAILED", "CANCELLED", "ERROR"]:
            log_msg(f"⚠️ Detected kernel interruption state ({status}). Triggering auto-restart...")
            if dataset_updated:
                log_msg("Waiting 20s for Kaggle to register updated dataset version...")
                time.sleep(20)
            restart_kernel()
        elif status == "COMPLETE" and len(completed_models) < 5:
            log_msg("⚠️ Kernel completed step but remaining models pending. Restarting for next models...")
            if dataset_updated:
                log_msg("Waiting 20s for Kaggle to register updated dataset version...")
                time.sleep(20)
            restart_kernel()
        elif status in ["RUNNING", "QUEUED"]:
            log_msg("Kernel is actively processing on Kaggle GPUs. Monitoring...")

        time.sleep(sleep_sec)

if __name__ == "__main__":
    run_automation_loop()
