"""Colab driver for the text-detector training run (paid A100/L4 or free T4),
via eval/train/queue/colab_job.sh. Files uploaded alongside it:
  /content/kernel.py    built by `GPU=1 DRY=1 JOBS=… sh eval/train/text/push_tpu.sh …`
  /content/raid_data.py eval/train/text/raid-data/raid_data.py
  /content/pairs.parquet   (optional) the paraphrase/humanizer pairs
1. Builds the RAID splits from the public RAID files (~7 minutes).
2. Runs the same training script as Kaggle, in GPU mode, with Colab paths.
Two-stage runs keep the GPU bill to GPU work: GT_STAGE=prep on a CPU machine
tokenises and publishes to the private dataset GT_PREP_REPO; GT_STAGE=train
on the A100 fetches it (seconds, inside Google's network) and trains.
Outputs land in /content/out/<job tag>/ (model, metrics.json, onnx/)."""
import os
import subprocess
import sys

os.makedirs("/content/input/raid", exist_ok=True)
os.makedirs("/content/out", exist_ok=True)
if os.path.exists("/content/pairs.parquet"):              # uploaded next to the script
    os.makedirs("/content/input/pairs", exist_ok=True)
    os.replace("/content/pairs.parquet", "/content/input/pairs/pairs.parquet")
if os.environ.get("GT_STAGE") not in ("train", "teacher"):   # these stages get their data from the prep repo
    subprocess.run([sys.executable, "-u", "/content/raid_data.py"], check=True, env={**os.environ, "GT_RAID_OUT": "/content/input/raid"})
    os.remove("/content/input/raid/raid_test.parquet") if os.path.exists("/content/input/raid/raid_test.parquet") else None
env = {**os.environ, "GT_DEVICE": "cuda", "GT_INPUT": "/content/input", "GT_WORK": "/content/out", "GT_BUDGET_H": os.environ.get("GT_BUDGET_H", "20")}
subprocess.run([sys.executable, "-u", "/content/kernel.py"], check=True, env=env)

# Big weights go straight to private Hugging Face repos from the datacentre
# (a 1B model is ~4 GB; the home uplink would take hours while the GPU bills).
# Only metrics and logs stay in /content/out for the trip home.
tok = os.environ.get("HF_TOKEN")
if tok:
    import glob, shutil
    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    for d in sorted(glob.glob("/content/out/*/")):
        tag = os.path.basename(d.rstrip("/"))
        if not (os.path.isdir(f"{d}model") or os.path.isdir(f"{d}onnx")):
            continue
        repo = f"nakas/gpttrace-{tag}"
        api.create_repo(repo, private=True, exist_ok=True)
        for sub in ("model", "onnx"):
            if os.path.isdir(f"{d}{sub}"):
                api.upload_folder(repo_id=repo, folder_path=f"{d}{sub}", path_in_repo=sub, commit_message=f"{tag}: {sub}")
                shutil.rmtree(f"{d}{sub}")
        print("uploaded", tag, "to", repo, "(private)", flush=True)
