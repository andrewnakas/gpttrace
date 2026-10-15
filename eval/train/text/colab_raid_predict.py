"""RAID leaderboard predictions on Colab for a model in a private HF repo.
Uploaded alongside: /content/kernel.py (the GPU training kernel, used for its
normalisation and bucketed scoring) and /content/raid_data.py.
  env: GT_MODEL_REPO=nakas/gpttrace-v5-1b  HF_TOKEN=…
→ /content/out/predictions.json  [{id, score}] with score = P(AI), the same
format as the merged leaderboard entry (liamdugan/raid#220)."""
import importlib.util
import json
import os
import subprocess
import sys

os.makedirs("/content/input/raid", exist_ok=True)
os.makedirs("/content/out", exist_ok=True)
subprocess.run([sys.executable, "-u", "/content/raid_data.py"], check=True, env={**os.environ, "GT_RAID_OUT": "/content/input/raid"})
os.environ.update({"GT_DEVICE": "cuda", "GT_INPUT": "/content/input"})
spec = importlib.util.spec_from_file_location("kernel", "/content/kernel.py")
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

d = snapshot_download(os.environ["GT_MODEL_REPO"], allow_patterns=["model/*"], token=os.environ.get("HF_TOKEN"))
tok = AutoTokenizer.from_pretrained(f"{d}/model")
model = AutoModelForSequenceClassification.from_pretrained(f"{d}/model", attn_implementation="eager").to("cuda").eval()
te = k.load("test")
print("test rows", len(te), flush=True)
logits = k.score(model, tok, te.generation.map(k.normalize_text).tolist(), bs=128)
preds = [{"id": i, "score": float(1 / (1 + np.exp(-l)))} for i, l in zip(te.id, logits)]
json.dump(preds, open("/content/out/predictions.json", "w"))
print("predictions", len(preds), "mean score", round(float(np.mean([p["score"] for p in preds])), 4), flush=True)
