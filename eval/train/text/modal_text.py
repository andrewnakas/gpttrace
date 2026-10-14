"""Text-model training on Modal A100s (the Kaggle scripts, unchanged).

    ~/.cache/gpttrace/venv/bin/modal run --detach eval/train/text/modal_text.py --tag v4

Steps (outputs persist in the Modal volume "gpttrace-text"):
  prep   CPU   RAID train/val/test sample (raid-data/raid_data.py, same split)
  gen    A100  topic-matched human/AI pairs (gen/gen_pairs.py)
  train  A100  fine-tune + RAID test predictions + int8 ONNX (train_raid.py)
The Kaggle paths those scripts expect (/kaggle/input, /kaggle/working) are
symlinked into the volume, so the code is identical on both platforms.

Budget: the user's free Modal credit (A100-40GB ≈ $2/h); stay under ~$25/month.
Fetch results: modal volume get gpttrace-text out-<tag>/metrics.json ."""
import os
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).parent
vol = modal.Volume.from_name("gpttrace-text", create_if_missing=True)
V = "/vol"

base = (modal.Image.debian_slim(python_version="3.11")
        .apt_install("wget")
        .pip_install("torch==2.5.1", "transformers==4.48.3", "datasets", "pandas", "pyarrow", "scikit-learn",
                     "onnx", "onnxruntime", "accelerate", "huggingface_hub")
        .add_local_file(HERE / "normalize.py", "/src/normalize.py")
        .add_local_file(HERE / "train_raid.py", "/src/train_raid.py")
        .add_local_file(HERE / "raid-data/raid_data.py", "/src/raid_data.py"))
gen_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("vllm", "pandas", "pyarrow", "huggingface_hub")
             .add_local_file(HERE / "gen/gen_pairs.py", "/src/gen_pairs.py"))
app = modal.App("gpttrace-text")


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.makedirs(src, exist_ok=True)
    if not os.path.exists(dst):
        os.symlink(src, dst)


@app.function(image=base, volumes={V: vol}, cpu=4, memory=32768, ephemeral_disk=100_000, timeout=4 * 3600)
def prep():
    if os.path.exists(f"{V}/raid/raid_test.parquet"):
        return "raid sample already in volume"
    link(f"{V}/raid", "/kaggle/working")
    subprocess.run(["python", "/src/raid_data.py"], check=True)
    vol.commit()
    return "raid sample written"


@app.function(image=gen_image, volumes={V: vol}, gpu="A100-40GB", timeout=3 * 3600)
def gen(n_per_source: int = 3000):
    if os.path.exists(f"{V}/pairs/pairs.parquet"):
        return "pairs already in volume"
    os.makedirs(f"{V}/pairs", exist_ok=True)
    env = dict(os.environ, GT_OUT=f"{V}/pairs", GT_TP="1", GT_NO_PIP="1", GT_N=str(n_per_source))
    subprocess.run(["python", "/src/gen_pairs.py"], check=True, env=env)
    vol.commit()
    return "pairs written"


@app.function(image=base, volumes={V: vol}, gpu="A100-40GB", memory=65536, ephemeral_disk=100_000, timeout=8 * 3600)
def train(tag: str, base_model: str = "jhu-clsp/ettin-encoder-150m", rows: int = 641520, lr: float = 5e-5, mage: int = 200000, smoke: bool = False):
    out = f"{V}/out-{tag}"
    link(f"{V}/raid", "/kaggle/input/raid")
    link(f"{V}/pairs", "/kaggle/input/pairs")
    link(out, "/kaggle/working")
    src = open("/src/normalize.py").read() + "\n" + open("/src/train_raid.py").read()
    for k, v in {"__SMOKE__": "1" if smoke else "0", "__BASE__": base_model, "__ROWS__": str(rows), "__LR__": str(lr), "__MAGE__": str(mage)}.items():
        src = src.replace(k, v)
    open("/tmp/kernel.py", "w").write(src)
    try:
        subprocess.run(["python", "/tmp/kernel.py"], check=True)
    finally:
        vol.commit()
    return f"done: {out}"


@app.local_entrypoint()
def main(tag: str = "v4", n_per_source: int = 3000, smoke: bool = False, data_only: bool = False):
    p, g = prep.spawn(), gen.spawn(n_per_source)
    print(p.get())
    print(g.get())
    if not data_only:
        print(train.remote(tag, smoke=smoke))
