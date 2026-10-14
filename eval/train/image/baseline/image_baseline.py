"""Kaggle job (CPU, internet): score GPTTrace's shipped image classifier
(Community Forensics ViT-S int8, as in js/engine/models.js) on samples of
public image benchmarks. Internal measurement only — most of these sets are
CC BY-NC; never train on them or publish their numbers.

Reads parquet row groups over HTTP (HfFileSystem) and zip members by range
request, so nothing is downloaded whole. Writes scores.parquet and
metrics.json to /kaggle/working."""
import io
import json
import random
import subprocess
import sys
import time
from collections import defaultdict

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime", "remotezip"], check=False)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image
import onnxruntime as ort

Image.MAX_IMAGE_PIXELS = None
OUT = "/kaggle/working"
PER_GROUP = 60          # images per (benchmark, generator) cell
rng = random.Random(0)
fs = HfFileSystem()

# ---------- model, mirroring js/engine/image/classifier.worker.js ----------
REV = "ac6ee457bea904a373065754107451793b56db00"
onnx_path = hf_hub_download("buildborderless/CommunityForensics-DeepfakeDet-ViT", "onnx/model_int8.onnx", revision=REV)
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
IN = sess.get_inputs()[0].name
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
EDGE, CROP = 440, 384


def tensor(img):
    a = (np.asarray(img, np.float32) / 255 - MEAN) / STD
    return a.transpose(2, 0, 1)[None]


def score_image(data):
    img = Image.open(io.BytesIO(data))
    fmt = img.format
    img = img.convert("RGB")
    w, h = img.size
    s = EDGE / min(w, h)
    side = CROP / s
    box = ((w - side) / 2, (h - side) / 2, (w + side) / 2, (h + side) / 2)
    centre = img.resize((CROP, CROP), Image.BICUBIC, box=box)
    batch = [tensor(centre)]
    if min(w, h) >= CROP * 2:
        for fx, fy in ((.25, .25), (.75, .25), (.25, .75), (.75, .75)):
            x, y = round(w * fx - CROP / 2), round(h * fy - CROP / 2)
            batch.append(tensor(img.crop((x, y, x + CROP, y + CROP))))
    logits = [float(sess.run(None, {IN: b})[0].ravel()[0]) for b in batch]
    return {"centre": logits[0], "crops_max": max(logits[1:]) if len(logits) > 1 else None, "w": w, "h": h, "fmt": fmt}


rows = []


def add(bench, gen, label, data, licence):
    try:
        r = score_image(data)
    except Exception as e:
        print("  skip", bench, gen, str(e)[:80], flush=True)
        return
    rows.append({"bench": bench, "gen": str(gen), "label": label, "licence": licence, **r})


# ---------- parquet sets: sample PER_GROUP rows per (label, generator) ----------
def parquet_bench(bench, files, img_col, gen_fn, label_fn, licence, max_files=None):
    t0 = time.time()
    files = sorted(files)
    rng.shuffle(files)
    files = files[:max_files] if max_files else files
    need = defaultdict(lambda: PER_GROUP)
    meta_cols = None
    for f in files:
        pf = pq.ParquetFile(fs.open(f))
        names = pf.schema_arrow.names
        meta_cols = [c for c in names if c != img_col and c not in ("mask", "prompt")]
        for g in rng.sample(range(pf.num_row_groups), pf.num_row_groups):
            meta = pf.read_row_group(g, columns=meta_cols).to_pandas()
            keys = [(label_fn(r), gen_fn(r)) for r in meta.itertuples(index=False)]
            if not any(need[k] > 0 for k in keys):
                continue
            imgs = pf.read_row_group(g, columns=[img_col]).column(0).to_pylist()
            for k, im in zip(keys, imgs):
                if need[k] <= 0:
                    continue
                data = im["bytes"] if isinstance(im, dict) else im
                if data:
                    add(bench, k[1], k[0], data, licence)
                    need[k] -= 1
        print(f"{bench}: {f.split('/')[-1]} done, {sum(1 for r in rows if r['bench'] == bench)} imgs, {time.time() - t0:.0f}s", flush=True)


def openfake():
    files = fs.glob("datasets/ComplexDataLab/OpenFake/core/test-*.parquet")
    parquet_bench("openfake", files, "image", lambda r: r.model if r.label != "real" else "real",
                  lambda r: "human" if str(r.label).lower() == "real" else "ai", "CC BY-NC 4.0", max_files=6)


def cfeval():
    files = fs.glob("datasets/OwensLab/CommunityForensics-Eval/data/*.parquet")
    parquet_bench("cf-eval", files, "image_data", lambda r: r.model_name,
                  lambda r: "human" if "real" in str(r.model_name).lower() else "ai", "CC BY-NC-SA 4.0", max_files=60)


def nanobanana():
    files = fs.glob("datasets/bitmind/nano-banana/data/*.parquet")
    parquet_bench("nano-banana", files, "image", lambda r: "nano-banana", lambda r: "ai", "MIT", max_files=1)


def aigibench():
    from remotezip import RemoteZip
    from huggingface_hub import hf_hub_url
    for gen in ["FLUX1-dev", "Imagen3", "Midjourney", "DALLE-3", "SD3", "SDXL", "SocialRF", "CommunityAI"]:
        url = hf_hub_url("HorizonTEL/AIGIBench", f"test/{gen}.zip", repo_type="dataset")
        try:
            with RemoteZip(url) as z:
                names = [n for n in z.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
                by = defaultdict(list)
                for n in names:
                    lab = "human" if "/0_real/" in n or "real" in n.lower().split("/")[-2] else "ai"
                    by[lab].append(n)
                for lab, ns in by.items():
                    for n in rng.sample(ns, min(PER_GROUP, len(ns))):
                        add("aigibench", gen, lab, z.read(n), "CC BY-NC-SA 4.0")
            print(f"aigibench {gen}: {dict((k, len(v)) for k, v in by.items())}", flush=True)
        except Exception as e:
            print("aigibench", gen, "failed:", str(e)[:200], flush=True)


# ---------- metrics ----------
def metrics(df):
    from sklearn.metrics import roc_auc_score
    out = {}
    for bench, g in df.groupby("bench"):
        m = {"n": len(g), "n_ai": int((g.label == "ai").sum()), "n_human": int((g.label == "human").sum()),
             "licence": g.licence.iloc[0]}
        y = (g.label == "ai").astype(int)
        if y.nunique() == 2:
            m["auc"] = round(float(roc_auc_score(y, g.centre)), 4)
            comb = g.centre.where(g.crops_max.isna(), np.maximum(g.centre, g.crops_max.fillna(-99)))
            m["auc_centre_or_cropmax"] = round(float(roc_auc_score(y, comb)), 4)
            m["acc_at_0"] = round(float(((g.centre > 0) == y.astype(bool)).mean()), 4)
            h = np.sort(g[g.label == "human"].centre.to_numpy())[::-1]
            thr = h[int(len(h) * 0.05)]
            m["tpr_at_5fpr"] = round(float((g[g.label == "ai"].centre > thr).mean()), 4)
        m["flagged_at_0"] = {k: round(float((v.centre > 0).mean()), 3) for k, v in g.groupby("gen")}
        out[bench] = m
    return out


if __name__ == "__main__":
    for fn in (nanobanana, openfake, aigibench, cfeval):
        try:
            fn()
        except Exception as e:
            print(fn.__name__, "FAILED", repr(e)[:300], flush=True)
        df = pd.DataFrame(rows)
        df.to_parquet(f"{OUT}/scores.parquet", index=False)
        json.dump(metrics(df), open(f"{OUT}/metrics.json", "w"), indent=1)
    print(json.dumps({k: {m: v.get(m) for m in ("n", "auc", "acc_at_0", "tpr_at_5fpr")} for k, v in metrics(pd.DataFrame(rows)).items()}, indent=1))
