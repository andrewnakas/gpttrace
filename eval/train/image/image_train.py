"""Kaggle job (GPU): fine-tune GPTTrace's image classifier (Community
Forensics ViT-S/384, MIT) on commercially usable data, export int8 ONNX.

Training data (licence): DRAGON train, 25 diffusion models (CC BY-SA 4.0);
open-image-preferences FLUX-dev / SD3.5 (Apache-2.0); ELSA1M track 1
(CC BY 4.0); DiffusionDB (CC0); nano-banana (MIT); megalith-small real
photos (CC0 via MIT repo; chunk 2 held out).
Held out and publishable: DRAGON test + unseen megalith chunk.
Internal only (CC BY-NC, never trained on): AIGIBench, OpenFake samples.

Every AI image is JPEG/WebP re-encoded before cropping and reals are
re-encoded half the time, so file format can't be learned as a shortcut."""
import io
import json
import os
import random
import subprocess
import sys
import tarfile
import time
from collections import defaultdict

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "remotezip"], check=False)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import HfFileSystem, hf_hub_download, hf_hub_url
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SMOKE = os.environ.get("GT_SMOKE", "0") == "1"
EPOCHS = int(os.environ.get("GT_EPOCHS", "2"))
LR = float(os.environ.get("GT_LR", "1e-5"))
OUT = "/kaggle/working"
REPO, REV = "buildborderless/CommunityForensics-DeepfakeDet-ViT", "ac6ee457bea904a373065754107451793b56db00"
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
EDGE, CROP = 440, 384
fs = HfFileSystem()
rng = random.Random(0)
dev = "cuda"
SCALE = 0.1 if SMOKE else 1.0


# ---------- data collection (bytes in memory, capped) ----------
_N = [0]


def shrink(data, maxside=1024):
    """Spill every image to /tmp so ~55k images fit (holding them in RAM got
    the run OOM-killed). JPEGs keep their original bytes unless downscaled
    (then q95); everything else is stored lossless (WebP) so AI pixels stay
    un-compressed until augmentation. Returns the file path."""
    os.makedirs("/tmp/imgs", exist_ok=True)
    im = Image.open(io.BytesIO(data))
    fmt = im.format
    _N[0] += 1
    if max(im.size) <= maxside and fmt == "JPEG":
        path = f"/tmp/imgs/{_N[0]}.jpg"
        open(path, "wb").write(data)
        return path
    im = im.convert("RGB")
    if max(im.size) > maxside:
        im.thumbnail((maxside, maxside), Image.LANCZOS)
    if fmt == "JPEG":
        path = f"/tmp/imgs/{_N[0]}.jpg"
        im.save(path, "JPEG", quality=95)
    else:
        path = f"/tmp/imgs/{_N[0]}.webp"
        im.save(path, "WEBP", lossless=True, method=1)
    return path


def open_img(x):
    return Image.open(x if isinstance(x, str) else io.BytesIO(x))


def from_tar(repo, files, n, gen_key="model.txt", img_key="png"):
    out = []
    for f in files:
        p = hf_hub_download(repo, f, repo_type="dataset")
        groups = defaultdict(dict)
        with tarfile.open(p) as t:
            for m in t.getmembers():
                if not m.isfile():
                    continue
                key, ext = m.name.rsplit(".", 1) if m.name.count(".") else (m.name, "")
                if m.name.endswith(".model.txt"):
                    key, ext = m.name[:-len(".model.txt")], "model.txt"
                elif m.name.endswith(".png"):
                    key, ext = m.name[:-4], "png"
                else:
                    continue
                groups[key][ext] = t.extractfile(m).read()
        for g in groups.values():
            if "png" in g:
                out.append((shrink(g["png"]), g.get("model.txt", b"?").decode().strip()))
        os.remove(p)
        print(repo, f, len(out), flush=True)
        if len(out) >= n:
            break
    return out[:n]


def from_parquet(pattern, n, col, gen, cap_files=40):
    files = sorted(fs.glob(pattern))
    rng.shuffle(files)
    out = []
    for f in files[:cap_files]:
        pf = pq.ParquetFile(fs.open(f))
        sch = pf.schema_arrow
        import pyarrow as pa
        image_cols = [fl.name for fl in sch if pa.types.is_binary(fl.type) or (pa.types.is_struct(fl.type) and "bytes" in [c.name for c in fl.type])]
        cols = image_cols if col == "*" else [col] if col in sch.names else image_cols[:1]
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=cols)
            for c in cols:
                for im in t.column(c).to_pylist():
                    data = im["bytes"] if isinstance(im, dict) else im
                    if data:
                        out.append((shrink(data), gen))
            if len(out) >= n:
                return out[:n]
    return out


def from_zip(repo, files, n, gen):
    out = []
    for f in files:
        # Download the part (range-request streaming got connection resets).
        p = hf_hub_download(repo, f, repo_type="dataset")
        import zipfile
        with zipfile.ZipFile(p) as z:
            for name in [x for x in z.namelist() if x.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]:
                out.append((shrink(z.read(name)), gen))
                if len(out) >= n:
                    break
        os.remove(p)
        if len(out) >= n:
            return out
    return out


def collect():
    n = lambda k: max(50, int(k * SCALE))
    ai, real = [], []
    def safe(name, fn):
        try:
            got = fn()
            print(name, len(got), flush=True)
            return got
        except Exception as e:
            print(name, "failed", repr(e)[:200], flush=True)
            return []
    ai += safe("dragon", lambda: from_tar("lesc-unifi/dragon", [f"train/dragon_train_00{i}.tar" for i in range(10)], n(12000)))
    ai += safe("open-image-preferences", lambda: from_parquet("datasets/data-is-better-together/open-image-preferences-v1/data/*.parquet", n(6000), "*", "flux/sd3.5-pref"))
    ai += safe("elsa1m", lambda: from_parquet("datasets/elsaEU/ELSA1M_track1/data/*.parquet", n(4000), "image", "elsa-sd"))
    ai += safe("diffusiondb", lambda: from_zip("poloclub/diffusiondb", [f"diffusiondb-large-part-1/part-00000{i}.zip" for i in range(1, 5)], n(3000), "diffusiondb-sd"))
    ai += safe("nano-banana", lambda: from_parquet("datasets/bitmind/nano-banana/data/*.parquet", n(2500), "image", "nano-banana"))
    ai += safe("kling", lambda: from_parquet("datasets/bitmind/klingai-images/**/*.parquet", n(2500), "image", "kling"))
    real += safe("megalith-0", lambda: from_parquet("datasets/bitmind/megalith-small/chunk_0000/*.parquet", n(10000), "image", "megalith"))
    real += safe("megalith-1", lambda: from_parquet("datasets/bitmind/megalith-small/chunk_0001/*.parquet", n(10000), "image", "megalith"))
    real += safe("megalith-3", lambda: from_parquet("datasets/bitmind/megalith-small/chunk_0003/*.parquet", n(10000), "image", "megalith"))
    print("collected ai", len(ai), "real", len(real), flush=True)
    return ai, real


def heldout():
    ai = from_tar("lesc-unifi/dragon", ["test/dragon_test_00.tar", "test/dragon_test_05.tar"], int(2000 * SCALE) or 100)
    real = from_parquet("datasets/bitmind/megalith-small/chunk_0002/*.parquet", int(2000 * SCALE) or 100, "image", "megalith")
    return ai, real


# ---------- augmentation & tensors (mirror js/engine/image/classifier.worker.js) ----------
def reencode(im):
    b = io.BytesIO()
    if rng.random() < 0.8:
        im.save(b, "JPEG", quality=rng.randint(65, 98))
    else:
        im.save(b, "WEBP", quality=rng.randint(70, 95))
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def train_view(data, is_ai):
    im = open_img(data).convert("RGB")
    if rng.random() < 0.3:  # random downscale (shared / resized uploads)
        s = rng.uniform(0.5, 0.9)
        im = im.resize((max(CROP, int(im.width * s)), max(CROP, int(im.height * s))), Image.BICUBIC)
    if is_ai or rng.random() < 0.5:
        im = reencode(im)
    w, h = im.size
    if min(w, h) >= CROP and rng.random() < 0.5:  # native-resolution crop
        x, y = rng.randint(0, w - CROP), rng.randint(0, h - CROP)
        im = im.crop((x, y, x + CROP, y + CROP))
    else:  # inference-style resize, random crop instead of centre
        s = EDGE / min(w, h)
        im = im.resize((max(CROP, round(w * s)), max(CROP, round(h * s))), Image.BICUBIC)
        x, y = rng.randint(0, im.width - CROP), rng.randint(0, im.height - CROP)
        im = im.crop((x, y, x + CROP, y + CROP))
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return tensor(im)


def tensor(im):
    a = (np.asarray(im, np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


def eval_views(data):
    """Centre view (shortest edge 440, centre 384) + up to 4 native crops."""
    im = open_img(data).convert("RGB")
    w, h = im.size
    s = EDGE / min(w, h)
    side = CROP / s
    views = [tensor(im.resize((CROP, CROP), Image.BICUBIC, box=((w - side) / 2, (h - side) / 2, (w + side) / 2, (h + side) / 2)))]
    if min(w, h) >= CROP * 2:
        for fx, fy in ((.25, .25), (.75, .25), (.25, .75), (.75, .75)):
            x, y = round(w * fx - CROP / 2), round(h * fy - CROP / 2)
            views.append(tensor(im.crop((x, y, x + CROP, y + CROP))))
    return views


class TrainDS(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        data, y = self.items[i]
        try:
            return train_view(data, y == 1), torch.tensor(float(y))
        except Exception:
            return self[(i + 1) % len(self)]


# ---------- model ----------
def load_model(weights="model.safetensors"):
    from transformers import ViTForImageClassification
    return ViTForImageClassification.from_pretrained(REPO, revision=REV)


@torch.no_grad()
def score(model, items):
    model.eval()
    out = []
    for data, *_ in items:
        try:
            v = torch.stack(eval_views(data)).to(dev)
        except Exception:
            out.append((np.nan, np.nan))
            continue
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(pixel_values=v).logits.float().view(-1).cpu().numpy()
        out.append((float(lg[0]), float(lg[1:].max()) if len(lg) > 1 else np.nan))
    return out


def metrics(rows):
    from sklearn.metrics import roc_auc_score
    df = pd.DataFrame(rows)
    res = {}
    for bench, g in df.groupby("bench"):
        g = g.dropna(subset=["centre"])
        y = (g.label == "ai").astype(int)
        m = {"n_ai": int(y.sum()), "n_human": int((1 - y).sum())}
        for col in ("base", "ft"):
            c = g[f"{col}_centre"]
            comb = np.maximum(c, g[f"{col}_crops"].fillna(-99))
            m[col] = {"auc": round(float(roc_auc_score(y, c)), 4), "auc_max_crop": round(float(roc_auc_score(y, comb)), 4),
                      "acc_at_0": round(float(((c > 0) == y.astype(bool)).mean()), 4),
                      "flagged_by_gen": {k: round(float((v[f"{col}_centre"] > 0).mean()), 3) for k, v in g.groupby("gen")}}
        res[bench] = m
    return res


# ---------- internal NC benchmarks (never trained on) ----------
def internal_bench():
    items = []
    try:
        from remotezip import RemoteZip
        for gen in ["FLUX1-dev", "Imagen3", "Midjourney", "DALLE-3", "SD3", "SDXL", "SocialRF"]:
            with RemoteZip(hf_hub_url("HorizonTEL/AIGIBench", f"test/{gen}.zip", repo_type="dataset")) as z:
                names = [x for x in z.namelist() if x.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
                by = defaultdict(list)
                for x in names:
                    by["human" if "real" in x.lower() else "ai"].append(x)
                for lab, ns in by.items():
                    for x in rng.sample(ns, min(int(60 * max(SCALE, .3)), len(ns))):
                        items.append((z.read(x), "aigibench", gen, lab))
    except Exception as e:
        print("aigibench failed", repr(e)[:200], flush=True)
    try:
        files = sorted(fs.glob("datasets/ComplexDataLab/OpenFake/core/test-*.parquet"))
        need = defaultdict(lambda: int(40 * max(SCALE, .3)))
        need["real"] = int(600 * max(SCALE, .3))
        for f in files[:5]:
            pf = pq.ParquetFile(fs.open(f))
            for gi in range(pf.num_row_groups):
                meta = pf.read_row_group(gi, columns=["label", "model"]).to_pandas()
                keys = ["real" if str(l).lower() == "real" else str(m) for l, m in zip(meta.label, meta.model)]
                if not any(need[k] > 0 for k in keys):
                    continue
                imgs = pf.read_row_group(gi, columns=["image"]).column(0).to_pylist()
                for k, im in zip(keys, imgs):
                    if need[k] > 0 and im and im.get("bytes"):
                        items.append((shrink(im["bytes"]), "openfake", k, "human" if k == "real" else "ai"))
                        need[k] -= 1
    except Exception as e:
        print("openfake failed", repr(e)[:200], flush=True)
    return items


def main():
    t0 = time.time()
    ai, real = collect()
    items = [(d, 1) for d, _ in ai] + [(d, 0) for d, _ in real]
    hai, hreal = heldout()
    evalset = [(d, "dragon+megalith", g, "ai") for d, g in hai] + [(d, "dragon+megalith", "megalith", "human") for d, _ in hreal]
    evalset += internal_bench()
    print(f"data ready in {time.time() - t0:.0f}s: train {len(items)}, eval {len(evalset)}", flush=True)

    model = load_model().to(dev)
    base_scores = score(model, evalset)

    pos = sum(y for _, y in items)
    w = [0.5 / pos if y else 0.5 / (len(items) - pos) for _, y in items]
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(items), replacement=True)
    dl = torch.utils.data.DataLoader(TrainDS(items), batch_size=32, sampler=sampler, num_workers=4, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler()
    for ep in range(EPOCHS):
        model.train()
        for i, (x, y) in enumerate(dl):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.binary_cross_entropy_with_logits(model(pixel_values=x.to(dev)).logits.float().view(-1), y.to(dev))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if i % 100 == 0:
                print(f"ep {ep} {i}/{len(dl)} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    ft_scores = score(model, evalset)
    rows = [{"bench": b, "gen": g, "label": l, "centre": 0, "base_centre": bs[0], "base_crops": bs[1], "ft_centre": fs_[0], "ft_crops": fs_[1]}
            for (_, b, g, l), bs, fs_ in zip(evalset, base_scores, ft_scores)]
    pd.DataFrame(rows).to_parquet(f"{OUT}/scores.parquet")
    m = {"epochs": EPOCHS, "lr": LR, "train": len(items), "train_ai": pos, "results": metrics(rows)}
    json.dump(m, open(f"{OUT}/metrics.json", "w"), indent=1)
    print(json.dumps({b: {k: v[k]["auc"] for k in ("base", "ft")} for b, v in m["results"].items()}), flush=True)
    export(model, m)


def export(model, m):
    model = model.float().cpu().eval()
    model.save_pretrained(f"{OUT}/model")
    os.makedirs(f"{OUT}/onnx", exist_ok=True)
    torch.onnx.export(model, (torch.randn(1, 3, CROP, CROP),), "/tmp/model.onnx", input_names=["pixel_values"], output_names=["logits"],
                      dynamic_axes={"pixel_values": {0: "b"}, "logits": {0: "b"}}, opset_version=17, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic("/tmp/model.onnx", f"{OUT}/onnx/model_int8.onnx", weight_type=QuantType.QInt8)
    import onnxruntime as ort
    sess = ort.InferenceSession(f"{OUT}/onnx/model_int8.onnx", providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, CROP, CROP)
    with torch.no_grad():
        ref = model(pixel_values=x).logits.view(-1).numpy()
    q = np.array([sess.run(None, {"pixel_values": x[i:i + 1].numpy()})[0].ravel()[0] for i in range(4)])
    m["export"] = {"bytes_int8": os.path.getsize(f"{OUT}/onnx/model_int8.onnx"), "int8_vs_fp32_max_abs": float(np.abs(q - ref).max())}
    json.dump(m, open(f"{OUT}/metrics.json", "w"), indent=1)
    print("export", m["export"], flush=True)


if __name__ == "__main__":
    main()
