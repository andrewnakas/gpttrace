"""Kaggle job (GPU): fine-tune a small speech encoder as GPTTrace's neural
audio deepfake check, measure EER on public benchmarks, export int8 ONNX.

Train: ASVspoof 2019 LA train (ODC-By). Eval: ASVspoof 2019 LA eval,
ASVspoof 2021 DF eval (ODC-ODbL; keys from asvspoof.org), In-the-Wild
(CC BY-SA 4.0). Inputs are mounted Kaggle datasets (see kernel-metadata).

Model input: mono 16 kHz float32 [1, n], n = 4 s windows; output logits
[human, ai]. The browser averages logits over windows (median)."""
import glob
import json
import os
import random
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "soundfile"], check=False)

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

BASE = os.environ.get("GT_BASE", "facebook/wav2vec2-base")
EPOCHS = int(os.environ.get("GT_EPOCHS", "4"))
SMOKE = os.environ.get("GT_SMOKE", "0") == "1"
SR, WIN = 16000, 64000
EVAL_N = 300 if SMOKE else 6000
OUT = "/kaggle/working"
rng = random.Random(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
# GT_EVAL_ONLY=1: score the model from a previous run's output (mounted as a
# kernel source) on the benchmarks, without training. Runs on CPU.
EVAL_ONLY = os.environ.get("GT_EVAL_ONLY", "0") == "1"
# GT_EXTRA=1: also train on WaveFake vocoder fakes (CC BY-SA 4.0) and their
# LJSpeech source recordings (public domain) — same voice on both sides.
EXTRA = os.environ.get("GT_EXTRA", "0") == "1"
WINDOWS = 4  # eval: mean logit over up to 4 evenly spaced 4 s windows, as the browser does


_INDEX = None


def index():
    """One pruned walk of /kaggle/input: small files by name, and each big
    audio folder (flac/, wav/) as a directory — never list 600k files."""
    global _INDEX
    if _INDEX is None:
        files, dirs = {}, {}
        for root, ds, fs in os.walk("/kaggle/input"):
            for f in fs:
                if not f.endswith((".flac", ".wav")):
                    files.setdefault(f, []).append(os.path.join(root, f))
            audio = [d for d in ds if d in ("flac", "wav")]
            for d in audio:
                dirs.setdefault(os.path.join(os.path.basename(root), d), []).append(os.path.join(root, d))
            ds[:] = [d for d in ds if d not in audio]
            if any(f.endswith((".flac", ".wav")) for f in fs[:50]):  # a flat audio folder
                dirs.setdefault(os.path.basename(root), []).append(root)
                ds[:] = []
        _INDEX = (files, dirs)
        print("indexed", len(files), "files,", {k: len(v) for k, v in dirs.items()}, flush=True)
    return _INDEX


def find(pattern):
    """`pattern` is 'name-glob' for small files or 'parent/flac' for folders."""
    import fnmatch
    files, dirs = index()
    if pattern in dirs:
        return dirs[pattern]
    return [p for name, ps in files.items() if fnmatch.fnmatch(name, pattern) for p in ps]


def load_wav(path):
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(1)
    except Exception:  # some corpus FLACs trip libsndfile; ffmpeg reads them
        raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
        x, sr = np.frombuffer(raw, np.float32).copy(), SR
        if not len(x):
            raise ValueError("empty decode")
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def crop(x, train):
    if len(x) >= WIN:
        s = rng.randrange(len(x) - WIN + 1) if train else (len(x) - WIN) // 2
        return x[s:s + WIN]
    return np.tile(x, WIN // max(1, len(x)) + 1)[:WIN]


CODECS = [("libmp3lame", "mp3", ["32k", "64k", "128k"]), ("libopus", "ogg", ["16k", "32k", "64k"]), ("aac", "adts", ["32k", "64k", "96k"])]


def codec(x):
    """Round-trip through a lossy codec at a random bitrate (ASVspoof 2021 DF
    and real-world audio are mostly compressed)."""
    enc, fmt, rates = rng.choice(CODECS)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    try:
        comp = subprocess.run(["ffmpeg", "-v", "quiet", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-", "-c:a", enc, "-b:a", rng.choice(rates), "-f", fmt, "-"],
                              input=pcm, capture_output=True, check=True, timeout=20).stdout
        raw = subprocess.run(["ffmpeg", "-v", "quiet", "-f", fmt, "-i", "-", "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                             input=comp, capture_output=True, check=True, timeout=20).stdout
        y = np.frombuffer(raw, np.float32)
        if len(y) < len(x) // 2:
            return x
        return np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)].astype(np.float32)
    except Exception:
        return x


def augment(x):
    if rng.random() < 0.5:
        x = codec(x)
    x = x * 10 ** (rng.uniform(-12, 6) / 20)
    if rng.random() < 0.5:
        x = x + rng.uniform(0, 0.01) * np.random.randn(len(x)).astype(np.float32)
    if rng.random() < 0.3:  # band-limit, like phone/codec paths
        k = rng.choice([2, 3])
        x = resample_poly(resample_poly(x, 1, k), k, 1)[: len(x)].astype(np.float32)
    return np.clip(x, -1, 1)


# ---------- data ----------
def asv19(split):
    proto = find(f"ASVspoof2019.LA.cm.{split}.tr*.txt")[0]
    root = find(f"ASVspoof2019_LA_{split}/flac")[0]
    rows = []
    for ln in open(proto):
        p = ln.split()
        rows.append((os.path.join(root, p[1] + ".flac"), 0 if p[-1] == "bonafide" else 1, p[3]))
    return rows


def asv21df():
    keyf = "/tmp/DF-keys/keys/DF/CM/trial_metadata.txt"
    if not os.path.exists(keyf):
        subprocess.run("cd /tmp && wget -q -T 60 -t 3 https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz && mkdir -p DF-keys && tar xzf DF-keys-full.tar.gz -C DF-keys", shell=True, timeout=900)
    roots = find("ASVspoof2021_DF_eval/flac")
    rows = []
    for ln in open(keyf):
        p = ln.split()
        for r in roots:
            f = os.path.join(r, p[1] + ".flac")
            if os.path.exists(f):
                rows.append((f, 0 if "bonafide" in p else 1, p[4] if len(p) > 4 else "?"))
                break
        if len(rows) >= 20 * EVAL_N:
            break
    return rows


def itw():
    # Kaggle mirror bhaveshkumars/release-in-the-wild: <split>/{fake,real}/*.wav
    # (the original release is one test set; the mirror's splits are pooled).
    rows, tops = [], []
    for root, ds, _ in os.walk("/kaggle/input"):
        if os.path.basename(root) == "release_in_the_wild":
            tops.append(root)
            ds[:] = []
        elif root.count(os.sep) > 6 or os.path.basename(root) in ("flac", "wav", "fake", "real"):
            ds[:] = []
    for top in tops:
        for split in os.listdir(top):
            for lab in ("fake", "real"):
                d = os.path.join(top, split, lab)
                if os.path.isdir(d):
                    rows += [(os.path.join(d, f), 1 if lab == "fake" else 0, split) for f in os.listdir(d) if f.endswith(".wav")]
    return rows


def find_dirs(name, maxdepth=7):
    hits = []
    for root, ds, _ in os.walk("/kaggle/input"):
        if os.path.basename(root) == name:
            hits.append(root)
            ds[:] = []
        elif root.count(os.sep) > maxdepth or os.path.basename(root) in ("flac",):
            ds[:] = []
    return hits


def wavefake(per_dir=1500):
    rows = []
    for top in find_dirs("generated_audio"):
        for sub in sorted(os.listdir(top)):
            d = os.path.join(top, sub)
            if os.path.isdir(d):
                fs = [f for f in os.listdir(d) if f.endswith(".wav")]
                rows += [(os.path.join(d, f), 1, "wavefake:" + sub) for f in rng.sample(fs, min(per_dir, len(fs)))]
    return rows


def ljspeech(n=13000):
    rows = []
    for top in find_dirs("wavs"):
        fs = [f for f in os.listdir(top) if f.startswith("LJ") and f.endswith(".wav")]
        rows += [(os.path.join(top, f), 0, "ljspeech") for f in rng.sample(fs, min(n, len(fs)))]
    return rows


def sample(rows, n):
    if len(rows) <= n:
        return rows
    by = {0: [r for r in rows if r[1] == 0], 1: [r for r in rows if r[1] == 1]}
    k = min(len(by[0]), n // 2)
    return rng.sample(by[0], k) + rng.sample(by[1], min(len(by[1]), n - k))


class DS(torch.utils.data.Dataset):
    def __init__(self, rows, train):
        self.rows, self.train = rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, y, _ = self.rows[i]
        try:
            x = crop(load_wav(path), self.train)
        except Exception:  # a few corpus files don't decode; skip them
            return (torch.zeros(WIN) if self.train else torch.zeros(WINDOWS, WIN)), -1
        if self.train:
            x = augment(x)
            x = (x - x.mean()) / (x.std() + 1e-7)  # wav2vec2 feature-extractor normalisation
            return torch.from_numpy(x.astype(np.float32)), y
        return self.views(path), y

    def views(self, path):
        """Eval: up to WINDOWS evenly spaced windows (repeat-padded to WINDOWS)."""
        full = load_wav(path)
        n = max(1, min(WINDOWS, len(full) // WIN))
        starts = [0] if len(full) <= WIN else np.linspace(0, len(full) - WIN, n).astype(int)
        ws = [crop(full[s:s + WIN], False) for s in starts]
        ws = [(w - w.mean()) / (w.std() + 1e-7) for w in ws]
        ws += [ws[i % len(ws)] for i in range(WINDOWS - len(ws))]
        return torch.from_numpy(np.stack(ws).astype(np.float32))


# ---------- model ----------
def build():
    from transformers import AutoModelForAudioClassification
    return AutoModelForAudioClassification.from_pretrained(BASE, num_labels=2)


def eer(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


@torch.no_grad()
def evaluate(model, rows):
    model.eval()
    dl = torch.utils.data.DataLoader(DS(rows, False), batch_size=8, num_workers=4)
    s, ok = [], []
    for x, yy in dl:
        b = x.shape[0]
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            lg = model(x.view(-1, WIN).to(dev)).logits.float()
        s += (lg[:, 1] - lg[:, 0]).view(b, -1).mean(1).cpu().tolist()
        ok += (yy >= 0).tolist()
    ok = np.array(ok, dtype=bool)
    s = np.where(ok, s, np.nan).tolist()
    y = np.array([r[1] for r in rows])[ok]
    s_ok = np.array(s)[ok]
    from sklearn.metrics import roc_auc_score
    return {"n": int(ok.sum()), "unreadable": int((~ok).sum()), "eer": round(eer(y, s_ok), 4), "auc": round(float(roc_auc_score(y, s_ok)), 4)}, s


class OnnxModel:
    """The browser's int8 ONNX export behind the torch model's interface."""
    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = os.cpu_count()
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    def eval(self):
        return self

    def to(self, _):
        return self

    def __call__(self, x):
        out = np.concatenate([self.sess.run(None, {"input_values": x[i:i + 1].float().numpy()})[0] for i in range(x.shape[0])])
        return type("O", (), {"logits": torch.from_numpy(out)})()


def eval_only():
    from transformers import AutoModelForAudioClassification
    cfg = [p for p in glob.glob("/kaggle/input/**/model/config.json", recursive=True)][0]
    models = {"fp32": AutoModelForAudioClassification.from_pretrained(os.path.dirname(cfg)).to(dev)}
    q = glob.glob("/kaggle/input/**/onnx/model_quantized.onnx", recursive=True)
    if q:
        models["int8"] = OnnxModel(q[0])
    torch.set_num_threads(os.cpu_count())
    metrics = {"model": cfg}
    for name, fn in (("in-the-wild", itw), ("asv21-df", asv21df)):
        try:
            rows = sample(fn(), EVAL_N)
            for tag, model in models.items():
                m, s = evaluate(model, rows)
                metrics[f"{name}/{tag}"] = m
                print(name, tag, m, flush=True)
        except Exception as e:
            metrics[name] = {"error": repr(e)[:300]}
            print(name, "FAILED", repr(e)[:300], flush=True)
        json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=1)


def main():
    if EVAL_ONLY:
        return eval_only()
    train = asv19("train")
    if EXTRA:
        wf, lj = wavefake(), ljspeech()
        print("extra train: wavefake", len(wf), "ljspeech", len(lj), flush=True)
        train += wf + lj
    if SMOKE:
        train = sample(train, 800)
    bal = [r for r in train if r[1] == 0]
    w = [len(train) / (2 * len(bal)) if r[1] == 0 else len(train) / (2 * (len(train) - len(bal))) for r in train]
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(train), replacement=True)
    dl = torch.utils.data.DataLoader(DS(train, True), batch_size=24, sampler=sampler, num_workers=4, drop_last=True)
    model = build().to(dev)
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.01)
    steps = EPOCHS * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-5, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    evals = {"asv19-dev": sample(asv19("dev"), EVAL_N // 3)}
    metrics = {"base": BASE, "epochs": EPOCHS, "train": len(train), "history": []}
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for i, (x, y) in enumerate(dl):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(x.to(dev)).logits.float(), y.to(dev), ignore_index=-1)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if i % 200 == 0:
                print(f"ep {ep} step {i}/{len(dl)} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        m, _ = evaluate(model, evals["asv19-dev"])
        metrics["history"].append({"epoch": ep, **m})
        print("dev", m, flush=True)
    model.save_pretrained(f"{OUT}/model")
    for name, fn in (("asv19-eval", lambda: asv19("eval")), ("asv21-df", asv21df), ("in-the-wild", itw)):
        try:
            rows = sample(fn(), EVAL_N)
            m, s = evaluate(model, rows)
            metrics[name] = m
            pd.DataFrame({"path": [r[0] for r in rows], "label": [r[1] for r in rows], "slice": [r[2] for r in rows], "score": s}).to_parquet(f"{OUT}/scores-{name}.parquet")
            print(name, m, flush=True)
        except Exception as e:
            metrics[name] = {"error": repr(e)[:300]}
            print(name, "FAILED", repr(e)[:300], flush=True)
        json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=1)
    export(model)


def export(model):
    model = model.float().cpu().eval()
    os.makedirs(f"{OUT}/onnx", exist_ok=True)
    x = torch.randn(1, WIN)
    torch.onnx.export(model, (x,), f"{OUT}/onnx/model.onnx", input_names=["input_values"], output_names=["logits"],
                      dynamic_axes={"input_values": {0: "b", 1: "n"}, "logits": {0: "b"}}, opset_version=17, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    # MatMul only: quantising the convolutional feature encoder too took EER
    # from 5.7% to 29% (gpttrace-audio-requant).
    quantize_dynamic(f"{OUT}/onnx/model.onnx", f"{OUT}/onnx/model_quantized.onnx", weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
    import onnxruntime as ort
    sess = ort.InferenceSession(f"{OUT}/onnx/model_quantized.onnx", providers=["CPUExecutionProvider"])
    xs = torch.randn(4, WIN)
    with torch.no_grad():
        ref = model(xs).logits.numpy()
    q = np.stack([sess.run(None, {"input_values": xs[i:i + 1].numpy()})[0][0] for i in range(4)])
    m = json.load(open(f"{OUT}/metrics.json"))
    m["export"] = {"bytes_int8": os.path.getsize(f"{OUT}/onnx/model_quantized.onnx"), "int8_vs_fp32_max_abs": float(np.abs(q - ref).max())}
    json.dump(m, open(f"{OUT}/metrics.json", "w"), indent=1)
    os.remove(f"{OUT}/onnx/model.onnx")
    print("export", m["export"], flush=True)


if __name__ == "__main__":
    main()
