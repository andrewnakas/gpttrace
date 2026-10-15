"""Scores GPTTrace's voice-clone detector on speech-deepfake leaderboard sets,
on a free Colab T4 (eval/train/queue/colab_job.sh). Audio comes from the
SpeechAntiSpoofingBenchmarks mirrors on Hugging Face (parquet, streamed one
shard at a time), and the model is our full-precision release
(nakas/gpttrace-wavlm-audio-v3, pytorch/). Preprocessing is exactly the
training-time evaluation in eval/train/audio/audio_train.py: 16 kHz mono, up
to 4 evenly spaced 4 s windows (repeat-padded to 4), each normalised, logits
averaged.

Writes, per set, to /content/out/:
  sab/<mirror>.txt         "<utterance_id> <score> <source path>"   (SpeechAntiSpoofingBenchmarks arena)
  dfarena/<protocol>.txt   "<file_name> <score>"      (Speech DF Arena format)
  <mirror>.json            EER and Speech DF Arena protocol coverage
Score = human logit − AI logit, so higher means bona fide, as both arenas expect.

  GT_SETS="InTheWild,DFADD"   which mirrors to score (see SETS)
  GT_LIMIT=2000               rows per set, for a quick check
"""
import io, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "soundfile", "huggingface_hub", "pyarrow", "scikit-learn"], check=False)
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download
from scipy.signal import resample_poly
from sklearn.metrics import roc_curve

SR, WIN, WINDOWS = 16000, 64000, 4
ORG = "SpeechAntiSpoofingBenchmarks"
OUT = "/content/out"
DF_PROTO = "https://raw.githubusercontent.com/Speech-Arena/speech_df_arena/main/protocol_files/{}.csv"

# mirror → (Speech DF Arena protocol name or None, utterance_id → DF-Arena file_name)
SETS = {
    "InTheWild": ("in_the_wild", lambda u: u[4:] if u.startswith("ITW_") else u),
    "DFADD": ("dfadd", lambda u: u.split("__", 1)[-1]),
    "ASVspoof2019_LA": ("asvspoof_2019", lambda u: u),
    "ASVspoof2021_LA": ("asvspoof_2021_la", lambda u: u),
    "ASVspoof2021_DF": ("asvspoof_2021_df", lambda u: u),
    "ASVspoof5": ("asvspoof_2024", lambda u: u),
    "ADD22_eval_31": ("add_2022_track_3", lambda u: u),
    "ADD2023_track12_test_r1": ("add_2023_round_1", lambda u: u),
    "SONAR": ("sonar", lambda u: u.split("__", 1)[-1]),
    "LibriSeVoc": ("librisevoc", lambda u: u.split("__", 1)[-1]),
}
# Sets that are not mirrored: downloaded from their own hosts (all NC-licensed;
# evaluation only, never training). Output: dfarena/<proto>.txt keyed the way
# Speech DF Arena's sample protocols name files; labels are joined locally.
ZEN = "https://zenodo.org/records/{}/files/{}?download=1"
DIRECT = {
    "FakeOrReal": {"proto": "fake_or_real", "parts": [["https://bil.eecs.yorku.ca/share/for-norm.tar.gz"]], "unpack": "tar",
                   "match": "/testing/", "key": lambda p: p.split("/")[-2] + "/" + os.path.basename(p)[:-4].replace(".wav_", "_")},
    "ADD22_T1": {"proto": "add_2022_track_1", "parts": [[ZEN.format(10843991, f) for f in ["track1test.zip"] + [f"track1test.z{i:02d}" for i in range(1, 10)]]],
                 "unpack": "splitzip", "key": lambda p: os.path.basename(p).rsplit(".", 1)[0]},
    "ADD22_T3": {"proto": "add_2022_track_3", "parts": [[ZEN.format(12188055, f"track32test.zip.{i:03d}") for i in range(1, 12)]],
                 "unpack": "splitzip", "key": lambda p: os.path.basename(p).rsplit(".", 1)[0]},
    "ADD23_R1": {"proto": "add_2023_round_1", "parts": [[ZEN.format(12175884, f"track1_2-testR1.tar.gz.{i:02d}") for i in range(8)]],
                 "unpack": "cattar", "key": lambda p: os.path.basename(p).rsplit(".", 1)[0]},
    "ADD23_R2": {"proto": "add_2023_round_2", "parts": [[ZEN.format(12176326, f"track1_2-testR2.tar.gz.{i:02d}") for i in range(8)]],
                 "unpack": "cattar", "key": lambda p: os.path.basename(p).rsplit(".", 1)[0]},
    "Codecfake": {"proto": "codecfake", "parts": [[ZEN.format(11169781, f"C{i}.zip")] for i in range(1, 7)]
                  + [[ZEN.format(11125029, f) for f in ["C7_split.zip"] + [f"C7_split.z{i:02d}" for i in range(1, 7)]]],
                  "unpack": "auto", "key": lambda p: p.split("/")[-2].split("_")[0] + "/" + os.path.basename(p).rsplit(".", 1)[0]},
}
SETS_RUN = [s for s in os.environ.get("GT_SETS", "InTheWild,DFADD").split(",") if s]
LIMIT = int(os.environ.get("GT_LIMIT", "0"))
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    from transformers import AutoModelForAudioClassification
    d = snapshot_download("nakas/gpttrace-wavlm-audio-v3", allow_patterns=["pytorch/*"])
    m = AutoModelForAudioClassification.from_pretrained(os.path.join(d, "pytorch")).to(dev).eval()
    return m.half() if dev == "cuda" else m


def decode(b):
    try:
        x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
        x = x.mean(1)
    except Exception:
        raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], input=b, capture_output=True).stdout
        x, sr = np.frombuffer(raw, np.float32).copy(), SR
    if not len(x):
        return None
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def views(x):
    def crop(w):
        if len(w) >= WIN:
            s = (len(w) - WIN) // 2
            return w[s:s + WIN]
        return np.tile(w, WIN // max(1, len(w)) + 1)[:WIN]
    n = max(1, min(WINDOWS, len(x) // WIN))
    starts = [0] if len(x) <= WIN else np.linspace(0, len(x) - WIN, n).astype(int)
    ws = [crop(x[s:s + WIN]) for s in starts]
    ws = [(w - w.mean()) / (w.std() + 1e-7) for w in ws]
    ws += [ws[i % len(ws)] for i in range(WINDOWS - len(ws))]
    return np.stack(ws).astype(np.float32)


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float((fpr[i] + 1 - tpr[i]) / 2)


@torch.no_grad()
def score_batch(model, batch):
    x = torch.from_numpy(np.concatenate(batch)).to(dev)
    if dev == "cuda":
        x = x.half()
    lg = model(x).logits.float().view(-1, WINDOWS, 2).mean(1)
    return (lg[:, 0] - lg[:, 1]).cpu().numpy()          # human − AI


def utt_id(row):
    try:
        return json.loads(row.get("notes") or "{}").get("utterance_id")
    except Exception:
        return None


def run_set(model, name):
    proto_name, to_df = SETS[name]
    fs = HfFileSystem(token=False)
    shards = sorted(f for f in fs.glob(f"datasets/{ORG}/{name}/**/*.parquet") if "labels" not in f)
    os.makedirs(f"{OUT}/sab", exist_ok=True); os.makedirs(f"{OUT}/dfarena", exist_ok=True)
    sab = open(f"{OUT}/sab/{name}.txt", "w")
    ids, labels, scores, failed, t0 = [], [], [], 0, time.time()
    pool = ThreadPoolExecutor(4)
    for sh in shards:
        rel = sh.split(f"{ORG}/{name}/", 1)[1]
        local = hf_hub_download(f"{ORG}/{name}", rel, repo_type="dataset")
        pf = pq.ParquetFile(local)
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g).to_pylist()
            vs = list(pool.map(lambda r: decode((r.get("audio") or {}).get("bytes") or b""), t))
            batch, keep = [], []
            for r, x in zip(t, vs):
                src = (r.get("audio") or {}).get("path") or r.get("path") or ""
                u = utt_id(r) or os.path.splitext(os.path.basename(src))[0]
                if x is None:
                    failed += 1; continue
                batch.append(views(x)); keep.append((u, int(r["label"]), src))
                if len(batch) == 16:
                    for (uu, yy, pp), s in zip(keep, score_batch(model, batch)):
                        ids.append(uu); labels.append(yy); scores.append(float(s)); sab.write(f"{uu} {s:.5f} {pp}\n")
                    batch, keep = [], []
            if batch:
                for (uu, yy, pp), s in zip(keep, score_batch(model, batch)):
                    ids.append(uu); labels.append(yy); scores.append(float(s)); sab.write(f"{uu} {s:.5f} {pp}\n")
            sab.flush()
            print(f"{name}: {len(ids)} scored, {failed} undecodable, {len(ids) / (time.time() - t0):.1f}/s", flush=True)
            if LIMIT and len(ids) >= LIMIT:
                break
        os.remove(local)
        if LIMIT and len(ids) >= LIMIT:
            break
    sab.close()
    y = np.array(labels)                                   # mirror labels: 1 = spoof
    res = {"set": name, "n": len(ids), "undecodable": failed, "eer_percent": round(100 * eer(1 - y, np.array(scores)), 3) if len(set(labels)) == 2 else None}
    if proto_name:
        import csv, urllib.request
        keys = [os.path.splitext(os.path.basename(r["file_name"]))[0] for r in csv.DictReader(io.StringIO(urllib.request.urlopen(DF_PROTO.format(proto_name)).read().decode()))]
        mine = {to_df(u): s for u, s in zip(ids, scores)}
        hit = [k for k in keys if k in mine]
        res.update({"dfarena_protocol": proto_name, "dfarena_rows": len(keys), "dfarena_covered": len(hit)})
        if not LIMIT and len(hit) < len(keys):
            res["dfarena_missing_examples"] = [k for k in keys if k not in mine][:5]
            res["mirror_id_examples"] = ids[:5]
        with open(f"{OUT}/dfarena/{proto_name}.txt", "w") as f:
            for k in keys:
                if k in mine:
                    f.write(f"{k} {mine[k]:.5f}\n")
    json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=1)
    print("RESULT", json.dumps(res), flush=True)


def fetch(urls, d):
    os.makedirs(d, exist_ok=True)
    files = []
    for u in urls:
        f = os.path.join(d, u.split("/")[-1].split("?")[0])
        subprocess.run(["wget", "-q", "-c", "-t", "8", "-O", f, u], check=True)
        files.append(f)
    return files


def unpack(files, kind, d):
    first = files[0]
    if kind == "tar" or (kind == "auto" and first.endswith((".tar.gz", ".tgz"))):
        subprocess.run(["tar", "xzf", first, "-C", d], check=True)
    elif kind == "cattar":
        subprocess.run(f"cat {' '.join(files)} | tar xz -C {d}", shell=True, check=True)
    elif kind == "splitzip" or (kind == "auto" and len(files) > 1):
        # 7-Zip reads split zips (.zip + .z01…) in place, so the parts are
        # never joined into one more copy (C7 alone is ~32 GB).
        subprocess.run("command -v 7z >/dev/null || apt-get -qq install -y p7zip-full >/dev/null", shell=True)
        first = next((f for f in files if f.endswith(".zip")), sorted(files)[0])   # .zip + .z01… or .zip.001…
        subprocess.run(["7z", "x", "-y", f"-o{d}", first], check=True, stdout=subprocess.DEVNULL)
    else:
        subprocess.run(["unzip", "-q", "-o", first, "-d", d], check=True)
    for f in files + [f"{d}/joined.zip"]:
        if os.path.exists(f):
            os.remove(f)


def run_direct(model, name):
    cfg = DIRECT[name]
    os.makedirs(f"{OUT}/dfarena", exist_ok=True)
    out = open(f"{OUT}/dfarena/{cfg['proto']}.txt", "a")
    pool, n, failed, t0 = ThreadPoolExecutor(4), 0, 0, time.time()
    for k, urls in enumerate(cfg["parts"]):
        d = f"/content/dl/{name}/{k}"
        unpack(fetch(urls, d), cfg["unpack"], d)
        paths = sorted(os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs
                       if f.lower().endswith((".wav", ".flac", ".mp3")) and cfg.get("match", "/") in os.path.join(r, f))
        for i in range(0, len(paths), 64):
            chunk = paths[i:i + 64]
            xs = list(pool.map(lambda p: decode(open(p, "rb").read()), chunk))
            ok = [(p, x) for p, x in zip(chunk, xs) if x is not None]
            failed += len(chunk) - len(ok)
            for j in range(0, len(ok), 16):
                part = ok[j:j + 16]
                for (p, _), sc in zip(part, score_batch(model, [views(x) for _, x in part])):
                    out.write(f"{cfg['key'](p)} {sc:.5f}\n"); n += 1
            out.flush()
            print(f"{name}: {n} scored, {failed} undecodable, {n / (time.time() - t0):.1f}/s", flush=True)
        subprocess.run(["rm", "-rf", d])
    out.close()
    res = {"set": name, "n": n, "undecodable": failed, "dfarena_protocol": cfg["proto"]}
    json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=1)
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    model = load_model()
    print("model on", dev, flush=True)
    for s in SETS_RUN:
        try:
            run_direct(model, s) if s in DIRECT else run_set(model, s)
        except Exception as e:
            print("SET FAILED", s, repr(e)[:400], flush=True)
