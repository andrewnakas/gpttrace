"""Samples audio benchmarks into ~/.cache/gpttrace/eval/bench/audio/<name>/
with an index.json of {id, label, slice, file} rows, reading only the bytes
needed (parquet row groups over HTTP, zip members by range request).

  ~/.cache/gpttrace/venv/bin/python eval/bench/sample_audio.py [n_per_slice]

Sets (licence): asvspoof2019-la eval (ODC-By), in-the-wild (CC BY-SA 4.0)."""
import json
import os
import random
import sys
from collections import defaultdict

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_url
from remotezip import RemoteZip

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
ROOT = os.path.expanduser("~/.cache/gpttrace/eval/bench/audio")
rng = random.Random(0)


def write(name, rows):
    d = os.path.join(ROOT, name)
    json.dump(rows, open(os.path.join(d, "index.json"), "w"), indent=0)
    counts = defaultdict(int)
    for r in rows:
        counts[r["slice"]] += 1
    print(name, len(rows), dict(counts), flush=True)


def asvspoof2019():
    # label 1 = spoof (63,882 in the eval set), 0 = bona fide (7,355). The
    # mirror has no attack ids, so spoofs are one slice.
    name, fs = "asvspoof2019-la", HfFileSystem(token=False)
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    files = sorted(fs.glob("datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/data/test-*.parquet"))
    need = defaultdict(lambda: N)
    rows = []
    for f in files:
        pf = pq.ParquetFile(fs.open(f))
        for g in rng.sample(range(pf.num_row_groups), pf.num_row_groups):
            t = pf.read_row_group(g).to_pylist()
            for r in t:
                slice_ = "ai:spoof" if r["label"] == 1 else "human:bonafide"
                if need[slice_] <= 0:
                    continue
                a = r["audio"]
                fn = os.path.basename(a.get("path") or r["path"]) or f"{len(rows)}.flac"
                open(os.path.join(d, fn), "wb").write(a["bytes"])
                rows.append({"id": f"{name}-{fn}", "label": "human" if slice_.startswith("human") else "ai", "slice": slice_, "file": f"bench/audio/{name}/{fn}"})
                need[slice_] -= 1
            if len(need) == 2 and all(v <= 0 for v in need.values()):
                break
        if len(need) == 2 and all(v <= 0 for v in need.values()):
            break
        print(f, len(rows), flush=True)
    write(name, rows)


def in_the_wild():
    name = "in-the-wild"
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    with RemoteZip(hf_hub_url("mueller91/In-The-Wild", "release_in_the_wild.zip", repo_type="dataset")) as z:
        names = z.namelist()
        meta = [n for n in names if n.endswith("meta.csv")][0]
        lines = z.read(meta).decode().splitlines()[1:]
        by = defaultdict(list)
        for ln in lines:
            fn, speaker, lab = ln.split(",")[:3]
            by["human:bona-fide" if lab.strip().strip('"') == "bona-fide" else "ai:spoof"].append((fn, speaker))
        prefix = meta[: -len("meta.csv")]
        rows = []
        for slice_, items in by.items():
            for fn, spk in rng.sample(items, min(N, len(items))):
                open(os.path.join(d, fn), "wb").write(z.read(prefix + fn))
                rows.append({"id": f"{name}-{fn}", "label": "human" if slice_.startswith("human") else "ai", "slice": slice_, "speaker": spk, "file": f"bench/audio/{name}/{fn}"})
    write(name, rows)


if __name__ == "__main__":
    for fn in (asvspoof2019, in_the_wild):
        try:
            fn()
        except Exception as e:
            print(fn.__name__, "FAILED", repr(e)[:300], flush=True)
