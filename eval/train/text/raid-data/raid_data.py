"""Kaggle job A (CPU, internet): stream RAID and write sampled train/val
splits plus the full test set as parquet.

Splits are by title: every generation, attack and the human text for one
title land on the same side, so validation measures unseen documents.
Sampling caps each (domain, model, decoding, repetition_penalty, attack)
cell so no generator/attack dominates; all human rows are kept."""
import hashlib
import os
import random
import subprocess

import pandas as pd

BASE = "https://dataset.raid-bench.xyz/"
OUT = "/kaggle/working"
COLS = ["id", "model", "decoding", "repetition_penalty", "attack", "domain", "title", "generation"]
TRAIN_CAP = 150   # AI rows per cell for training (~4.2k cells)
VAL_CAP = 12      # AI rows per cell for validation
VAL_PCT = 5       # % of titles held out

rng = random.Random(0)


def is_val(title):
    return int(hashlib.md5(str(title).encode()).hexdigest(), 16) % 100 < VAL_PCT


def cell(r):
    return (r.domain, r.model, r.decoding, r.repetition_penalty, r.attack)


def reservoir(store, key, row, cap, seen):
    seen[key] = seen.get(key, 0) + 1
    bucket = store.setdefault(key, [])
    if len(bucket) < cap:
        bucket.append(row)
    else:
        j = rng.randrange(seen[key])
        if j < cap:
            bucket[j] = row


def fetch(name):
    # pandas reads an http URL fully into memory; stream it to local disk.
    path = f"/tmp/{name}"
    subprocess.run(["wget", "-q", "-O", path, BASE + name], check=True)
    print(name, os.path.getsize(path) // 2**20, "MB", flush=True)
    return path


def sample_train():
    tr, va, seen_t, seen_v, humans = {}, {}, {}, {}, []
    n = 0
    for chunk in pd.read_csv(fetch("train.csv"), usecols=COLS, chunksize=200_000, dtype=str, keep_default_na=False):
        for r in chunk.itertuples(index=False):
            v = is_val(r.title)
            if r.model == "human":
                humans.append((r, v))
            elif v:
                reservoir(va, cell(r), r, VAL_CAP, seen_v)
            else:
                reservoir(tr, cell(r), r, TRAIN_CAP, seen_t)
        n += len(chunk)
        print(f"train rows read: {n:,}", flush=True)
    rows_t = [r for b in tr.values() for r in b] + [r for r, v in humans if not v]
    rows_v = [r for b in va.values() for r in b] + [r for r, v in humans if v]
    for name, rows in (("train", rows_t), ("val", rows_v)):
        df = pd.DataFrame(rows, columns=COLS).sample(frac=1, random_state=0)
        df.to_parquet(f"{OUT}/raid_{name}.parquet", index=False)
        print(name, len(df), df.model.value_counts().to_dict(), df.attack.value_counts().to_dict(), flush=True)
    os.remove("/tmp/train.csv")


def save_test():
    # The test set's labels and metadata are hidden; keep what it has.
    cols = ["id", "domain", "attack", "generation"]
    parts = [c for c in pd.read_csv(fetch("test.csv"), usecols=lambda c: c in cols, chunksize=200_000, dtype=str, keep_default_na=False)]
    df = pd.concat(parts)
    df.to_parquet(f"{OUT}/raid_test.parquet", index=False)
    print("test", len(df), list(df.columns), flush=True)


if __name__ == "__main__":
    sample_train()
    save_test()
    print(os.listdir(OUT))
