"""Samples text benchmarks into ~/.cache/gpttrace/eval/bench/text/<name>/
index.jsonl ({id, label, slice, text}).

  ~/.cache/gpttrace/venv/bin/python eval/bench/sample_text.py [n_per_slice]

Sets (licence):
  mage            MAGE test, stratified by source (Apache-2.0)
  mage-ood-gpt4   MAGE unseen-model OOD set (GPT-4 text)
  mage-ood-para   the same, paraphrased (paraphrase attack)
  ghostbuster     Ghostbuster essays / WritingPrompts / Reuters (CC BY 3.0)
"""
import io
import json
import os
import random
import sys
import tarfile
import urllib.request
from collections import defaultdict

import pandas as pd

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
ROOT = os.path.expanduser("~/.cache/gpttrace/eval/bench/text")
MAGE = "https://huggingface.co/datasets/yaful/MAGE/resolve/main/"
rng = random.Random(0)


def write(name, rows):
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "index.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    c = defaultdict(int)
    for r in rows:
        c[r["label"]] += 1
    print(name, len(rows), dict(c), flush=True)


def mage_rows(file, name, per_slice):
    df = pd.read_csv(MAGE + file)
    rows = []
    # label 1 = human, 0 = machine. src is "<domain>_<generator or human>".
    for src, g in df.groupby("src"):
        for i, r in g.sample(n=min(per_slice, len(g)), random_state=0).iterrows():
            lab = "human" if int(r.label) == 1 else "ai"
            rows.append({"id": f"{name}-{i}", "label": lab, "slice": f"{lab}:{src}", "text": str(r.text)})
    return rows


def ghostbuster():
    # Repo tarball; folders <domain>/<source>/*.txt, source "human" or a model.
    url = "https://codeload.github.com/vivek3141/ghostbuster-data/tar.gz/refs/heads/master"
    by = defaultdict(list)
    with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url, timeout=600).read()), mode="r:gz") as t:
        for m in t.getmembers():
            p = m.name.split("/")
            if not m.isfile() or not p[-1].endswith(".txt") or len(p) < 4 or p[1] not in ("essay", "wp", "reuter"):
                continue
            src = p[2] if p[1] != "reuter" else p[2]
            if src.startswith("prompts") or "logprob" in m.name:
                continue
            by[(p[1], src)].append(m)
        rows = []
        for (dom, src), ms in sorted(by.items()):
            for m in rng.sample(ms, min(N * 2, len(ms))):
                lab = "human" if src == "human" else "ai"
                rows.append({"id": f"ghostbuster-{dom}-{src}-{os.path.basename(m.name)}", "label": lab, "slice": f"{lab}:{dom}-{src}",
                             "text": t.extractfile(m).read().decode("utf-8", "replace")})
    return rows


if __name__ == "__main__":
    for name, fn in [("mage", lambda: mage_rows("test.csv", "mage", N)),
                     ("mage-ood-gpt4", lambda: mage_rows("test_ood_set_gpt.csv", "mage-ood-gpt4", 10**6)),
                     ("mage-ood-para", lambda: mage_rows("test_ood_set_gpt_para.csv", "mage-ood-para", 10**6)),
                     ("ghostbuster", ghostbuster)]:
        try:
            write(name, fn())
        except Exception as e:
            print(name, "FAILED", repr(e)[:300], flush=True)
