"""Builds the Speech DF Arena submission from the Colab scoring runs
(eval/bench/arena_job.py → ~/.cache/gpttrace/colab/arena*/out/) and reports
EER per set, average and pooled EER, and protocol coverage.

Keys come from the arena's own sample protocols (protocol_files/*.csv in the
DF-Arena-Sample-Files zip, unpacked at ~/.cache/gpttrace/arena/). Mirror
scores (sab/*.txt: "<utterance_id> <score> <source path>") are matched to them
by utterance id first, then by source path; direct-download sets already use
arena keys (dfarena/*.txt).

  ~/.cache/gpttrace/venv/bin/python eval/bench/arena_assemble.py
  → ~/.cache/gpttrace/arena/submission/<protocol>.txt + summary.json
"""
import csv, glob, json, os
import numpy as np
from sklearn.metrics import roc_curve

HOME = os.path.expanduser("~/.cache/gpttrace")
PROTO = f"{HOME}/arena/DF-Arena-Sample-Files/DF-Arena-Sample-Files/protocol_files"
OUT = f"{HOME}/arena/submission"
MIRROR = {"in_the_wild": "InTheWild", "dfadd": "DFADD", "asvspoof_2019": "ASVspoof2019_LA", "asvspoof_2021_la": "ASVspoof2021_LA",
          "asvspoof_2021_df": "ASVspoof2021_DF", "asvspoof_2024": "ASVspoof5", "add_2022_track_3": "ADD22_eval_31",
          "add_2023_round_1": "ADD2023_track12_test_r1", "sonar": "SONAR", "librisevoc": "LibriSeVoc"}


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def candidates(uid, path):
    """Ways a mirror row can be named in an arena protocol."""
    stem = os.path.splitext(os.path.basename(path))[0] if path else ""
    parent = os.path.basename(os.path.dirname(path)) if path else ""
    out = [uid, uid.split("__", 1)[-1], uid[4:] if uid.startswith("ITW_") else uid, stem]
    if parent:
        out += [f"{parent}/{stem}", f"{parent}/{uid.split('__', 1)[-1]}"]
    return out


def load_scores(proto):
    scores = {}
    for f in sorted(glob.glob(f"{HOME}/colab/arena*/out/dfarena/{proto}.txt")):
        for line in open(f):
            k, s = line.split()[:2]
            scores[k] = float(s)
    return scores


def main():
    os.makedirs(OUT, exist_ok=True)
    summary, pooled_y, pooled_s = {}, [], []
    for pf in sorted(glob.glob(f"{PROTO}/*.csv")):
        proto = os.path.basename(pf)[:-4]
        rows = list(csv.DictReader(open(pf)))
        keys = {r["file_name"]: r["label"] for r in rows}
        got = load_scores(proto) if proto not in MIRROR else {}
        if proto in MIRROR:
            for f in sorted(glob.glob(f"{HOME}/colab/arena*/out/sab/{MIRROR[proto]}.txt")):
                for line in open(f):
                    parts = line.rstrip("\n").split(" ", 2)
                    uid, s, path = parts[0], float(parts[1]), parts[2] if len(parts) > 2 else ""
                    for c in candidates(uid, path):
                        if c in keys:
                            got[c] = s
                            break
        hit = [k for k in keys if k in got]
        res = {"rows": len(keys), "scored": len(hit), "coverage": round(len(hit) / len(keys), 4)}
        if hit:
            y = np.array([keys[k] == "bonafide" for k in hit], int)
            s = np.array([got[k] for k in hit])
            if len(set(y)) == 2:
                res["eer_percent"] = round(100 * eer(y, s), 3)
                pooled_y += list(y); pooled_s += list(s)
            if len(hit) == len(keys):
                with open(f"{OUT}/{proto}.txt", "w") as f:
                    for k in keys:
                        f.write(f"{k} {got[k]:.5f}\n")
        summary[proto] = res
        print(f"{proto:20s} {len(hit):>7}/{len(keys):<7} EER {res.get('eer_percent', '—')}")
    done = [v["eer_percent"] for v in summary.values() if v.get("coverage") == 1 and "eer_percent" in v]
    summary["_average_eer_complete_sets"] = round(float(np.mean(done)), 3) if done else None
    summary["_pooled_eer_scored"] = round(100 * eer(np.array(pooled_y), np.array(pooled_s)), 3) if pooled_y else None
    summary["_complete_sets"] = len(done)
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    print("average EER over complete sets:", summary["_average_eer_complete_sets"], "| pooled EER over everything scored:", summary["_pooled_eer_scored"], f"| {len(done)}/14 sets complete")


if __name__ == "__main__":
    main()
