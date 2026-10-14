"""Hybrid text detector analysis (workstream A3). Reads the rows written by
eval/hybrid-text.js and compares, with logistic regression:
  ettin        the shipped fine-tuned classifier logit alone
  zero-shot    small-LM statistics alone (ll, ent, bino, fdg, rank)
  hybrid       both
Reports 5-fold CV on each set, and transfer: fit on the in-house set, test on
each benchmark set unchanged (the honest out-of-distribution number).
  ~/.cache/gpttrace/venv/bin/python eval/hybrid-fit.py [inhouse mage-ood-para ...]
"""
import json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = os.path.expanduser("~/.cache/gpttrace/eval/hybrid")
SETS = sys.argv[1:] or [f[:-6] for f in sorted(os.listdir(D)) if f.endswith(".jsonl")]
ZS = ["ll", "ent", "bino", "fdg", "rank"]
GROUPS = {"ettin": ["ettin"], "zero-shot": ZS, "hybrid": ["ettin"] + ZS}

def load(name):
    rows = [json.loads(l) for l in open(os.path.join(D, name + ".jsonl")) if l.strip()]
    rows = [r for r in rows if r.get("ettin") is not None and r.get("fdg") is not None]
    y = np.array([r["label"] == "ai" for r in rows], int)
    return rows, y

def X(rows, cols):
    a = np.array([[r[c] for c in cols] for r in rows], float)
    return np.clip(a, -50, 50)

def tpr_at(y, s, fpr_max=0.05):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= fpr_max].max())

def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))

def cv(rows, y, cols):
    s = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(rows, y):
        m = model().fit(X([rows[i] for i in tr], cols), y[tr])
        s[te] = m.decision_function(X([rows[i] for i in te], cols))
    return s

data = {n: load(n) for n in SETS}
print(f"{'set':<18}{'n':>6}  " + "  ".join(f"{g:>22}" for g in GROUPS))
for n, (rows, y) in data.items():
    if len(set(y)) < 2: continue
    cells = []
    for g, cols in GROUPS.items():
        s = cv(rows, y, cols)
        cells.append(f"AUC {roc_auc_score(y, s):.3f} T@5 {tpr_at(y, s):.3f}")
    print(f"{n:<18}{len(y):>6}  " + "  ".join(f"{c:>22}" for c in cells))

if "inhouse" in data:
    rows, y = data["inhouse"]
    print("\ntransfer: fit on inhouse, test unchanged")
    fitted = {g: model().fit(X(rows, cols), y) for g, cols in GROUPS.items()}
    for g, m in fitted.items():
        coef = m[-1].coef_[0]
        print(f"  {g:<10} coef " + ", ".join(f"{c}={w:+.2f}" for c, w in zip(GROUPS[g], coef)))
    for n, (r2, y2) in data.items():
        if n == "inhouse" or len(set(y2)) < 2: continue
        cells = [f"AUC {roc_auc_score(y2, m.decision_function(X(r2, GROUPS[g]))):.3f}" for g, m in fitted.items()]
        print(f"  {n:<16}{len(y2):>6}  " + "  ".join(f"{g}: {c}" for g, c in zip(GROUPS, cells)))
    # Human false positives by slice at the threshold giving 5% FPR in-house.
    print("\nin-house human slices flagged at the 5%-FPR threshold (CV scores)")
    for g, cols in GROUPS.items():
        s = cv(rows, y, cols)
        thr = np.quantile(s[y == 0], 0.95)
        sl = {}
        for r, v in zip(rows, s):
            if r["label"] == "human": sl.setdefault(r["slice"], []).append(v > thr)
        print(f"  {g:<10} " + ", ".join(f"{k}: {np.mean(v):.0%}" for k, v in sorted(sl.items()) if len(v) >= 20))
