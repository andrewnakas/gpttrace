"""Kaggle job B (GPU T4x2): fine-tune a small encoder on RAID for GPTTrace's
deep text scan, export int8 ONNX for the browser.

Built into one file by eval/train/text/push.sh (normalize.py is prepended).
Reads job A's output (gpttrace-raid-data). Writes to /kaggle/working:
  model/            fine-tuned HF model + tokenizer
  onnx/             model.onnx, model_quantized.onnx (+ tokenizer files)
  metrics.json      RAID-style validation scores (TMR control and ours)
  val_scores.parquet  per-row validation scores, for ensemble fitting
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

SMOKE = os.environ.get("GT_SMOKE", "__SMOKE__") == "1"
BASE_MODEL = os.environ.get("GT_BASE", "__BASE__")
MAX_LEN = 512
TRAIN_ROWS = 2_000 if SMOKE else int(os.environ.get("GT_ROWS", "__ROWS__"))
VAL_ROWS = 1_500 if SMOKE else None
LR = float(os.environ.get("GT_LR", "__LR__"))
# Extra training rows from MAGE train (Apache-2.0; 27 LLMs, 10 human domains;
# label 1 = human). 0 disables. MAGE's OOD test sets are always reported.
MAGE_ROWS = 2_000 if SMOKE else int(os.environ.get("GT_MAGE", "__MAGE__"))
MAGE = "https://huggingface.co/datasets/yaful/MAGE/resolve/main/"
OUT = "/kaggle/working"
TMR = "Oxidane/tmr-ai-text-detector"


INPUT = os.environ.get("GT_INPUT", "/kaggle/input")    # Colab runs point this at /content


def data_dir():
    hits = glob.glob(f"{INPUT}/**/raid_train.parquet", recursive=True)
    if not hits:
        raise SystemExit(f"raid_train.parquet not found under {INPUT}")
    return os.path.dirname(hits[0])


def load(split, n=None):
    df = pd.read_parquet(f"{data_dir()}/raid_{split}.parquet")
    if n:
        df = df.sample(n=min(n, len(df)), random_state=0)
    return df.reset_index(drop=True)


# ---------- RAID-style metric (mirrors raid/evaluate.py) ----------
def raid_scores(df, score_col, fpr=0.05):
    """Per-domain threshold so `fpr` of unattacked human texts are flagged;
    report TPR over all AI rows and broken down by attack/model/domain."""
    out = {"thresholds": {}}
    flagged = pd.Series(False, index=df.index)
    for d, g in df.groupby("domain"):
        h = np.sort(g[(g.model == "human") & (g.attack == "none")][score_col].to_numpy())[::-1]
        if not len(h):
            continue
        k = int(np.floor(len(h) * fpr))
        thr = h[k] + 1e-9 if k < len(h) else -np.inf
        out["thresholds"][d] = float(thr)
        flagged[g.index] = g[score_col] >= thr
    ai = df.model != "human"
    out["tpr_all"] = float(flagged[ai].mean())
    out["tpr_noadv"] = float(flagged[ai & (df.attack == "none")].mean())
    for dim in ("attack", "model", "domain"):
        out[dim] = {k: round(float(flagged[ai & (df[dim] == k)].mean()), 4) for k in sorted(df.loc[ai, dim].unique())}
    from sklearn.metrics import roc_auc_score
    out["auroc"] = float(roc_auc_score(ai.astype(int), df[score_col]))
    return out


# ---------- batched inference ----------
@torch.no_grad()
def score(model, tok, texts, bs=64, device="cuda"):
    model.eval().to(device)
    order = np.argsort([len(t) for t in texts])
    out = np.zeros(len(texts), dtype=np.float32)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**enc).logits.float()
        out[idx] = (logits[:, 1] - logits[:, 0]).cpu().numpy()
    return out


def control(val):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TMR)
    model = AutoModelForSequenceClassification.from_pretrained(TMR)
    t0 = time.time()
    val["tmr_raw"] = score(model, tok, val.generation.tolist())
    val["tmr_norm"] = score(model, tok, val.text.tolist())
    print(f"TMR control scored in {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    return {"tmr_raw": raid_scores(val, "tmr_raw"), "tmr_norm": raid_scores(val, "tmr_norm")}


# ---------- training (run under torchrun) ----------
def load_pairs():
    """Topic-matched human/AI pairs from gen/gen_pairs.py, if mounted. 10% of
    topics (by source text) are held out for evaluation."""
    hits = glob.glob(f"{INPUT}/**/pairs.parquet", recursive=True)
    if not hits:
        return None
    d = pd.read_parquet(hits[0])
    d["text"] = d.text.astype(str).map(normalize_text)
    # Pairs are interleaved per source; hold out every 10th row of each source.
    d["holdout"] = d.groupby(["source", "label"]).cumcount() % 10 == 0
    return d


def pairs_eval(model, tok):
    d = load_pairs()
    if d is None:
        return None
    from sklearn.metrics import roc_auc_score
    h = d[d.holdout].reset_index(drop=True)
    h["score"] = score(model, tok, h.text.tolist())
    out = {"all": round(float(roc_auc_score(h.label, h.score)), 4)}
    for src, g in h.groupby("source"):
        if g.label.nunique() == 2:
            thr = np.sort(g[g.label == 0].score.to_numpy())[::-1][int(0.05 * (g.label == 0).sum())]
            out[src] = {"auroc": round(float(roc_auc_score(g.label, g.score)), 4), "tpr_at_5fpr": round(float((g[g.label == 1].score > thr).mean()), 4),
                        "human_flagged_at_0": round(float((g[g.label == 0].score > 0).mean()), 4)}
    return out


def tokenize_train():
    """Once, before torchrun: two ranks × a tokeniser pool each ran out of
    memory on 680k texts."""
    from transformers import AutoTokenizer
    import datasets
    tr = load("train", TRAIN_ROWS)
    # Paraphrase is the hardest RAID attack (98.2% vs ≥99.7% for the rest
    # in run 1); see its AI rows twice.
    para = tr[(tr.attack == "paraphrase") & (tr.model != "human")]
    tr = pd.concat([tr, para]).sample(frac=1, random_state=1).reset_index(drop=True)
    tr["text"] = tr.generation.map(normalize_text)
    tr["label"] = (tr.model != "human").astype(int)
    pairs = load_pairs()
    if pairs is not None:
        tr_p = pairs[~pairs.holdout]
        tr = pd.concat([tr[["text", "label"]], tr_p[["text", "label"]]]).sample(frac=1, random_state=3).reset_index(drop=True)
        print("added pair rows", len(tr_p), flush=True)
    if MAGE_ROWS:
        mg = pd.read_csv(MAGE + "train.csv").sample(n=MAGE_ROWS, random_state=0)
        mg = pd.DataFrame({"text": mg.text.astype(str).map(normalize_text), "label": 1 - mg.label.astype(int)})
        tr = pd.concat([tr[["text", "label"]], mg]).sample(frac=1, random_state=2).reset_index(drop=True)
        print("added MAGE rows", len(mg), flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ds = datasets.Dataset.from_pandas(tr[["text", "label"]])
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True, remove_columns=["text"], num_proc=2)
    ds.save_to_disk("/tmp/train_tok")
    print("tokenized", len(ds), flush=True)


def train():
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)
    import datasets
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ds = datasets.load_from_disk("/tmp/train_tok")
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    import inspect
    kw = dict(
        output_dir="/tmp/ckpt", per_device_train_batch_size=16, gradient_accumulation_steps=2,
        learning_rate=LR, num_train_epochs=1, lr_scheduler_type="cosine", weight_decay=0.01,
        fp16=True, logging_steps=50, save_strategy="no", report_to=[], dataloader_num_workers=2,
        group_by_length=True, ddp_find_unused_parameters=False, seed=0,
    )
    # transformers 5 folded warmup_ratio into warmup_steps (a float < 1 is a ratio).
    params = inspect.signature(TrainingArguments.__init__).parameters
    kw["warmup_ratio" if "warmup_ratio" in params else "warmup_steps"] = 0.05
    args = TrainingArguments(**{k: v for k, v in kw.items() if k in params})
    tkw = {"processing_class": tok} if "processing_class" in inspect.signature(Trainer.__init__).parameters else {"tokenizer": tok}
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorWithPadding(tok), **tkw)
    t0 = time.time()
    trainer.train()
    if trainer.is_world_process_zero():
        print(f"trained {len(ds)} rows in {time.time() - t0:.0f}s ({len(ds) / (time.time() - t0):.1f} rows/s)", flush=True)
        trainer.save_model(f"{OUT}/model")
        tok.save_pretrained(f"{OUT}/model")


# ---------- export ----------
def export(val):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"{OUT}/model")
    model = AutoModelForSequenceClassification.from_pretrained(f"{OUT}/model").eval()
    os.makedirs(f"{OUT}/onnx", exist_ok=True)
    enc = tok(["An example sentence to trace the graph."], return_tensors="pt")
    torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]), f"{OUT}/onnx/model.onnx",
                      input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}},
                      opset_version=17, dynamo=False)
    # Dynamic int8 drifts a lot on ModernBERT; try a few settings and keep the
    # one closest to torch (batch 1, as in the browser).
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import onnxruntime as ort
    from sklearn.metrics import roc_auc_score
    sub = val.sample(n=min(300, len(val)), random_state=1)
    encs = [tok(t, truncation=True, max_length=MAX_LEN, return_tensors="np") for t in sub.text]
    y = (sub.model != "human").astype(int)
    trials = {}
    for name, kw in {"s8": dict(weight_type=QuantType.QInt8), "s8pc": dict(weight_type=QuantType.QInt8, per_channel=True),
                     "u8pc": dict(weight_type=QuantType.QUInt8, per_channel=True),
                     "s8pc-matmul": dict(weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul"])}.items():
        path = f"/tmp/q-{name}.onnx"
        try:
            quantize_dynamic(f"{OUT}/onnx/model.onnx", path, **kw)
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            q = np.array([(lambda lg: lg[1] - lg[0])(sess.run(None, {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)})[0][0]) for e in encs])
            d = np.abs(q - sub.ours.to_numpy())
            trials[name] = {"bytes": os.path.getsize(path), "mean_abs": float(d.mean()), "max_abs": float(d.max()), "auc": float(roc_auc_score(y, q))}
        except Exception as e:
            trials[name] = {"error": repr(e)[:200]}
        print("quant", name, trials[name], flush=True)
    ok = {k: v for k, v in trials.items() if "mean_abs" in v}
    best = min(ok, key=lambda k: ok[k]["mean_abs"])
    shutil.move(f"/tmp/q-{best}.onnx", f"{OUT}/onnx/model_quantized.onnx")
    tok.save_pretrained(f"{OUT}/onnx")
    os.remove(f"{OUT}/onnx/model.onnx")
    return {"auc_torch_sample": float(roc_auc_score(y, sub.ours)), "chosen": best, "trials": trials}


def mage_ood(model, tok):
    """MAGE's out-of-distribution tests: unseen GPT-4 text, and the same
    paraphrased (Apache-2.0, publishable). AUROC and TPR at 5% FPR."""
    from sklearn.metrics import roc_auc_score
    out = {}
    for name in ("test_ood_set_gpt", "test_ood_set_gpt_para"):
        d = pd.read_csv(MAGE + name + ".csv")
        y = 1 - d.label.astype(int).to_numpy()
        s_ = score(model, tok, d.text.astype(str).map(normalize_text).tolist())
        thr = np.sort(s_[y == 0])[::-1][int(0.05 * (y == 0).sum())]
        out[name] = {"n": len(d), "auroc": round(float(roc_auc_score(y, s_)), 4), "tpr_at_5fpr": round(float((s_[y == 1] > thr).mean()), 4)}
    return out


@torch.no_grad()
def predict_test(model, tok):
    """RAID leaderboard predictions: [{id, score}] with score = P(AI) from
    the fp16 torch model (the browser runs its int8 export)."""
    te = load("test")
    if SMOKE:
        te = te.head(2000)
    logits = score(model, tok, te.generation.map(normalize_text).tolist(), bs=128)
    preds = [{"id": i, "score": float(1 / (1 + np.exp(-l)))} for i, l in zip(te.id, logits)]
    json.dump(preds, open(f"{OUT}/predictions.json", "w"))
    return len(preds)


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], check=False)
    val = load("val", VAL_ROWS)
    val["text"] = val.generation.map(normalize_text)
    metrics = {"base": BASE_MODEL, "smoke": SMOKE, "train_rows": TRAIN_ROWS, "lr": LR, "val_rows": len(val)}
    # TMR was trained on RAID train, so this split flatters it; only run the
    # control in smoke runs, as a pipeline check.
    if SMOKE:
        metrics["control"] = control(val)
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=1)

    tokenize_train()
    t0 = time.time()
    subprocess.run(["torchrun", "--nproc_per_node", str(torch.cuda.device_count()), __file__, "--train"], check=True)
    metrics["train_seconds"] = round(time.time() - t0)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"{OUT}/model")
    model = AutoModelForSequenceClassification.from_pretrained(f"{OUT}/model")
    val["ours"] = score(model, tok, val.text.tolist())
    metrics["ours"] = raid_scores(val, "ours")
    print("ours", json.dumps({m: metrics["ours"][m] for m in ("tpr_all", "tpr_noadv", "auroc")}), flush=True)
    print("ours by attack", metrics["ours"]["attack"], flush=True)
    metrics["pairs_holdout"] = pairs_eval(model, tok)
    print("pairs holdout", metrics["pairs_holdout"], flush=True)
    metrics["mage_ood"] = mage_ood(model, tok)
    print("mage ood", metrics["mage_ood"], flush=True)
    metrics["test_predictions"] = predict_test(model, tok)
    print("test predictions", metrics["test_predictions"], flush=True)
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=1)
    metrics["export"] = export(val)
    print("export", metrics["export"], flush=True)
    val.drop(columns=["generation", "text"]).to_parquet(f"{OUT}/val_scores.parquet", index=False)
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=1)


if __name__ == "__main__":
    if "--train" in sys.argv:
        train()
    else:
        main()
