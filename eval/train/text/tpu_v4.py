
# ---------- v4 on a Kaggle TPU v5e-8 (appended after normalize.py + train_raid.py) ----------
# Same data as v3 (RAID + MAGE) plus ~100k extra human rows from commercially
# usable sources, to cut false positives on encyclopedic and non-native
# academic writing: Wikipedia 2023-11 (CC BY-SA), PMC open-access sections
# (CC BY), arXiv abstracts (CC0 metadata), Project Gutenberg (public domain).
# Any training text that shares 8-grams with the TextSight human sets used by
# GPTTrace's in-house eval is dropped, so that eval stays a fair test.
# Training runs data-parallel over the 8 chips with torch_xla, in bf16, on
# fixed-shape batches (one XLA graph per length bucket, not per batch).
# GT_DEVICE=cuda runs the same data, step order and evaluation on Kaggle's
# 2xT4 instead (torchrun DDP, fp16): the fallback when the TPU queue stalls.
import contextlib
import shutil
import hashlib
import random as _random

OUT = os.environ.get("GT_OUTDIR", OUT)                 # per-job output folder
STATE = os.environ.get("GT_STATE", "/tmp/v4_state.pt")
DEVICE = os.environ.get("GT_DEVICE", "__DEVICE__")
if DEVICE.startswith("__"):
    DEVICE = "tpu"
HUMAN_N = 3_000 if SMOKE else int(os.environ.get("GT_HUMAN", "__HUMAN__"))
EPOCHS = float(os.environ.get("GT_EPOCHS", "__EPOCHS__"))
PER_CORE = 8                      # x 8 cores = 64 rows per step, as v3 (16 x 2 accum x 2 GPUs)
BUCKETS = (64, 128, 256, 384, 512)
TS = "https://huggingface.co/datasets/textsightai/ai-text-detection-benchmark-2026/resolve/main/data/"
TS_HUMAN = ("human_literature_publicdomain.jsonl", "human_pmc_ccby_nonanglophone.jsonl", "human_wikipedia_pre2022.jsonl")


def _grams(t, n=8):
    w = t.lower().split()
    return {hashlib.blake2b(" ".join(w[i:i + n]).encode(), digest_size=8).digest() for i in range(0, max(0, len(w) - n + 1))}


def eval_grams():
    import urllib.request
    g = set()
    for f in TS_HUMAN:
        for line in urllib.request.urlopen(TS + f).read().decode().splitlines():
            if line.strip():
                g |= _grams(json.loads(line).get("text", ""))
    print("textsight eval 8-grams", len(g), flush=True)
    return g


def extra_human():
    """Human rows by source, minus anything overlapping the in-house eval."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs, rng = HfFileSystem(), _random.Random(0)
    per = HUMAN_N // 4

    def ok(t, lo=120, hi=600):
        return lo <= len(t.split()) <= hi

    def groups(pattern, cols, max_files=8):
        files = sorted(fs.glob(pattern))
        rng.shuffle(files)
        for f in files[:max_files]:
            pf = pq.ParquetFile(fs.open(f))
            for g in rng.sample(range(pf.num_row_groups), pf.num_row_groups):
                yield pf.read_row_group(g, columns=cols).to_pandas()

    rows = []

    def take(src, it):
        n = 0
        for t in it:
            if ok(t):
                rows.append((src, t)); n += 1
                if n >= per:
                    break
        print("human source", src, n, flush=True)

    def wiki():
        for t in groups("datasets/wikimedia/wikipedia/20231101.en/*.parquet", ["text"]):
            for x in t.text:
                yield "\n\n".join(x.split("\n\n")[:4])

    def pmc():
        # The commercial-use subset, stored as Arrow IPC streams (not parquet).
        import pyarrow as pa
        files = sorted(fs.glob("datasets/TomTBT/pmc_open_access_section/data/commercial/*.arrow"))
        rng.shuffle(files)
        for f in files[:6]:
            with fs.open(f, "rb") as fh:
                reader = pa.ipc.open_stream(fh)
                cols = [c for c in reader.schema.names if c.lower() in ("introduction", "discussion", "methods", "results", "conclusion")]
                for batch in reader:
                    for r in batch.select(cols).to_pylist():
                        c = rng.choice(cols)
                        v = r.get(c)
                        v = " ".join(v) if isinstance(v, (list, tuple)) else str(v or "")
                        yield " ".join(v.split()[:450])

    def arxiv():
        # One gzipped JSONL file; read it as a stream.
        import gzip
        with fs.open("datasets/gfissore/arxiv-abstracts-2021/arxiv-abstracts.jsonl.gz", "rb") as raw, gzip.open(raw, "rt") as fh:
            for line in fh:
                if rng.random() < 0.05:
                    yield " ".join(str(json.loads(line).get("abstract", "")).split())

    def gutenberg():
        for t in groups("datasets/manu/project_gutenberg/data/en-*.parquet", ["text"], max_files=4):
            for x in t.text:
                paras = [p for p in str(x).split("\n\n") if len(p.split()) > 40]
                for _ in range(8 if len(paras) > 40 else 0):
                    i = rng.randrange(10, len(paras) - 10)
                    yield "\n\n".join(paras[i:i + 3])

    for src, it in (("wikipedia", wiki()), ("pmc", pmc()), ("arxiv", arxiv()), ("gutenberg", gutenberg())):
        try:
            take(src, it)
        except Exception as e:
            print("human source failed", src, repr(e)[:300], flush=True)
    d = pd.DataFrame(rows, columns=["source", "text"])
    d["text"] = d.text.map(normalize_text)
    eg = eval_grams()
    keep = [len(_grams(t) & eg) < 3 for t in d.text]
    print("dropped for eval overlap", int(len(d) - sum(keep)), "of", len(d), flush=True)
    d = d[keep].reset_index(drop=True)
    d["label"] = 0
    d["holdout"] = d.groupby("source").cumcount() % 20 == 0     # 5% per source for the FPR check
    d.to_parquet("/tmp/extra_human.parquet", index=False)
    return d


def tokenize_train():
    """RAID (+ paraphrase x2) + MAGE + extra human rows, tokenised once into a
    flat int32 array the 8 training processes memory-map."""
    from transformers import AutoTokenizer
    tr = load("train", TRAIN_ROWS)
    para = tr[(tr.attack == "paraphrase") & (tr.model != "human")]
    tr = pd.concat([tr, para]).sample(frac=1, random_state=1).reset_index(drop=True)
    tr["text"] = tr.generation.map(normalize_text)
    tr["label"] = (tr.model != "human").astype(int)
    parts = [tr[["text", "label"]]]
    if MAGE_ROWS:
        mg = pd.read_csv(MAGE + "train.csv").sample(n=MAGE_ROWS, random_state=0)
        parts.append(pd.DataFrame({"text": mg.text.astype(str).map(normalize_text), "label": 1 - mg.label.astype(int)}))
    eh = extra_human()
    parts.append(eh[~eh.holdout][["text", "label"]])
    # Topic-matched pairs with paraphrase/humanizer rewrites (gen/gen_pairs.py),
    # when that kernel's output is mounted. Same label convention: 1 = AI.
    pr = load_pairs()
    if pr is not None:
        pr = pr[~pr.holdout]
        print("pairs rows", len(pr), pr.groupby(["source", "label"]).size().to_dict(), flush=True)
        parts.append(pr[["text", "label"]])
    tr = pd.concat(parts).sample(frac=1, random_state=2).reset_index(drop=True)
    print("train rows", len(tr), "human share", round(float((tr.label == 0).mean()), 3), flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    chunks, lens = [], []
    for i in range(0, len(tr), 4096):
        for x in tok(tr.text[i:i + 4096].tolist(), truncation=True, max_length=MAX_LEN)["input_ids"]:
            chunks.append(np.asarray(x, dtype=np.int32)); lens.append(len(x))
    offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    np.save("/tmp/tok_ids.npy", np.concatenate(chunks)); del chunks
    np.save("/tmp/tok_offs.npy", offs)
    np.save("/tmp/tok_labels.npy", tr.label.to_numpy(dtype=np.int64))
    # Global batches of similar length (shuffle, then sort inside 100-batch
    # windows), each padded to a bucket so XLA compiles one graph per bucket.
    lens = np.diff(offs)
    rng = np.random.default_rng(0)
    gb = PER_CORE * 8
    steps = []
    for ep in range(int(np.ceil(EPOCHS))):
        order = rng.permutation(len(lens))
        if ep + 1 > EPOCHS:                                   # fractional last epoch
            order = order[: int(len(order) * (EPOCHS - ep))]
        for w in range(0, len(order), gb * 100):
            win = order[w:w + gb * 100]
            win = win[np.argsort(lens[win], kind="stable")]
            for b in range(0, len(win) - gb + 1, gb):
                steps.append(win[b:b + gb])
    rng.shuffle(steps)
    np.save("/tmp/tok_steps.npy", np.stack(steps))
    print("tokenized", len(lens), "rows,", len(steps), "steps of", gb, flush=True)
    return eh


def _bucket(n):
    return next(b for b in BUCKETS if b >= n)


def train_cuda():
    """DDP over the GPUs (torchrun): each global step's 64 rows are split
    across ranks, then into micro-batches of 16, so the effective batch and
    the LR schedule match the TPU run."""
    import torch.distributed as dist
    from transformers import AutoModelForSequenceClassification
    gpu = torch.cuda.is_available()                           # CPU + gloo only for local smoke tests
    dist.init_process_group("nccl" if gpu else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if gpu:
        torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank) if gpu else torch.device("cpu")
    ids = np.load("/tmp/tok_ids.npy", mmap_mode="r")
    offs = np.load("/tmp/tok_offs.npy")
    labels = np.load("/tmp/tok_labels.npy")
    steps = np.load("/tmp/tok_steps.npy")
    if SMOKE:
        steps = steps[:60]
    teacher = np.load("/tmp/teacher.npy") if os.path.exists("/tmp/teacher.npy") else None
    torch.manual_seed(0)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2).to(dev)
    big = sum(p.numel() for p in model.parameters()) > 300e6
    if big:                                                   # 400M/1B: trade compute for memory
        model.gradient_checkpointing_enable()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank] if gpu else None, find_unused_parameters=False)
    model.train()
    decay = [p for nme, p in model.named_parameters() if not any(k in nme for k in ("bias", "norm"))]
    rest = [p for nme, p in model.named_parameters() if any(k in nme for k in ("bias", "norm"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": rest, "weight_decay": 0.0}], lr=LR)
    total, warm = len(steps), max(1, int(0.05 * len(steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: st / warm if st < warm else 0.5 * (1 + np.cos(np.pi * (st - warm) / max(1, total - warm))))
    bf16 = gpu and torch.cuda.is_bf16_supported()             # A100/L4/H100: bf16, no loss scaling; T4: fp16 + scaler
    scaler = torch.amp.GradScaler("cuda", enabled=gpu and not bf16)
    MICRO = 16
    if rank == 0:
        print(f"DDP over {world} GPUs ({'bf16' if bf16 else 'fp16'}{', checkpointing' if big else ''}), {total} steps of {len(steps[0])}", flush=True)
    # Checkpoints to a private HF repo (GT_CKPT_REPO; Colab runs) every
    # CKPT_EVERY steps, so a reclaimed VM costs at most that much training.
    # Resuming reloads the weights and skips finished steps; the optimiser
    # state is not saved, so the LR re-warms over 100 steps.
    ckpt_repo, start = os.environ.get("GT_CKPT_REPO"), 0
    CKPT_EVERY = int(os.environ.get("GT_CKPT_EVERY", "1500"))
    api, uploads = None, []
    if ckpt_repo and os.environ.get("HF_TOKEN"):
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(ckpt_repo, private=True, exist_ok=True)
        try:
            ck = torch.load(hf_hub_download(ckpt_repo, "ckpt/ckpt.pt", token=os.environ["HF_TOKEN"]), map_location="cpu")
            model.module.load_state_dict({k: v.float() for k, v in ck["model"].items()})
            start = ck["step"] + 1
            for _ in range(start):
                sched.step()
            if rank == 0:
                print(f"resumed from {ckpt_repo} at step {start}", flush=True)
        except Exception as e:
            if rank == 0:
                print("no checkpoint to resume:", type(e).__name__, flush=True)
    rewarm = 100 if start else 0
    t0 = time.time()
    for st, rows in enumerate(steps):
        if st < start:
            continue
        if rewarm and st - start < rewarm:
            for g in opt.param_groups:
                g["lr"] = sched.get_last_lr()[0] * (st - start + 1) / rewarm
        mine = rows[rank::world]
        chunks = [mine[i:i + MICRO] for i in range(0, len(mine), MICRO)]
        for ci, part in enumerate(chunks):
            L = int(max(offs[r + 1] - offs[r] for r in part))
            x = np.zeros((len(part), L), dtype=np.int64); m = np.zeros((len(part), L), dtype=np.int64)
            for j, r in enumerate(part):
                seq = ids[offs[r]:offs[r + 1]]
                x[j, :len(seq)] = seq; m[j, :len(seq)] = 1
            ctx = model.no_sync() if ci < len(chunks) - 1 else contextlib.nullcontext()
            with ctx:                                           # all-reduce only on the last micro-batch
                with torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16) if gpu else contextlib.nullcontext():
                    out_ = model(input_ids=torch.from_numpy(x).to(dev), attention_mask=torch.from_numpy(m).to(dev), labels=torch.from_numpy(labels[part]).to(dev))
                    loss = out_.loss
                    if teacher is not None:                   # knowledge distillation (Hinton et al.), T = 2
                        tt = torch.from_numpy(teacher[part].astype(np.float32)).to(dev)
                        kd = torch.nn.functional.kl_div(torch.log_softmax(out_.logits.float() / 2, -1), torch.softmax(tt / 2, -1), reduction="batchmean") * 4
                        loss = 0.5 * loss + 0.5 * kd
                    loss = loss / len(chunks)
                scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
        uploads = [u for u in uploads if u.is_alive()]
        if api and rank == 0 and st and st % CKPT_EVERY == 0 and st < total - 1 and not uploads:   # skip while one is still uploading
            import threading
            torch.save({"step": st, "model": {k: v.detach().to("cpu", torch.bfloat16) for k, v in model.module.state_dict().items()}}, "/tmp/ckpt.pt")
            th = threading.Thread(target=lambda: api.upload_file(path_or_fileobj="/tmp/ckpt.pt", path_in_repo="ckpt/ckpt.pt", repo_id=ckpt_repo, commit_message=f"checkpoint step {st}"))
            th.start(); uploads.append(th)
            print(f"checkpoint step {st} → {ckpt_repo}", flush=True)
        if rank == 0 and (st % 500 == 0 or st == total - 1):
            el = time.time() - t0
            print(f"step {st}/{total} loss {loss.item() * len(chunks):.4f} lr {sched.get_last_lr()[0]:.2e} {(st + 1) * len(rows) / max(el, 1e-9):.0f} rows/s", flush=True)
    if rank == 0:
        print(f"trained {total * len(steps[0])} rows in {time.time() - t0:.0f}s", flush=True)
        torch.save({k: v.cpu() for k, v in model.module.state_dict().items()}, STATE)
    dist.barrier()
    dist.destroy_process_group()


def train():
    """Single-process SPMD over all chips: each global batch of 64 rows is
    sharded along a 'data' mesh axis and XLA inserts the gradient all-reduce.
    (Multi-process torch_xla.launch failed on Kaggle's v5e-8: 'Expected 8
    worker addresses, got 1'.)"""
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.spmd as xs
    from transformers import AutoModelForSequenceClassification
    xr.use_spmd()
    n = xr.global_runtime_device_count()
    mesh = xs.Mesh(np.arange(n), (n, 1), ("data", "model"))
    dev = torch_xla.device()
    ids = np.load("/tmp/tok_ids.npy", mmap_mode="r")
    offs = np.load("/tmp/tok_offs.npy")
    labels = np.load("/tmp/tok_labels.npy")
    steps = np.load("/tmp/tok_steps.npy")
    if SMOKE:
        steps = steps[:60]
    torch.manual_seed(0)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2, attn_implementation="eager").to(dev)
    model.train()
    decay = [p for nme, p in model.named_parameters() if not any(k in nme for k in ("bias", "norm"))]
    rest = [p for nme, p in model.named_parameters() if any(k in nme for k in ("bias", "norm"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": rest, "weight_decay": 0.0}], lr=LR)
    total, warm = len(steps), max(1, int(0.05 * len(steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: st / warm if st < warm else 0.5 * (1 + np.cos(np.pi * (st - warm) / max(1, total - warm))))
    print(f"SPMD over {n} devices, {total} steps of {len(steps[0])}", flush=True)
    t0 = time.time()
    for st, rows in enumerate(steps):
        L = _bucket(int(max(offs[r + 1] - offs[r] for r in rows)))
        x = np.zeros((len(rows), L), dtype=np.int64)
        m = np.zeros((len(rows), L), dtype=np.int64)
        for j, r in enumerate(rows):
            seq = ids[offs[r]:offs[r + 1]]
            x[j, :len(seq)] = seq; m[j, :len(seq)] = 1
        xb, mb, yb = torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev), torch.from_numpy(labels[rows]).to(dev)
        xs.mark_sharding(xb, mesh, ("data", None)); xs.mark_sharding(mb, mesh, ("data", None)); xs.mark_sharding(yb, mesh, ("data",))
        with torch.autocast("xla", dtype=torch.bfloat16):
            loss = model(input_ids=xb, attention_mask=mb, labels=yb).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        xm.mark_step()
        if st % 500 == 0 or st == total - 1:
            el = time.time() - t0
            print(f"step {st}/{total} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {(st + 1) * len(rows) / max(el, 1e-9):.0f} rows/s", flush=True)
    print(f"trained {total * len(steps[0])} rows in {time.time() - t0:.0f}s", flush=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, STATE)


def score(model, tok, texts, bs=64, device=None):
    """Single-chip XLA (or one GPU) scoring on bucketed, fixed-size batches."""
    if DEVICE == "cuda":
        dev = torch.device("cuda")
    else:
        import torch_xla
        dev = torch_xla.device()
    model.eval().to(dev)
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN)["input_ids"]
    order = np.argsort([len(e) for e in enc])
    out = np.zeros(len(texts), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            L = _bucket(max(len(enc[j]) for j in idx))
            x = np.zeros((bs, L), dtype=np.int64); m = np.zeros((bs, L), dtype=np.int64)
            for k, j in enumerate(idx):
                x[k, :len(enc[j])] = enc[j]; m[k, :len(enc[j])] = 1
            m[len(idx):, 0] = 1                                   # dummy rows: avoid all-masked
            with torch.autocast("cuda" if DEVICE == "cuda" else "xla", dtype=torch.float16 if DEVICE == "cuda" else torch.bfloat16):
                lg = model(input_ids=torch.from_numpy(x).to(dev), attention_mask=torch.from_numpy(m).to(dev)).logits.float()
            lg = lg.cpu().numpy()[: len(idx)]
            out[idx] = lg[:, 1] - lg[:, 0]
    return out


def human_holdout(model, tok, eh, val):
    """False positives on held-out extra human rows, at RAID's 5%-FPR threshold
    over its unattacked humans and at logit 0."""
    h = eh[eh.holdout].reset_index(drop=True)
    h["score"] = score(model, tok, h.text.tolist())
    hv = val[(val.model == "human") & (val.attack == "none")].ours.to_numpy()
    thr = float(np.sort(hv)[::-1][int(0.05 * len(hv))])
    return {src: {"n": len(g), "flagged_at_0": round(float((g.score > 0).mean()), 4), "flagged_at_raid5": round(float((g.score > thr).mean()), 4)}
            for src, g in h.groupby("source")}


# One session can train several models (GT_JOBS, "base[:tag],base[:tag]"):
# the data is tokenised once (the Ettin sizes share a tokenizer) and every
# step that touches the accelerator runs in its own child process, because
# a process that has opened the TPU holds it until it exits.
# GT_JOBS: "base[:tag[:lr]],…"
JOBS = [j for j in os.environ.get("GT_JOBS", "__JOBS__").split(",") if j and not j.startswith("__")] or [BASE_MODEL]
BUDGET_H = float(os.environ.get("GT_BUDGET_H", "8.3"))      # Kaggle stops TPU sessions at 9 h
EST_H = {"17m": 0.3, "32m": 0.4, "68m": 0.6, "150m": 1.6, "400m": 3.6, "1b": 8.0}   # train + eval + export, v5e-8, ~900k rows


def _job(spec):
    """Parse "base[:tag[:lr]]" into (base, tag, lr or None)."""
    base, tag, lr = (spec.split(":") + ["", ""])[:3]
    return base, tag or base.split("/")[-1], lr or None


def _env(base, tag, lr=None):
    work = os.environ.get("GT_WORK", "/kaggle/working")
    env = {**os.environ, "GT_BASE": base, "GT_OUTDIR": f"{work}/{tag}", "GT_STATE": f"/tmp/{tag}_state.pt"}
    if lr:
        env["GT_LR"] = lr
    if os.environ.get("HF_TOKEN"):
        env["GT_CKPT_REPO"] = f"nakas/gpttrace-{tag}"
    return env


def _install_xla():
    """torch 2.8.0 + torch_xla 2.8.0 is the pairing verified on Kaggle's
    v5e-8; torch_xla 2.9 has crashed with std::bad_alloc while loading
    models there. Keep the image's build only if it is older than 2.9."""
    ok = subprocess.run([sys.executable, "-c", "import torch_xla, sys; v = tuple(int(x) for x in torch_xla.__version__.split('+')[0].split('.')[:2]); sys.exit(0 if v < (2, 9) else 1)"])
    if ok.returncode:
        print("installing torch 2.8.0 + torch_xla 2.8.0", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.8.0", "torch_xla[tpu]==2.8.0", "-f", "https://storage.googleapis.com/libtpu-releases/index.html"], check=True)


@torch.no_grad()
def teacher_logits(prep_repo):
    """GT_STAGE=teacher: the trained teacher (GT_TEACHER_REPO, model/) scores
    every training row; the [N, 2] logits go to the prep repo as
    teacher_<tag>.npy for students to distil from (aligned with tok_*.npy)."""
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModelForSequenceClassification
    repo, tok_ = os.environ["GT_TEACHER_REPO"], os.environ.get("HF_TOKEN")
    d = snapshot_download(repo, allow_patterns=["model/*"], token=tok_)
    model = AutoModelForSequenceClassification.from_pretrained(f"{d}/model", attn_implementation="eager").to("cuda").eval()
    ids = np.load("/tmp/tok_ids.npy", mmap_mode="r"); offs = np.load("/tmp/tok_offs.npy")
    n = len(offs) - 1
    order = np.argsort(np.diff(offs))
    out = np.zeros((n, 2), dtype=np.float32)
    t0, bs = time.time(), 128
    for i in range(0, n, bs):
        idx = order[i:i + bs]
        L = int(max(offs[r + 1] - offs[r] for r in idx))
        x = np.zeros((len(idx), L), dtype=np.int64); m = np.zeros((len(idx), L), dtype=np.int64)
        for j, r in enumerate(idx):
            seq = ids[offs[r]:offs[r + 1]]
            x[j, :len(seq)] = seq; m[j, :len(seq)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(input_ids=torch.from_numpy(x).cuda(), attention_mask=torch.from_numpy(m).cuda()).logits.float()
        out[idx] = lg.cpu().numpy()
        if (i // bs) % 500 == 0:
            print(f"teacher {i}/{n} {(i + bs) / (time.time() - t0):.0f} rows/s", flush=True)
    name = f"teacher_{repo.split('gpttrace-')[-1]}.npy"
    np.save(f"/tmp/{name}", out.astype(np.float16))
    HfApi(token=tok_).upload_file(path_or_fileobj=f"/tmp/{name}", path_in_repo=name, repo_id=prep_repo, repo_type="dataset", commit_message=f"teacher logits from {repo}")
    print("teacher logits uploaded:", name, out.shape, flush=True)


def main():
    t_start = time.time()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "scikit-learn", "pyarrow", "huggingface_hub"], check=False)
    if DEVICE == "tpu":
        # Multi-process launch leftovers that make PJRT expect several hosts.
        for k in ("TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"):
            os.environ.pop(k, None)
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        _install_xla()
        # Fail fast (before the data build) if one chip can't train the model.
        chk = subprocess.run([sys.executable, __file__, "--check"], capture_output=True, text=True)
        print("xla check rc", chk.returncode, chk.stdout[-800:], chk.stderr[-1500:], flush=True)
        if chk.returncode:
            raise SystemExit("XLA check failed")
    # GT_STAGE=prep (cheap CPU machine): tokenise once and publish the arrays
    # to a private HF dataset (GT_PREP_REPO). GT_STAGE=train (GPU machine):
    # fetch them instead of rebuilding, so the GPU only bills for GPU work.
    stage, prep_repo = os.environ.get("GT_STAGE", "all"), os.environ.get("GT_PREP_REPO")
    if stage in ("train", "teacher") and prep_repo:
        from huggingface_hub import snapshot_download
        d = snapshot_download(prep_repo, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
        for f in ("tok_ids.npy", "tok_offs.npy", "tok_labels.npy", "tok_steps.npy", "extra_human.parquet"):
            shutil.copy(f"{d}/{f}", f"/tmp/{f}")
        os.makedirs(f"{INPUT}/raid", exist_ok=True)
        shutil.copy(f"{d}/raid_val.parquet", f"{INPUT}/raid/raid_val.parquet")
        shutil.copy(f"{d}/raid_val.parquet", f"{INPUT}/raid/raid_train.parquet")   # data_dir() looks for this name
        teach = os.environ.get("GT_TEACHER")                  # distillation targets, e.g. teacher_v5-1b.npy
        if teach and os.path.exists(f"{d}/{teach}"):
            shutil.copy(f"{d}/{teach}", "/tmp/teacher.npy")
            print("distilling from", teach, flush=True)
        if os.path.exists(f"{d}/pairs.parquet"):              # for the pairs-holdout eval
            os.makedirs(f"{INPUT}/pairs", exist_ok=True)
            shutil.copy(f"{d}/pairs.parquet", f"{INPUT}/pairs/pairs.parquet")
        print("prep data from", prep_repo, flush=True)
    else:
        eh = tokenize_train()
        eh.to_parquet("/tmp/extra_human.parquet")
    if stage == "teacher":
        teacher_logits(prep_repo)
        return
    if stage == "prep":
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(prep_repo, repo_type="dataset", private=True, exist_ok=True)
        os.makedirs("/tmp/prep", exist_ok=True)
        for f in ("tok_ids.npy", "tok_offs.npy", "tok_labels.npy", "tok_steps.npy", "extra_human.parquet"):
            shutil.copy(f"/tmp/{f}", f"/tmp/prep/{f}")
        shutil.copy(f"{data_dir()}/raid_val.parquet", "/tmp/prep/raid_val.parquet")
        api.upload_folder(repo_id=prep_repo, repo_type="dataset", folder_path="/tmp/prep", commit_message="tokenised training data")
        print("prep uploaded to", prep_repo, flush=True)
        return
    done = []
    for spec in JOBS:
        base, tag, lr = _job(spec)
        size = next((k for k in EST_H if base.lower().endswith(k)), "400m")
        left = BUDGET_H - (time.time() - t_start) / 3600
        if not SMOKE and EST_H[size] > left:
            print(f"skip {tag}: needs ~{EST_H[size]} h, {left:.1f} h left", flush=True)
            continue
        env = _env(base, tag, lr)
        os.makedirs(env["GT_OUTDIR"], exist_ok=True)
        t0 = time.time()
        try:
            if DEVICE == "cuda":
                subprocess.run(["torchrun", "--nproc_per_node", str(torch.cuda.device_count()), __file__, "--train"], check=True, env=env)
            else:
                subprocess.run([sys.executable, __file__, "--train"], check=True, env={**env, "XLA_USE_SPMD": "1"})
            env["GT_TRAIN_SECONDS"] = str(round(time.time() - t0))
            subprocess.run([sys.executable, __file__, "--eval"], check=True, env=env)
            done.append(tag)
        except subprocess.CalledProcessError as e:
            # Keep going: a later job may still fit, and earlier outputs stay.
            print(f"job {tag} failed: {e}", flush=True)
    print("finished jobs:", done, f"in {(time.time() - t_start) / 3600:.2f} h", flush=True)
    if not done:
        raise SystemExit("no job finished")


def eval_job():
    """Child process: score the trained model (RAID val, held-out human text,
    MAGE OOD), save it, and export ONNX + int8."""
    val = load("val", VAL_ROWS)
    val["text"] = val.generation.map(normalize_text)
    eh = pd.read_parquet("/tmp/extra_human.parquet")
    metrics = {"base": BASE_MODEL, "smoke": SMOKE, "train_rows": TRAIN_ROWS, "mage_rows": MAGE_ROWS, "human_rows": HUMAN_N, "epochs": EPOCHS, "lr": LR, "val_rows": len(val),
               "device": "tpu-v5e-8" if DEVICE == "tpu" else f"cuda x{torch.cuda.device_count()}", "train_seconds": int(os.environ.get("GT_TRAIN_SECONDS", "0"))}
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2, attn_implementation="eager")
    model.load_state_dict(torch.load(STATE, map_location="cpu"))
    os.makedirs(f"{OUT}/model", exist_ok=True)
    model.save_pretrained(f"{OUT}/model"); tok.save_pretrained(f"{OUT}/model")
    val["ours"] = score(model, tok, val.text.tolist())
    metrics["ours"] = raid_scores(val, "ours")
    print("ours", json.dumps({m: metrics["ours"][m] for m in ("tpr_all", "tpr_noadv", "auroc")}), flush=True)
    metrics["human_holdout"] = human_holdout(model, tok, eh, val)
    print("human holdout", metrics["human_holdout"], flush=True)
    metrics["mage_ood"] = mage_ood(model, tok)
    print("mage ood", metrics["mage_ood"], flush=True)
    metrics["pairs_holdout"] = pairs_eval(model, tok)
    print("pairs holdout", metrics["pairs_holdout"], flush=True)
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=1)
    if os.environ.get("GT_SKIP_EXPORT") == "1":              # teachers: weights only (ONNX >2 GB fails anyway)
        return
    # ONNX export and int8 selection run on CPU, comparing against torch scores.
    model = model.to("cpu")
    metrics["export"] = export(val)
    print("export", metrics["export"], flush=True)
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=1)


def xla_check():
    import torch_xla
    import torch_xla.core.xla_model as xm
    from transformers import AutoModelForSequenceClassification
    dev = torch_xla.device()
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2, attn_implementation="eager").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    x = torch.randint(5, 50000, (PER_CORE, 512)).to(dev); m = torch.ones(PER_CORE, 512, dtype=torch.long).to(dev); y = torch.randint(0, 2, (PER_CORE,)).to(dev)
    for i in range(8):
        if i == 3:
            xm.wait_device_ops(); t0 = time.time()
        with torch.autocast("xla", dtype=torch.bfloat16):
            loss = model(input_ids=x, attention_mask=m, labels=y).loss
        loss.backward(); opt.step(); opt.zero_grad(); xm.mark_step()
    xm.wait_device_ops()
    print(f"xla ok: {5 * PER_CORE / (time.time() - t0):.1f} rows/s per chip at L=512, loss {loss.item():.3f}", flush=True)


if __name__ == "__main__":
    if "--check" in sys.argv:
        xla_check()
    elif "--train" in sys.argv:
        train_cuda() if DEVICE == "cuda" else train()
    elif "--eval" in sys.argv:
        eval_job()
    else:
        main()
