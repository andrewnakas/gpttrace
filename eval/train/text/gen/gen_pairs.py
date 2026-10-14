"""Kaggle job (GPU T4x2, internet): build topic-matched human/AI text pairs
for GPTTrace's text model, from commercially usable sources only.

Human (licence): PMC open-access sections (CC BY 4.0) — much of it written by
non-native English authors; Wikipedia 2023-11 (CC BY-SA); Project Gutenberg
(public domain); arXiv abstracts (CC0 metadata).
AI: open-weight instruct models (Apache-2.0 / MIT) write a text on the same
topic, of similar length, for each human text, so topic can't be a shortcut.

Attacks: a third of the AI texts are then rewritten by a different model
with "humanizer" or paraphrase instructions (the style of rewrite that
paraphrasers and AI humanizers apply). They stay labelled AI, with source
"<source>+para" and model "<writer>><rewriter>". Paraphrase is the measured
weak spot of the current detector (MAGE paraphrased AUROC 0.742).

Writes /kaggle/working/pairs.parquet: text, label (1 = AI), source, model."""
import os
import random
import subprocess
import sys

if not os.environ.get("GT_NO_PIP"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

N_PER_SOURCE = int(os.environ.get("GT_N", "3000"))
MODELS = ["Qwen/Qwen2.5-7B-Instruct-AWQ", "Qwen/Qwen3-8B-AWQ", "microsoft/Phi-4-mini-instruct"]
OUT = os.environ.get("GT_OUT", "/kaggle/working")
TP = int(os.environ.get("GT_TP", "2"))              # 2 on Kaggle's 2xT4, 1 on an A100
DTYPE = os.environ.get("GT_DTYPE", "float16")       # T4 has no bfloat16
rng = random.Random(0)
fs = HfFileSystem()


def words(t, lo=120, hi=600):
    n = len(t.split())
    return lo <= n <= hi


def sample_parquet(pattern, col, n, max_files=6, extra=None):
    files = sorted(fs.glob(pattern))
    rng.shuffle(files)
    out = []
    for f in files[:max_files]:
        pf = pq.ParquetFile(fs.open(f))
        for g in rng.sample(range(pf.num_row_groups), pf.num_row_groups):
            cols = [col] + ([extra] if extra else [])
            t = pf.read_row_group(g, columns=cols).to_pandas()
            for r in t.itertuples(index=False):
                out.append(r)
            if len(out) >= n * 6:
                break
        if len(out) >= n * 6:
            break
    return out


def human_rows():
    rows = []
    # Wikipedia: first ~section of an article; title is the topic.
    for r in sample_parquet("datasets/wikimedia/wikipedia/20231101.en/*.parquet", "text", N_PER_SOURCE, extra="title"):
        t = "\n\n".join(r.text.split("\n\n")[:4])
        if words(t):
            rows.append(("wikipedia", t, f"Write a Wikipedia-style encyclopedia introduction about \"{r.title}\"."))
        if sum(1 for x in rows if x[0] == "wikipedia") >= N_PER_SOURCE:
            break
    # PMC open-access sections (CC BY): academic prose.
    n = 0
    try:
        files = sorted(fs.glob("datasets/TomTBT/pmc_open_access_section/**/*.parquet"))
        rng.shuffle(files)
        pf = pq.ParquetFile(fs.open(files[0]))
        print("pmc columns", pf.schema_arrow.names, flush=True)
        textcols = [c for c in pf.schema_arrow.names if c.lower() in ("introduction", "discussion", "abstract", "methods", "results", "conclusion")]
        for f in files[:8]:
            pf = pq.ParquetFile(fs.open(f))
            for g in range(pf.num_row_groups):
                t = pf.read_row_group(g).to_pandas()
                for r in t.to_dict("records"):
                    for c in textcols:
                        v = r.get(c)
                        v = " ".join(v) if isinstance(v, (list, tuple)) else (v or "")
                        v = " ".join(str(v).split()[:450])
                        if words(v, 150, 600):
                            title = r.get("title") or r.get("article_title") or "a biomedical study"
                            rows.append(("pmc-" + c.lower(), v, f"Write the {c.lower()} section of a scientific paper titled \"{title}\"."))
                            n += 1
                            break
                    if n >= N_PER_SOURCE:
                        break
                if n >= N_PER_SOURCE:
                    break
            if n >= N_PER_SOURCE:
                break
    except Exception as e:
        print("pmc failed", repr(e)[:300], flush=True)
    # arXiv abstracts (CC0 metadata).
    for r in sample_parquet("datasets/gfissore/arxiv-abstracts-2021/**/*.parquet", "abstract", N_PER_SOURCE, extra="title"):
        t = " ".join(str(r.abstract).split())
        if words(t, 100, 400):
            rows.append(("arxiv", t, f"Write the abstract of a research paper titled \"{' '.join(str(r.title).split())}\"."))
        if sum(1 for x in rows if x[0] == "arxiv") >= N_PER_SOURCE:
            break
    # Gutenberg: a passage from the middle of a book; continue the story.
    try:
        for r in sample_parquet("datasets/manu/project_gutenberg/data/en-*.parquet", "text", N_PER_SOURCE // 6, max_files=3):
            paras = [p for p in str(r.text).split("\n\n") if len(p.split()) > 40]
            if len(paras) < 40:
                continue
            for _ in range(6):
                i = rng.randrange(10, len(paras) - 10)
                t = "\n\n".join(paras[i:i + 3])
                if words(t, 150, 600):
                    rows.append(("gutenberg", t, "Write a passage of literary fiction in the style of a 19th-century novel, continuing from: \"" + " ".join(paras[i - 1].split()[-40:]) + "\""))
            if sum(1 for x in rows if x[0] == "gutenberg") >= N_PER_SOURCE:
                break
    except Exception as e:
        print("gutenberg failed", repr(e)[:300], flush=True)
    df = pd.DataFrame(rows, columns=["source", "text", "prompt"])
    print(df.source.value_counts().to_dict(), flush=True)
    return df


def generate(df):
    from vllm import LLM, SamplingParams
    outs = []
    for i, name in enumerate(MODELS):
        part = df.iloc[i::len(MODELS)]
        try:
            llm = LLM(name, tensor_parallel_size=TP, dtype=DTYPE, max_model_len=2048, gpu_memory_utilization=0.85, trust_remote_code=True)
            tok = llm.get_tokenizer()
            prompts = []
            for r in part.itertuples():
                n = len(r.text.split())
                msg = [{"role": "user", "content": f"{r.prompt} Aim for about {n} words. Output only the text, with no title or preamble."}]
                kw = {"enable_thinking": False} if "Qwen3" in name else {}
                prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, **kw))
            sp = SamplingParams(temperature=rng.choice([0.7, 0.9, 1.0]), top_p=0.95, max_tokens=900)
            res = llm.generate(prompts, sp)
            for r, o in zip(part.itertuples(), res):
                t = o.outputs[0].text.strip()
                if len(t.split()) >= 60:
                    outs.append((t, 1, r.source, name))
            print(name, "generated", len(res), flush=True)
            del llm
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(name, "FAILED", repr(e)[:400], flush=True)
    return outs


REWRITE = [
    "Paraphrase the following text so that it keeps the meaning but uses different wording and sentence structure. Output only the rewritten text.",
    "Rewrite the following text so it reads as if a person wrote it: vary sentence length, use plainer words, allow a little informality, and avoid stock phrases. Output only the rewritten text.",
    "Humanize this text to avoid AI detection: change the rhythm, add natural transitions, mix short and long sentences, and keep every fact. Output only the result.",
]
PARA_FRAC = float(os.environ.get("GT_PARA", "0.33"))


def paraphrase(ai):
    """Rewrite a share of the AI texts with a different model than wrote them."""
    from vllm import LLM, SamplingParams
    picked = [a for a in ai if rng.random() < PARA_FRAC]
    outs = []
    for i, name in enumerate(MODELS):
        part = [a for a in picked if a[3] != name][i::len(MODELS)]
        if not part:
            continue
        try:
            llm = LLM(name, tensor_parallel_size=TP, dtype=DTYPE, max_model_len=2048, gpu_memory_utilization=0.85, trust_remote_code=True)
            tok = llm.get_tokenizer()
            kw = {"enable_thinking": False} if "Qwen3" in name else {}
            prompts = [tok.apply_chat_template([{"role": "user", "content": rng.choice(REWRITE) + "\n\n" + " ".join(a[0].split()[:600])}], tokenize=False, add_generation_prompt=True, **kw) for a in part]
            res = llm.generate(prompts, SamplingParams(temperature=0.9, top_p=0.95, max_tokens=900))
            for a, o in zip(part, res):
                t = o.outputs[0].text.strip()
                if len(t.split()) >= 60:
                    outs.append((t, 1, a[2] + "+para", a[3] + ">" + name))
            print(name, "paraphrased", len(res), flush=True)
            del llm
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(name, "paraphrase FAILED", repr(e)[:400], flush=True)
    return outs


if __name__ == "__main__":
    human = human_rows()
    ai = generate(human)
    ai += paraphrase(ai)
    rows = [(t, 0, s, "human") for t, s in zip(human.text, human.source)] + ai
    out = pd.DataFrame(rows, columns=["text", "label", "source", "model"])
    out.to_parquet(f"{OUT}/pairs.parquet", index=False)
    print(out.groupby(["source", "label"]).size().to_dict(), flush=True)
