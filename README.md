# GPTTrace

Open detectors for AI-generated text, images and voice that run entirely in
the browser, plus the evaluation and training code behind every number we
publish. This is the engine behind [gpttrace.com](https://gpttrace.com/) and
the GPTTrace browser extension.

- **Private by design:** the free detectors run on your device with ONNX Runtime
  Web (WASM). Nothing you check is uploaded.
- **Measured, reproducible:** every accuracy figure comes from `eval/` and can
  be re-run. We publish weak spots as well as strengths.
- **Open weights** on Hugging Face:
  - [gpttrace-ettin150-text-v3](https://huggingface.co/nakas/gpttrace-ettin150-text-v3) (text, MIT)
  - [gpttrace-cf-vit-ft-v1](https://huggingface.co/nakas/gpttrace-cf-vit-ft-v1) (images, CC BY-SA 4.0)
  - [gpttrace-wavlm-audio-v3](https://huggingface.co/nakas/gpttrace-wavlm-audio-v3) (voice, CC BY-SA 4.0)

## Models and measured results

| Modality | Model | Result |
|---|---|---|
| Text | **Ettin-1B v5** ([nakas/gpttrace-v5-1b](https://huggingface.co/nakas/gpttrace-v5-1b), CC BY-SA 4.0) | Official [RAID](https://raid-bench.xyz/) test: TPR@5%FPR **99.90%** with attacks, 99.42% at 1% FPR, AUROC 99.94 ([PR #223](https://github.com/liamdugan/raid/pull/223)); MAGE unseen GPT-4 AUROC 0.990, paraphrased 0.811 |
| Text | Ettin-150M fine-tuned on RAID (leaderboard entry) | [RAID leaderboard](https://raid-bench.xyz/): TPR@5%FPR 99.73% with attacks, AUROC 99.89 (5th of ~48 entries when merged, 2026-10-12) |
| Text | Ettin-150M v3 (RAID + MAGE), used on gpttrace.com | MAGE unseen GPT-4 AUROC 0.979; MAGE paraphrased AUROC 0.742 (a known weak spot); in-house cross-validated AUC 0.959 |
| Image | Community Forensics ViT-S + our fine-tune, logits averaged | In-house cross-validated AUC 0.98; weaker on FLUX dev and GPT Image |
| Voice | WavLM-base-plus, MatMul-only int8 | EER 5.8% on In-the-Wild, 4.2% on ASVspoof 2021 DF |

Full methodology: <https://gpttrace.com/methodology/> and
<https://gpttrace.com/ai-detector-benchmark/>.

## Layout

```
js/engine/      detection engines (pure ES modules, no DOM): text, image, audio, video, C2PA reader, scoring
js/ui/          the tool UIs used on gpttrace.com
vendor/         ONNX Runtime Web 1.30, exifr 7.1.3, @huggingface/tokenizers 0.2.0
extension/      browser extension (Chrome, Edge, Firefox; Safari via apps/safari)
apps/           Safari extension project, mobile and desktop app shells
eval/           labelled evaluation, benchmark samplers, weight fitting
eval/train/     training scripts (Kaggle/Modal) for the text, image and audio models
sw.js, manifest.webmanifest   installable PWA with share target
```

## Commands

```sh
npm run build:extension                  # extension/dist (Chrome/Edge) and extension/dist-firefox
node eval/fetch.js text|image            # labelled data → ~/.cache/gpttrace/eval
node eval/run.js text|image|audio [--deep] [--bench=<name>]
node eval/fit-weights.js text --write    # refit scoring weights with 5-fold CV
```

Requires Node 22+ and Google Chrome (the evaluation runs the real engines in
headless Chrome, so the numbers match what the browser does).

## Reproducing the RAID 1B entry

The run used one Colab A100-40GB and took 5.4 h. Weights and logs are at
[nakas/gpttrace-v5-1b](https://huggingface.co/nakas/gpttrace-v5-1b).

```sh
# 1. Build the training kernel: RAID + 200k MAGE + 100k extra human text,
#    1 epoch, lr 2e-5, global batch 64.
GPU=1 DRY=1 JOBS="jhu-clsp/ettin-encoder-1b:v5-1b:2e-5" \
  sh eval/train/text/push_tpu.sh gpttrace-colab-v5 0 jhu-clsp/ettin-encoder-1b 500000 2e-5 200000 100000 1

# 2. Tokenise once on a cheap CPU machine and publish the arrays.
#    Also mount the paraphrase/humanizer pairs (eval/train/text/gen/gen_pairs.py).
sh eval/train/queue/colab_run.sh v5prep eval/train/text/colab_train.py CPU \
  GT_STAGE=prep GT_PREP_REPO=<your-hf>/gpttrace-prep-v5 HF_TOKEN=@hf \
  UPLOAD=eval/train/text/build/gpttrace-colab-v5/kernel.py:/content/kernel.py \
  UPLOAD=eval/train/text/raid-data/raid_data.py:/content/raid_data.py \
  UPLOAD=pairs.parquet:/content/pairs.parquet

# 3. Train on the A100. The run checkpoints to HF and resumes after a VM reclaim.
sh eval/train/queue/colab_run.sh v5-1b eval/train/text/colab_train.py A100 \
  GT_STAGE=train GT_PREP_REPO=<your-hf>/gpttrace-prep-v5 GT_SKIP_EXPORT=1 HF_TOKEN=@hf \
  UPLOAD=eval/train/text/build/gpttrace-colab-v5/kernel.py:/content/kernel.py

# 4. Score the RAID test set (predictions.json for the leaderboard).
sh eval/train/queue/colab_run.sh raidpred eval/train/text/colab_raid_predict.py A100 \
  GT_MODEL_REPO=<your-hf>/gpttrace-v5-1b HF_TOKEN=@hf \
  UPLOAD=eval/train/text/build/gpttrace-colab-v5/kernel.py:/content/kernel.py \
  UPLOAD=eval/train/text/raid-data/raid_data.py:/content/raid_data.py
```

The same `eval/train/text/tpu_v4.py` trains on Kaggle TPU v5e-8 (SPMD) or on
GPUs (`GPU=1`, torchrun DDP). The text normalisation is in
`eval/train/text/normalize.py` and mirrors `js/engine/text/normalize.js`.

## Licence

Code: MIT. Each model's licence is in its Hugging Face model card; training
data licences are listed there too.
