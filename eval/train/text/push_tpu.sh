#!/bin/sh
# Builds and pushes the v4 training run (tpu_v4.py on top of train_raid.py).
#   eval/train/text/push_tpu.sh <slug> <smoke 0|1> <base model> <raid rows> <lr> <mage rows> <extra human rows> <epochs>
# e.g. push_tpu.sh gpttrace-text-v4-tpu 0 jhu-clsp/ettin-encoder-150m 500000 5e-5 200000 100000 1
# JOBS="jhu-clsp/ettin-encoder-150m:v4-150m,jhu-clsp/ettin-encoder-400m:v4-400m" trains several
# models in one session, one after another (outputs in /kaggle/working/<tag>/).
# PAIRS=1 also mounts the paraphrase/humanizer pairs for training: the private dataset
# andrewnakas/gpttrace-pairs (push_pairs_dataset.sh), or PAIRS=kernel for the
# gpttrace-text-pairs kernel's output.
# GPU=1 builds the same run for Kaggle's 2xT4 instead (GT_DEVICE=cuda), e.g.
#   GPU=1 push_tpu.sh gpttrace-text-v4-gpu 0 jhu-clsp/ettin-encoder-150m 500000 5e-5 200000 100000 1
set -e
cd "$(dirname "$0")"
SLUG=$1 SMOKE=$2 BASE=$3 ROWS=$4 LR=$5 MAGE=$6 HUMAN=$7 EPOCHS=$8
EXTRA_SOURCES=""; DATASETS=""
[ "$PAIRS" = kernel ] && EXTRA_SOURCES=', "andrewnakas/gpttrace-text-pairs"'
[ "$PAIRS" = 1 ] && DATASETS='"andrewnakas/gpttrace-pairs"'
if [ -n "$GPU" ]; then DEVICE=cuda; ACCEL='"enable_gpu": true,
  "enable_tpu": false,'; SHAPE='"machine_shape": "NvidiaTeslaT4"'
else DEVICE=tpu; ACCEL='"enable_gpu": false,
  "enable_tpu": true,'; SHAPE='"docker_image": "gcr.io/kaggle-private-byod/python-tpuvm@sha256:ed64e34ebee5f385ffe83e706d6d8e776ff61767171a3f3aadad41042d24ec9f",
  "machine_shape": "TpuV5E8"'; fi
D=build/$SLUG
mkdir -p "$D"
# train_raid.py's own entry point is dropped; tpu_v4.py redefines main/train/score.
{ cat normalize.py; echo; sed '/^if __name__ == "__main__":/,$d' train_raid.py; cat tpu_v4.py; } \
  | sed -e "s|__SMOKE__|$SMOKE|" -e "s|__BASE__|$BASE|" -e "s|__ROWS__|$ROWS|" -e "s|__LR__|$LR|" -e "s|__MAGE__|$MAGE|" -e "s|__HUMAN__|$HUMAN|" -e "s|__EPOCHS__|$EPOCHS|" -e "s|__DEVICE__|$DEVICE|" -e "s|__JOBS__|${JOBS:-}|" > "$D/kernel.py"
cat > "$D/kernel-metadata.json" <<JSON
{
  "id": "andrewnakas/$SLUG",
  "title": "$SLUG",
  "code_file": "kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  $ACCEL
  "enable_internet": true,
  "keywords": [],
  "dataset_sources": [$DATASETS],
  "kernel_sources": ["andrewnakas/gpttrace-raid-data"$EXTRA_SOURCES],
  "competition_sources": [],
  "model_sources": [],
  $SHAPE
}
JSON
python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$D/kernel.py"
[ -n "$DRY" ] || kaggle kernels push -p "$D"
