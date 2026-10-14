#!/bin/sh
# Builds and pushes the TPU v5e-8 training run (tpu_v4.py on top of train_raid.py).
#   eval/train/text/push_tpu.sh <slug> <smoke 0|1> <base model> <raid rows> <lr> <mage rows> <extra human rows> <epochs>
# e.g. push_tpu.sh gpttrace-text-v4-tpu 0 jhu-clsp/ettin-encoder-150m 500000 5e-5 200000 100000 1
set -e
cd "$(dirname "$0")"
SLUG=$1 SMOKE=$2 BASE=$3 ROWS=$4 LR=$5 MAGE=$6 HUMAN=$7 EPOCHS=$8
D=build/$SLUG
mkdir -p "$D"
# train_raid.py's own entry point is dropped; tpu_v4.py redefines main/train/score.
{ cat normalize.py; echo; sed '/^if __name__ == "__main__":/,$d' train_raid.py; cat tpu_v4.py; } \
  | sed -e "s|__SMOKE__|$SMOKE|" -e "s|__BASE__|$BASE|" -e "s|__ROWS__|$ROWS|" -e "s|__LR__|$LR|" -e "s|__MAGE__|$MAGE|" -e "s|__HUMAN__|$HUMAN|" -e "s|__EPOCHS__|$EPOCHS|" > "$D/kernel.py"
cat > "$D/kernel-metadata.json" <<JSON
{
  "id": "andrewnakas/$SLUG",
  "title": "$SLUG",
  "code_file": "kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": true,
  "enable_internet": true,
  "keywords": [],
  "dataset_sources": [],
  "kernel_sources": ["andrewnakas/gpttrace-raid-data"],
  "competition_sources": [],
  "model_sources": [],
  "docker_image": "gcr.io/kaggle-private-byod/python-tpuvm@sha256:ed64e34ebee5f385ffe83e706d6d8e776ff61767171a3f3aadad41042d24ec9f",
  "machine_shape": "TpuV5E8"
}
JSON
python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$D/kernel.py"
[ -n "$DRY" ] || kaggle kernels push -p "$D"
