#!/bin/sh
# Builds and pushes a Kaggle training run of train_raid.py.
#   eval/train/text/push.sh <slug> <smoke 0|1> <base model> <train rows> <lr> [mage rows]
# e.g. push.sh gpttrace-raid-ettin150 0 jhu-clsp/ettin-encoder-150m 400000 5e-5
# Results: kaggle kernels output andrewnakas/<slug> -p <dir>
set -e
cd "$(dirname "$0")"
SLUG=$1 SMOKE=$2 BASE=$3 ROWS=$4 LR=$5 MAGE=${6:-0}
D=build/$SLUG
mkdir -p "$D"
{ cat normalize.py; echo; cat train_raid.py; } | sed -e "s|__SMOKE__|$SMOKE|" -e "s|__BASE__|$BASE|" -e "s|__ROWS__|$ROWS|" -e "s|__LR__|$LR|" -e "s|__MAGE__|$MAGE|" > "$D/kernel.py"
cat > "$D/kernel-metadata.json" <<JSON
{
  "id": "andrewnakas/$SLUG",
  "title": "$SLUG",
  "code_file": "kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "keywords": [],
  "dataset_sources": [],
  "kernel_sources": ["andrewnakas/gpttrace-raid-data"],
  "competition_sources": [],
  "model_sources": [],
  "machine_shape": "NvidiaTeslaT4"
}
JSON
python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$D/kernel.py"
kaggle kernels push -p "$D"
