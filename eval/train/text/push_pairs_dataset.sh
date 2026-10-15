#!/bin/sh
# Publishes pairs.parquet (from the Colab or Kaggle pairs job) as a private
# Kaggle dataset, andrewnakas/gpttrace-pairs, for training jobs to mount
# (PAIRS=1 in push_tpu.sh). Re-running creates a new version.
#   sh eval/train/text/push_pairs_dataset.sh <path/to/pairs.parquet>
set -e
SRC=$1
D=build/pairs-dataset
mkdir -p "$(dirname "$0")/$D"; cd "$(dirname "$0")"
cp "$SRC" "$D/pairs.parquet"
cat > "$D/dataset-metadata.json" <<JSON
{ "title": "gpttrace-pairs", "id": "andrewnakas/gpttrace-pairs", "licenses": [{ "name": "other" }] }
JSON
if kaggle datasets status andrewnakas/gpttrace-pairs >/dev/null 2>&1; then
  kaggle datasets version -p "$D" -m "pairs $(date +%F)"
else
  kaggle datasets create -p "$D"
fi
