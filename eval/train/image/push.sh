#!/bin/sh
# Builds and pushes a Kaggle run of image_train.py.
#   eval/train/image/push.sh <slug> <smoke 0|1> <epochs> <lr>
set -e
cd "$(dirname "$0")"
SLUG=$1
D=build/$SLUG
mkdir -p "$D"
sed -e "s|os.environ.get(\"GT_SMOKE\", \"0\")|os.environ.get(\"GT_SMOKE\", \"$2\")|" -e "s|os.environ.get(\"GT_EPOCHS\", \"2\")|os.environ.get(\"GT_EPOCHS\", \"$3\")|" -e "s|os.environ.get(\"GT_LR\", \"1e-5\")|os.environ.get(\"GT_LR\", \"$4\")|" image_train.py > "$D/kernel.py"
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
  "kernel_sources": [],
  "competition_sources": [],
  "model_sources": [],
  "machine_shape": "NvidiaTeslaT4"
}
JSON
python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$D/kernel.py"
kaggle kernels push -p "$D"
