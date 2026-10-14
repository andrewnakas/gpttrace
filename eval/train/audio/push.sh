#!/bin/sh
# Builds and pushes a Kaggle run of audio_train.py.
#   eval/train/audio/push.sh <slug> <smoke 0|1> <base model> <epochs> [extra 0|1]
set -e
cd "$(dirname "$0")"
SLUG=$1
D=build/$SLUG
mkdir -p "$D"
sed -e "s|os.environ.get(\"GT_SMOKE\", \"0\")|os.environ.get(\"GT_SMOKE\", \"$2\")|" -e "s|os.environ.get(\"GT_BASE\", \"facebook/wav2vec2-base\")|os.environ.get(\"GT_BASE\", \"$3\")|" -e "s|os.environ.get(\"GT_EPOCHS\", \"4\")|os.environ.get(\"GT_EPOCHS\", \"$4\")|" -e "s|os.environ.get(\"GT_EXTRA\", \"0\")|os.environ.get(\"GT_EXTRA\", \"${5:-0}\")|" audio_train.py > "$D/kernel.py"
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
  "dataset_sources": ["awsaf49/asvpoof-2019-dataset", "mohammedabdeldayem/avsspoof-2021", "bhaveshkumars/release-in-the-wild", "walimuhammadahmad/fakeaudio", "awsaf49/ljspeech-sr16k-dataset"],
  "kernel_sources": [],
  "competition_sources": [],
  "model_sources": [],
  "machine_shape": "NvidiaTeslaT4"
}
JSON
python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$D/kernel.py"
kaggle kernels push -p "$D"
