#!/bin/sh
# Opens a RAID leaderboard PR from a finished Kaggle training run.
#   eval/train/text/submit_raid.sh <kaggle slug> <submission dir name>
# e.g. submit_raid.sh gpttrace-raid-ettin150-v2 gpttrace-ettin150
# Needs: the run's predictions.json (672,000 test ids) and metrics.json.
set -e
SLUG=$1 NAME=$2
W=${TMPDIR:-/tmp}/raid-submit-$NAME
rm -rf "$W" && mkdir -p "$W/out"
kaggle kernels output "andrewnakas/$SLUG" -p "$W/out" --file-pattern "(predictions|metrics)\.json$"
python3 - "$W/out" <<'EOF'
import json, sys
d = sys.argv[1]
p = json.load(open(f"{d}/predictions.json"))
ids = {r["id"] for r in p}
assert len(p) == 672000 and len(ids) == 672000, (len(p), len(ids))
assert all(isinstance(r["score"], float) and r["score"] == r["score"] for r in p), "bad score"
m = json.load(open(f"{d}/metrics.json"))
print("ok:", len(p), "predictions; held-out RAID-train TPR@5%FPR", round(m["ours"]["tpr_all"], 4))
EOF
gh repo fork liamdugan/raid --clone=false >/dev/null 2>&1 || true
git clone --depth 1 --filter=blob:none --sparse "https://github.com/andrewnakas/raid.git" "$W/raid"
cd "$W/raid"
git remote add upstream https://github.com/liamdugan/raid.git 2>/dev/null || true
git fetch --depth 1 upstream main && git checkout -B "submit-$NAME" upstream/main
git sparse-checkout set "leaderboard/submissions/$NAME"
mkdir -p "leaderboard/submissions/$NAME"
cp "$W/out/predictions.json" "leaderboard/submissions/$NAME/"
cat > "leaderboard/submissions/$NAME/metadata.json" <<JSON
{
  "date_released": "$(date +%Y-%m-%d)",
  "detector_name": "GPTTrace (Ettin-150M)",
  "website": "https://gpttrace.com",
  "huggingface_link": "https://huggingface.co/jhu-clsp/ettin-encoder-150m",
  "contact_info": "https://gpttrace.com"
}
JSON
git add "leaderboard/submissions/$NAME"
git commit -q -m "Add detector: GPTTrace (Ettin-150M)"
git push -q -u origin "submit-$NAME"
cat > "$W/body.md" <<MD
Adds \`leaderboard/submissions/$NAME/\` (predictions.json + metadata.json) for all 672,000 test ids.

**Detector.** [jhu-clsp/ettin-encoder-150m](https://huggingface.co/jhu-clsp/ettin-encoder-150m) (MIT) fine-tuned as a 2-class sequence classifier for one epoch on ~680k RAID-train rows: every human text plus a capped sample per (domain, model, decoding, repetition penalty, attack) cell, with paraphrased AI texts seen twice. Max 512 tokens, lr 5e-5, cosine schedule, 2×T4 fp16.

**Input.** Before tokenisation, invisible format characters (Unicode Cf, incl. zero-width spaces) are removed and Cyrillic/Greek lookalike letters are mapped to Latin in mostly-Latin text. No whitespace or case changes.

**Score.** softmax P(machine) from the fp16 model. The same model runs client-side (int8 ONNX) in GPTTrace's in-browser AI text detector.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
MD
gh pr create -R liamdugan/raid --head "andrewnakas:submit-$NAME" --title "Add detector: GPTTrace (Ettin-150M)" --body-file "$W/body.md"
