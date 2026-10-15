#!/bin/sh
# Runs one job on a free Colab GPU through the Colab CLI and brings the
# outputs home; the session is always stopped afterwards.
#   sh eval/train/queue/colab_job.sh <name> <script.py> [gpu, default T4] [KEY=VALUE…] [UPLOAD=local:remote…]
#   COLAB_OUT_ROOT=<dir> puts the results somewhere other than ~/.cache/gpttrace/colab
# The script runs on the VM as its own process and must write everything it wants kept to
# /content/out/; it is packed into out.tar.gz and unpacked into
# ~/.cache/gpttrace/colab/<name>/. Auth: gcloud application-default
# credentials (`colab --auth adc`), already set up on this Mac.
# Free tier: T4 not guaranteed, ~12 h per VM, so scripts should checkpoint.
set -e
NAME=$1 SCRIPT=$2 GPU=${3:-T4}
shift 2; [ $# -gt 0 ] && shift
OUT="${COLAB_OUT_ROOT:-$HOME/.cache/gpttrace/colab}/$NAME"
mkdir -p "$OUT"
C="colab --auth adc"
$C new --gpu "$GPU" -s "$NAME" < /dev/null
trap '$C stop -s "$NAME" < /dev/null || true' EXIT
# KEY=VALUE args become env vars on the VM, except UPLOAD=local:remote,
# which copies a local file to the VM first.
ENVS=""; UPLOADS=""
for kv in "$@"; do case "$kv" in UPLOAD=*) UPLOADS="$UPLOADS ${kv#UPLOAD=}";; *) ENVS="$ENVS --env $kv";; esac; done
# Run as a normal process, not inside the Jupyter kernel that `colab exec`
# uses (libraries such as vLLM need a real stdout file descriptor). Its
# output goes to /content/out/job.log, which comes home with the results.
$C upload -s "$NAME" "$SCRIPT" /content/job.py < /dev/null
for u in $UPLOADS; do $C upload -s "$NAME" "${u%%:*}" "${u#*:}" < /dev/null; done
printf 'import subprocess,sys,os\nos.makedirs("/content/out",exist_ok=True)\nwith open("/content/out/job.log","w") as log:\n    rc=subprocess.run([sys.executable,"-u","/content/job.py"],stdout=log,stderr=subprocess.STDOUT).returncode\nprint("job exit code",rc)\nprint(open("/content/out/job.log").read()[-4000:])\n' > "$OUT/.launch.py"
$C exec -s "$NAME" -f "$OUT/.launch.py" --timeout 43000 $ENVS < /dev/null 2>&1 | tee "$OUT/run.log"
printf 'import os,tarfile\nos.makedirs("/content/out",exist_ok=True)\nwith tarfile.open("/content/out.tar.gz","w:gz") as t: t.add("/content/out",arcname="out")\nprint("packed", sum(len(f) for _,_,f in os.walk("/content/out")), "files")\n' > "$OUT/.pack.py"
$C exec -s "$NAME" -f "$OUT/.pack.py" --timeout 600 < /dev/null
$C download -s "$NAME" /content/out.tar.gz "$OUT/out.tar.gz" < /dev/null
tar -xzf "$OUT/out.tar.gz" -C "$OUT" && rm "$OUT/out.tar.gz"
echo "colab job $NAME done → $OUT"
