#!/bin/sh
# Runs one job on a Colab GPU through the Colab CLI and brings the outputs
# home; the session is always stopped afterwards. (Successor to
# colab_job.sh, whose blocking exec hung when Colab reclaimed a VM.)
#   sh eval/train/queue/colab_run.sh <name> <script.py> [T4|L4|A100|CPU] [KEY=VALUE…] [UPLOAD=local:remote…]
#   HF_TOKEN=@hf passes this Mac's Hugging Face token (read at run time).
#   COLAB_OUT_ROOT=<dir> puts the results somewhere other than ~/.cache/gpttrace/colab
# The script runs on the VM as its own detached process (libraries such as
# vLLM need a real stdout) and must write what it wants kept to /content/out/;
# that is packed and unpacked into $COLAB_OUT_ROOT/<name>/. KEY=VALUE args
# become env vars on the VM; UPLOAD=local:remote copies a file there first.
# Every 5 minutes the Mac pulls /content/out/job.log, finishes when
# /content/out/DONE appears, and fails fast if the VM is gone.
# Auth: gcloud application-default credentials (`colab --auth adc`).
set -e
NAME=$1 SCRIPT=$2 GPU=${3:-T4}
shift 2; [ $# -gt 0 ] && shift
OUT="${COLAB_OUT_ROOT:-$HOME/.cache/gpttrace/colab}/$NAME"
mkdir -p "$OUT"
C="colab --auth adc"
UPLOADS=""; ENVPY=""
for kv in "$@"; do
  case "$kv" in
    UPLOAD=*) UPLOADS="$UPLOADS ${kv#UPLOAD=}";;
    HF_TOKEN=@hf) ENVPY="$ENVPY\"HF_TOKEN\": \"$(cat "$HOME/.cache/huggingface/token")\", ";;   # read at run time, never stored in queue files
    *) ENVPY="$ENVPY\"${kv%%=*}\": \"${kv#*=}\", ";;
  esac
done
if [ "$GPU" = CPU ]; then $C new -s "$NAME" < /dev/null; else $C new --gpu "$GPU" -s "$NAME" < /dev/null; fi
trap '$C stop -s "$NAME" < /dev/null || true' EXIT
up() { for i in 1 2 3 4; do $C upload -s "$NAME" "$1" "$2" < /dev/null && return 0; echo "upload retry $i: $1"; sleep 20; done; return 1; }
up "$SCRIPT" /content/job.py
for u in $UPLOADS; do up "${u%%:*}" "${u#*:}"; done
cat > "$OUT/.launch.py" <<PY
import os, subprocess, sys
os.makedirs("/content/out", exist_ok=True)
env = {**os.environ, $ENVPY}
cmd = sys.executable + " -u /content/job.py > /content/out/job.log 2>&1; echo \$? > /content/out/DONE"
if os.path.exists("/content/out/.started"):
    print("already running")
else:
    open("/content/out/.started", "w").close()
    subprocess.Popen(["bash", "-c", cmd], env=env, start_new_session=True)
    print("started")
PY
# The launcher is idempotent (it does nothing if the job already started), so
# a launch whose reply was lost can be retried safely.
started=""
for i in 1 2 3 4 5; do
  if $C exec -s "$NAME" -f "$OUT/.launch.py" --timeout 120 < /dev/null 2>&1 | grep -q -E "^(started|already running)"; then started=1; break; fi
  echo "launch retry $i"; sleep 20
done
rm -f "$OUT/.launch.py"          # may hold secrets passed as KEY=VALUE (e.g. HF_TOKEN)
[ -n "$started" ] || { echo "colab job $NAME FAILED: could not start the job"; exit 1; }
# Wait by running a blocking check on the VM in ~20-minute chunks, so the
# notebook kernel stays busy (a detached job with an idle kernel got the A100
# reclaimed twice on 2026-10-14/15). A failed check (expired token, flaky
# network) is not proof the VM is gone: only three consecutive session
# listings that succeed but omit this session count; otherwise the EXIT trap
# would stop a VM that is still training.
cat > "$OUT/.wait.py" <<'PY'
import os, time
t = time.time()
while time.time() - t < 240 and not os.path.exists("/content/out/DONE"):
    time.sleep(20)
log = open("/content/out/job.log", errors="replace").read() if os.path.exists("/content/out/job.log") else ""
print(log.replace("\r", "\n").strip().splitlines()[-1][:200] if log.strip() else "(no log yet)")
print("DONE" if os.path.exists("/content/out/DONE") else "RUNNING")
PY
gone=0; n=0
while true; do
  n=$((n+1)); [ $n -gt 240 ] && { echo "colab job $NAME FAILED: gave up waiting after 240 checks"; exit 1; }
  # Short blocking checks keep the kernel busy; long ones drop on this link.
  res=$($C exec -s "$NAME" -f "$OUT/.wait.py" --timeout 400 < /dev/null 2>&1) && echo "$res" | grep -q -E "^(DONE|RUNNING)" && { gone=0; echo "$res" | grep -v -E "^(DONE|RUNNING)$" | tail -n 1; }
  # Whatever the check said, the DONE file decides.
  if $C download -s "$NAME" /content/out/DONE "$OUT/DONE" < /dev/null > /dev/null 2>&1; then
    $C download -s "$NAME" /content/out/job.log "$OUT/job.log" < /dev/null > /dev/null 2>&1 || true
    break
  fi
  echo "$res" | grep -q -E "^(DONE|RUNNING)" && continue
  sleep 30
  list=$($C sessions < /dev/null 2>&1) || { echo "check failed (transient), retrying"; continue; }
  if echo "$list" | grep -q "credentials\|Traceback\|rror"; then echo "check failed (transient), retrying"; continue; fi
  if echo "$list" | grep -q "\[$NAME\]"; then gone=0; else gone=$((gone+1)); fi
  if [ $gone -ge 3 ]; then
    echo "colab job $NAME FAILED: the VM is gone (reclaimed or timed out); partial log in $OUT/job.log"; exit 1
  fi
done
echo "job exit code $(cat "$OUT/DONE")"
printf 'import os,tarfile\nwith tarfile.open("/content/out.tar.gz","w:gz") as t: t.add("/content/out",arcname="out")\nprint("packed", sum(len(f) for _,_,f in os.walk("/content/out")), "files")\n' > "$OUT/.pack.py"
$C exec -s "$NAME" -f "$OUT/.pack.py" --timeout 3600 < /dev/null
$C download -s "$NAME" /content/out.tar.gz "$OUT/out.tar.gz" < /dev/null
tar -xzf "$OUT/out.tar.gz" -C "$OUT" && rm "$OUT/out.tar.gz"
echo "colab job $NAME done → $OUT"
[ "$(cat "$OUT/DONE")" = "0" ]
