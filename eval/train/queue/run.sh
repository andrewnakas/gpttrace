#!/bin/sh
# Kaggle job queue runner. Kaggle allows one queued-or-running batch TPU
# session per account (and a couple of GPU ones), so jobs wait here and are
# pushed one at a time per lane as the previous one finishes.
#   nohup sh eval/train/queue/run.sh > ~/.cache/gpttrace/queue.log 2>&1 &
# Queue files (one job per line, "<slug> <command…>", run from the repo root):
#   eval/train/queue/tpu.txt   eval/train/queue/gpu.txt
# Finished lines move to eval/train/queue/done.txt with their final status.
# A lane line may start with "after:YYYY-MM-DD" to hold it until that date
# (e.g. the weekly GPU quota reset).
cd "$(dirname "$0")/../../.."
Q=eval/train/queue
OWNER=andrewnakas
status() { kaggle kernels status "$OWNER/$1" 2>/dev/null | sed -n 's/.*KernelWorkerStatus\.\([A-Z_]*\).*/\1/p'; }
lane() { # lane file, state file
  f=$Q/$1.txt; st=$Q/.$1.current
  if [ -s "$st" ]; then
    cur=$(cat "$st"); s=$(status "$cur")
    case "$s" in
      QUEUED|RUNNING|"") return ;;        # busy (or a transient API/SSL error)
      *) echo "$(date '+%F %T') $1 $cur finished: $s" ; echo "$(date '+%F') $1 $cur $s" >> $Q/done.txt; : > "$st" ;;
    esac
  fi
  [ -s "$f" ] || return
  line=$(head -n 1 "$f")
  case "$line" in after:*) d=${line#after:}; d=${d%% *}; [ "$(date +%F)" \< "$d" ] && return; line=${line#after:$d };; esac
  slug=${line%% *}; cmd=${line#* }
  echo "$(date '+%F %T') $1 push $slug: $cmd"
  if sh -c "$cmd"; then echo "$slug" > "$st"; tail -n +2 "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  else echo "$(date '+%F %T') $1 push failed, retrying next round"; fi
}
while true; do
  lane tpu; lane gpu
  [ -s $Q/tpu.txt ] || [ -s $Q/.tpu.current ] || [ -s $Q/gpu.txt ] || [ -s $Q/.gpu.current ] || { echo "$(date '+%F %T') queue empty"; exit 0; }
  sleep 600
done
