#!/bin/bash
# Run a long GPU job without cooking a laptop.
#
#   scripts/run_guarded.sh <anything you'd pass to `uv run`>
#   scripts/run_guarded.sh train Mjlab-BallKick-Flat-MicroDuck --env.scene.num-envs 1024
#   scripts/run_guarded.sh scripts/train_fastsac.py --max-minutes 235
#
# The job runs in its own process group. Every 5 s the GPU temperature is checked:
#   >= PAUSE_AT  -> SIGSTOP the job        <= RESUME_AT -> SIGCONT it
#   >= STOP_AT   -> SIGINT (the training scripts save + export on SIGINT), then SIGKILL
# Early stop from another shell: `touch $OUT_DIR/STOP_REQUEST`.
# Output goes to $OUT_DIR (default: guarded_run/): train.log, watchdog.log, gpu.csv.

PAUSE_AT=${PAUSE_AT:-83}
RESUME_AT=${RESUME_AT:-72}
STOP_AT=${STOP_AT:-88}
OUT_DIR=${OUT_DIR:-guarded_run}

cd "$(dirname "$0")/.." || exit 1
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/STOP_REQUEST"

setsid nice -n 10 uv run "$@" > "$OUT_DIR/train.log" 2>&1 &
PGID=$!
paused=0

graceful_stop() {
  echo "$(date +%T) stopping" >> "$OUT_DIR/watchdog.log"
  kill -CONT -"$PGID"; kill -INT -"$PGID"
  for _ in $(seq 1 60); do kill -0 "$PGID" 2>/dev/null || break; sleep 2; done
  kill -KILL -"$PGID" 2>/dev/null
}
trap 'graceful_stop; echo "finished $(date +%T)" >> "$OUT_DIR/watchdog.log"; exit 0' TERM INT

echo "started pgid=$PGID $(date +%T)" > "$OUT_DIR/watchdog.log"
while kill -0 "$PGID" 2>/dev/null; do
  [ -f "$OUT_DIR/STOP_REQUEST" ] && { graceful_stop; break; }
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
  echo "$(date +%T),$t,$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits)" >> "$OUT_DIR/gpu.csv"
  if [ "$t" -ge "$STOP_AT" ]; then
    echo "$(date +%T) hard stop at ${t}C" >> "$OUT_DIR/watchdog.log"; graceful_stop; break
  elif [ "$t" -ge "$PAUSE_AT" ] && [ $paused -eq 0 ]; then
    kill -STOP -"$PGID"; paused=1; echo "$(date +%T) pause at ${t}C" >> "$OUT_DIR/watchdog.log"
  elif [ "$t" -le "$RESUME_AT" ] && [ $paused -eq 1 ]; then
    kill -CONT -"$PGID"; paused=0; echo "$(date +%T) resume at ${t}C" >> "$OUT_DIR/watchdog.log"
  fi
  sleep 5
done
echo "finished $(date +%T)" >> "$OUT_DIR/watchdog.log"
