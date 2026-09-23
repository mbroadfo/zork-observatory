#!/usr/bin/env bash
# One arm at a time, because there is one GPU.
#
# Every arm changes exactly one thing from the arm above it, so a difference
# between two rows has one candidate explanation. The traces land in
# traces/sweep/ and tools/summarize_sweep.py reads them back.
#
#   bash tools/sweep.sh            # the whole ladder
#   bash tools/sweep.sh bare coached   # named arms only
#
# Runs inside the observatory container: jericho is a C extension and the
# model server is the other service.
set -u

CONTAINER="${CONTAINER:-zork-observatory-observatory-1}"
TURNS="${TURNS:-350}"
LIVES="${LIVES:-3}"
SEED="${SEED:-12345}"
MODEL="${MODEL:-qwen3:14b}"
CTX="${CTX:-8192}"
OUT="traces/sweep"
LOG="$OUT/sweep.log"

COMMON="--engine jericho --rom roms/zork1.z5 --turns $TURNS --lives $LIVES --seed $SEED"
# Thinking off, deliberately. Left to its own default qwen3:14b spends 1.5k-7k
# characters of it per turn and a turn takes 30s instead of 3s; the same ladder
# with thinking on is its own sweep, not a row in this one.
OLLAMA="--agent ollama --model $MODEL --num-ctx $CTX --think off"

# name|arguments
ARMS=(
  "floor|--agent random"
  "bare|$OLLAMA --info-level parser"
  "episodic|$OLLAMA --info-level parser --recall episodic"
  "coached|$OLLAMA --info-level coached"
  "coached-episodic|$OLLAMA --info-level coached --recall episodic"
  "candidates|$OLLAMA --info-level coached --recall episodic --candidates"
  "nudge|$OLLAMA --info-level coached --recall episodic --nudge"
  "agenda|$OLLAMA --info-level coached --recall episodic --candidates --agenda --vocabulary"
  "cold-agenda|$OLLAMA --info-level cold --recall episodic --candidates --agenda --vocabulary"
  "temp03|$OLLAMA --info-level coached --recall episodic --candidates --agenda --vocabulary --temperature 0.3"
  "journal|$OLLAMA --info-level coached --recall episodic --candidates --agenda --vocabulary --temperature 0.3 --journal new --runs 2"
)

mkdir -p "$OUT"
wanted=("$@")

for arm in "${ARMS[@]}"; do
  name="${arm%%|*}"
  args="${arm#*|}"
  if [ ${#wanted[@]} -gt 0 ] && [[ ! " ${wanted[*]} " =~ " ${name} " ]]; then continue; fi
  trace="$OUT/$name.jsonl"
  if [ -s "$trace" ]; then
    echo "[$(date +%H:%M:%S)] $name already has a trace, skipping" | tee -a "$LOG"
    continue
  fi
  echo "[$(date +%H:%M:%S)] === $name === $args" | tee -a "$LOG"
  started=$(date +%s)
  docker exec "$CONTAINER" observatory play $COMMON $args --trace "$trace" \
    >> "$OUT/$name.out" 2>&1
  code=$?
  echo "[$(date +%H:%M:%S)] $name finished in $(( ($(date +%s) - started) / 60 ))m (exit $code)" | tee -a "$LOG"
done

echo "[$(date +%H:%M:%S)] sweep done" | tee -a "$LOG"
