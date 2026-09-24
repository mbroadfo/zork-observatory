#!/usr/bin/env bash
# One arm at a time, because there is one GPU.
#
# Every arm changes exactly one thing from the arm above it, so a difference
# between two rows has one candidate explanation. The traces land in
# traces/sweep/; tools/summarize_sweep.py reads them back as scores and turn
# quality, tools/structure.py as the shape of the map each arm drew.
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
OLLAMA="--agent ollama --model $MODEL --num-ctx $CTX"
# Thinking off everywhere except the arm that exists to price it.
OFF="$OLLAMA --think off"

# name|arguments
#
# The scaffold arms this list used to carry are gone with the code that served
# them: --recall, --candidates, --agenda, --vocabulary, --nudge and --journal
# were all measured in September 2026 and all removed. The traces are still in
# traces/sweep/ and the arms that produced them are named in
# docs/experiments/2026-09-scaffold-sweep.md; the tag scaffolds-2026-09 has the
# flags themselves, should anyone want to rerun one against a different model.
#
# What is left is the ladder that survived: the information levels, the
# transcript window, and the two dials that moved anything.
ARMS=(
  "floor|--agent random"
  "cold|$OFF --info-level cold"
  "game|$OFF --info-level game"
  "bare|$OFF --info-level parser"
  "coached|$OFF --info-level coached"
  "window10|$OFF --info-level parser --history-turns 10"
  "window60|$OFF --info-level parser --history-turns 60"
  "temp03|$OFF --info-level parser --temperature 0.3"
  "think|$OLLAMA --info-level parser --think on"
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
