#!/usr/bin/env bash
# Full pipeline for one dataset config. Usage: bash scripts/run_all.sh configs/<dataset>.json
# Set PYTHON to the interpreter of the environment (default: python on PATH).
set -euo pipefail
CFG=${1:?usage: bash scripts/run_all.sh configs/<dataset>.json}
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."
$PY scripts/00_download.py "$CFG"
$PY scripts/01_audit_metadata.py "$CFG"
$PY scripts/02_make_splits.py "$CFG"
$PY scripts/03_extract_frames.py "$CFG"
for S in random_clip random_episode held_out_operator held_out_robot; do
  $PY scripts/04_finetune_lora.py "$CFG" --split "$S"
  for M in zero_shot text_only finetuned; do
    $PY scripts/05_evaluate.py "$CFG" --split "$S" --mode "$M"
  done
done
# Controls on the random-episode split: shuffled-label fine-tune, and 4-bit weights.
$PY scripts/04_finetune_lora.py "$CFG" --split random_episode --shuffle-labels
$PY scripts/05_evaluate.py "$CFG" --split random_episode --mode shuffled
$PY scripts/05_evaluate.py "$CFG" --split random_episode --mode finetuned --quant nf4
$PY scripts/06_metrics.py "$CFG"
