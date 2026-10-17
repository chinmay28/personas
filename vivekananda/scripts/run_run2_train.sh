#!/bin/bash
# run2 (owner-approved 2026-10-16): QLoRA on the run2 dataset -> export as vivekananda:8b-run2 -> phase 8 eval + comparisons.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
set -x
for m in $(ollama ps | awk 'NR>1 {print $1}'); do ollama stop "$m"; done   # training needs the GPU free (14b is 9.3 GB)
sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader
$PY train/train_qlora.py --out train/out/run2 > train/run2.log 2>&1 || { echo TRAIN_FAILED; exit 1; }
cp train/out/run2/metrics.md train/run2_metrics.md
$PY train/export.py --run train/out/run2 --name vivekananda:8b-run2 --gguf-prefix vivekananda-8b-run2 > train/export_run2.log 2>&1 || { echo EXPORT_FAILED; exit 1; }
$PY scripts/unified.py "What is Maya?" --model vivekananda:8b-run2 | head -20
$PY scripts/eval_full.py run --tag unified8b_run2 --backend unified --model vivekananda:8b-run2 > eval/results/full_unified8b_run2.log 2>&1 || { echo EVAL_FAILED; exit 1; }
$PY scripts/eval_full.py compare prompted14b unified8b_run2 > /dev/null 2>&1
$PY scripts/eval_full.py compare unified8b unified8b_run2 > /dev/null 2>&1
ls eval/results/ | grep -i run2
echo RUN2_TRAIN_DONE
