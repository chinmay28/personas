#!/bin/bash
# run2 data pipeline (2026-10-16): oracle re-render of tiers 2/3, tier 9 expansions, enlarged tier 6, assembly. Resumable: rerun to continue.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
set -x
$PY scripts/build_dataset.py oracle23 || exit 1
$PY scripts/build_dataset.py tier9 --n 1000 || exit 1
$PY scripts/build_dataset.py tier6 || exit 1
$PY scripts/build_dataset.py tier9 --n 1000 || exit 1   # second pass picks up the new tier 6 questions in the pool (no-op if already 1000)
$PY scripts/assemble_dataset.py > data/tiers/assemble_run2.out 2>&1 || exit 1
tail -40 data/tiers/assemble_run2.out
echo RUN2_DATA_DONE
