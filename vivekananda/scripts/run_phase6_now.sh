#!/usr/bin/env bash
cd "$(dirname "$0")/.."
rm -f data/tiers/LONG_RUN_DONE
echo "start $(date)"
.venv/bin/python scripts/assemble_dataset.py > data/tiers/assemble_interim.out 2>&1
echo "interim assembly done $(date)"
scripts/run_phase6_long.sh > data/tiers/run_long.out 2>&1
echo "long run exit $? $(date)" | tee data/tiers/LONG_RUN_DONE
