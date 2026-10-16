#!/usr/bin/env bash
# Waits for the tier 3 build to finish, assembles an interim dataset, then runs the long tier 5 + tier 6 + assembly job. Detached via setsid.
cd "$(dirname "$0")/.."
rm -f data/tiers/LONG_RUN_DONE
while pgrep -f "build_dataset.py tier3" >/dev/null; do sleep 15; done
echo "chain done $(date)"
.venv/bin/python scripts/assemble_dataset.py > data/tiers/assemble_interim.out 2>&1
echo "interim assembly done $(date)"
scripts/run_phase6_long.sh > data/tiers/run_long.out 2>&1
echo "long run exit $? $(date)" | tee data/tiers/LONG_RUN_DONE
