#!/usr/bin/env bash
# Phase 6 long GPU runs (owner-confirmed): tier 5 scholar (~1,480 questions) and tier 6 persona (440 questions), then assembly.
# Resumable: re-run the same script after an interruption. Log: data/tiers/build.log. Est. 3-4 h with qwen3:14b on the RTX 4080.
#   VK_PERSONA_MODEL=qwen3.8:27b scripts/run_phase6_long.sh   # higher-quality tier 6 answers, ~4 h more
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/build_dataset.py tier5 --gen-sections 400
.venv/bin/python scripts/build_dataset.py tier6
.venv/bin/python scripts/assemble_dataset.py
