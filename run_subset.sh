#!/usr/bin/env bash
# Reproduce the When2Tool tool-necessity probe on Apple Silicon (MPS) or CPU.
#
# Reproduces the paper's claim (README: "Linear probes on hidden states achieve
# AUROC 0.89-0.96 for predicting tool necessity") on a stratified subset, without
# vLLM/CUDA.
#
# Usage:
#   ./run_subset.sh                        # qwen3-1.7b, mps, 1024 tokens, 3 tasks/cell
#   ./run_subset.sh qwen3-1.7b mps 256 3   # control arm (256-token budget)
#   ./run_subset.sh qwen3-1.7b cpu 1024 2  # CPU fallback
#
# Args: MODEL DEVICE MAX_NEW_TOKENS PER_CELL
#
# Outputs (probe_data/subset_<tokens>/):
#   *_hidden_no_reasoning.pt        hidden states at the last prompt token
#   *_labels_no_reasoning.json      task_meta + tool_necessary labels
#   *_no_tool_outputs.json          raw no-tool generations (audit these!)
#   probe_layers_no_reasoning.png   test AUROC by layer
#
# Note: hidden states are extracted from the prompt prefill and are independent of
# the generation budget. Changing MAX_NEW_TOKENS therefore changes ONLY the labels
# (tool_necessary comes from whether the no-tool rollout answered correctly), which
# is what makes the 256-vs-1024 comparison a clean single-variable experiment.

set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:-qwen3-1.7b}"
DEVICE="${2:-mps}"
TOKENS="${3:-1024}"
PER_CELL="${4:-3}"

TAG="subset_${TOKENS}"
TRAIN_JSON="./data/scaled_train.json"
TEST_JSON="./data/scaled_test.json"

# Keep all HuggingFace state inside the repo so nothing writes to ~/.cache.
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

mkdir -p outputs/smoke probe_data data

py() { python "$@"; }
if [[ -x .venv/bin/python ]]; then py() { .venv/bin/python "$@"; }; fi

echo "== 0. Build stratified subset (per-cell=${PER_CELL}) =="
if [[ ! -f "$TRAIN_JSON" ]]; then
  py tools/make_subset.py --split train --per-cell "$PER_CELL" --out "$TRAIN_JSON"
else
  echo "   reusing $TRAIN_JSON"
fi
if [[ ! -f "$TEST_JSON" ]]; then
  py tools/make_subset.py --split test --per-cell "$PER_CELL" --out "$TEST_JSON"
else
  echo "   reusing $TEST_JSON"
fi

echo
echo "== 1. Labels + hidden states (model=${MODEL}, device=${DEVICE}, tokens=${TOKENS}) =="
echo "   ~5 h for 270 tasks on M4 MPS at 256 tokens; ~11 h at 1024 tokens."
echo "   Use --phase extract to skip generation if labels already exist."
py src/extract_features.py \
  --model_path "$MODEL" \
  --data_path "$TRAIN_JSON" \
  --data_path_test "$TEST_JSON" \
  --output_dir "./probe_data/$TAG" \
  --phase all \
  --backend hf \
  --device "$DEVICE" \
  --max_new_tokens "$TOKENS" \
  --max_rounds 8 \
  2>&1 | tee "outputs/smoke/${TAG}.log"

echo
echo "== 2. Linear probe: per-layer scan + per-difficulty AUROC =="
py tools/report_tables.py --data_dir "./probe_data/$TAG" \
  2>&1 | tee "outputs/smoke/probe_${TAG}.log"

echo
echo "== 3. Label-validity audit (unfiltered vs parseable-only) =="
py tools/probe_parseable_only.py --data_dir "./probe_data/$TAG" \
  2>&1 | tee "outputs/smoke/audit_${TAG}.log"

echo
echo "Done. Layer curve: probe_data/$TAG/probe_layers_no_reasoning.png"
echo "To compare both budgets, run this script twice (TOKENS=256 and 1024), then:"
echo "  python tools/compare_arms.py --out_md arms.md --out_png compare_arms.png"
