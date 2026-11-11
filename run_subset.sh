#!/usr/bin/env bash
# 在 Mac 上跑 When2Tool 的缩减版复现：只复现 "linear probe 的 AUROC" 这一条。
#
# 用法：
#   chmod +x run_subset.sh
#   ./run_subset.sh            # 默认 qwen3-1.7b + MPS
#
# 产物：
#   probe_data/<tag>/train_hidden_no_reasoning.pt   隐藏状态（前向传播，不生成）
#   probe_data/<tag>/test_labels_no_reasoning.json  标签（tool_necessary）
#   outputs/<tag>/probe_all_layers.json             基线 AUROC
#   outputs/<tag>/probe_single_layer.json           修改后 AUROC

set -euo pipefail

MODEL="${1:-qwen3-1.7b}"
DEVICE="${2:-mps}"
TAG="${MODEL}-subset"
TRAIN_JSON="./data/tasks_v1_train_subset.json"
TEST_JSON="./data/tasks_v1_test_subset.json"

mkdir -p outputs/"$TAG" probe_data data

echo "== 0. 准备子集 =="
if [[ ! -f "$TRAIN_JSON" ]]; then
  python tools/make_subset.py --split train --per-cell 6 --out "$TRAIN_JSON"
fi
if [[ ! -f "$TEST_JSON" ]]; then
  python tools/make_subset.py --split test --per-cell 5 --out "$TEST_JSON"
fi

echo
echo "== 1. 提特征 + 造标签（阶段 all：先跑 no-tool 判定是否需要工具，再取隐藏状态）=="
echo "   预计耗时：240 条任务约 15-30 分钟（MPS）；CPU 上会慢 5-10 倍"
time python src/extract_features.py \
  --model_path "$MODEL" \
  --data_path "$TRAIN_JSON" \
  --data_path_test "$TEST_JSON" \
  --output_dir "./probe_data/$TAG" \
  --phase all \
  --backend hf \
  --device "$DEVICE" \
  --max_new_tokens 64 \
  --max_rounds 1 \
  2>&1 | tee "outputs/$TAG/extract.log"

echo
echo "== 2. 基线：全层拼接的线性探针（复现目标）=="
python src/train_probe.py \
  --data_dir "./probe_data/$TAG" \
  --mode no_reasoning \
  --reg 10000 \
  --all_layers \
  2>&1 | tee "outputs/$TAG/probe_all_layers.log"

echo
echo "== 3. 修改：只用单层（消融「全层拼接」这个设计选择）=="
python src/train_probe.py \
  --data_dir "./probe_data/$TAG" \
  --mode no_reasoning \
  --reg 10000 \
  --layer 14 \
  2>&1 | tee "outputs/$TAG/probe_single_layer.log"

echo
echo "== 完成。把两个 log 里的 test_auroc 填进报告第 4、5 节的表 =="
