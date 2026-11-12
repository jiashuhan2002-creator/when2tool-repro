#!/usr/bin/env bash
# 等当前 256-token 运行结束后，自动启动 1024-token 版本。
#
# 为什么串行：16 GB 统一内存同时跑两个 1.7B 模型会触发 swap 抖动，反而更慢。
#
# 为什么保留 256 那版：它是"生成预算不足"这一变量的对照臂 —— 同一批 270 条任务、
# 同一模型、同一探针，只改 max_new_tokens。
#
# 重要：必须用「受管后台任务」方式启动，不能用 nohup & ——后者的进程会随父 shell
# 退出被回收，看起来在跑实际已死（本次踩过，22:16 启动的 watcher 在 2 分钟内消失）。
set -u

cd "$(dirname "$0")" || exit 1
PREV_LOG=outputs/smoke/scaled.log
NEXT_LOG=outputs/smoke/scaled1024.log
MAX_WAIT=$((4 * 3600))
waited=0

echo "[watcher] $(date '+%F %T') 开始等待 256-token 运行结束"
while ! grep -q "Phase 2 done" "$PREV_LOG" 2>/dev/null; do
    sleep 60
    waited=$((waited + 60))
    if [ "$waited" -ge "$MAX_WAIT" ]; then
        echo "[watcher] $(date '+%F %T') 等待超时，放弃启动"
        exit 1
    fi
done

echo "[watcher] $(date '+%F %T') 前一轮已完成，启动 1024-token 运行"
HF_HOME="$PWD/.hf_cache" HF_HUB_OFFLINE=1 \
    .venv/bin/python -u src/extract_features.py \
    --model_path qwen3-1.7b \
    --data_path ./data/scaled_train.json \
    --data_path_test ./data/scaled_test.json \
    --output_dir ./probe_data/scaled1024 \
    --phase all \
    --backend hf \
    --device mps \
    --max_new_tokens 1024 \
    --max_rounds 8 \
    > "$NEXT_LOG" 2>&1

echo "[watcher] $(date '+%F %T') 1024-token 运行结束 EXIT=$?"
