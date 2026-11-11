#!/usr/bin/env python3
"""从 HuggingFace 拉取 When2Tool 的一个分层子集，转成仓库期望的 JSON 格式。

为什么需要它：src/extract_features.py 只接受 data_path，没有 --max_samples，
不缩小数据集就得在 Mac 上跑 2250 条任务（不可行）。

用法：
    python tools/make_subset.py --split test  --per-cell 5 --out data/tasks_v1_test_subset.json
    python tools/make_subset.py --split train --per-cell 8 --out data/tasks_v1_train_subset.json

"per-cell" = 每个 (env_name, difficulty) 格子取几条。15 env × 3 难度 = 45 格。
per-cell=5 → 约 225 条；per-cell=8 → 约 360 条。
分层抽样是为了保证 easy/medium/hard 都有，否则探针会因单一类别训不出来。

依赖：pyarrow（装 datasets 时自带）。直接下 parquet，比 datasets-server 的 rows API 稳，
因为那个接口会 429 限流。
"""

import argparse
import json
import os
import urllib.request
from collections import Counter

import pyarrow.parquet as pq

REPO = "https://huggingface.co/datasets/cesun/When2Tool/resolve/main"
CACHE = "data/_hf_cache"


def fetch(split):
    os.makedirs(CACHE, exist_ok=True)
    url = f"{REPO}/single_hop/{split}-00000-of-00001.parquet"
    path = os.path.join(CACHE, f"single_hop_{split}.parquet")
    if not os.path.exists(path):
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    print(f"reading {path} ({os.path.getsize(path)/1e6:.2f} MB)")
    return pq.read_table(path).to_pylist()


def to_task(row):
    """转成 src/utils.py 里 _load_from_hf 生成的同构结构。"""
    task = {
        "id": row["id"],
        "difficulty": row["difficulty"],
        "multi_step": row["multi_step"],
        "instruction": row["instruction"],
        "environments": [{
            "name": row["env_name"],
            "tools": json.loads(row["tools"]),
            "parameters": json.loads(row["parameters"]),
        }],
        "expected": {"answer": row["answer"]},
        "tags": json.loads(row["tags"]),
    }
    steps = json.loads(row["steps"])
    if steps:
        task["expected"]["steps"] = steps
    return task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--per-cell", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = fetch(args.split)

    cells = {}
    for row in rows:
        cells.setdefault((row["env_name"], row["difficulty"]), []).append(row)

    picked = []
    for key in sorted(cells):
        picked.extend(cells[key][: args.per_cell])

    tasks = [to_task(r) for r in picked]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"\nwrote {len(tasks)} tasks -> {args.out}")
    print(f"cells: {len(cells)} (env × difficulty)")
    print("by difficulty:", dict(Counter(t["difficulty"] for t in tasks)))
    print("by env:", dict(Counter(t["environments"][0]["name"] for t in tasks)))


if __name__ == "__main__":
    main()
