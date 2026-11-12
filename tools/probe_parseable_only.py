#!/usr/bin/env python3
"""标签有效性审计：把"没能产出 \\boxed{} 答案"的样本从探针训练/评测中剔除。

为什么需要：仓库的标签是 `tool_necessary = 0 if correct else 1`，而"答错"包含两种
完全不同的情况：
  (a) 模型给出了可解析的答案，但答错了  → 真的需要工具
  (b) 模型在给定 token 预算内没写出完整的 \\boxed{...}  → 标签被误判为"需要工具"

(b) 在论文默认配置（2048 tokens / 12 rounds）下很少，但任何压缩生成预算的复现都会
大量踩到。后果是探针可能学成"预测输出会不会被截断"，而不是"预测是否需要工具"。
症状是 AUROC 偏低，且不报错 —— 静默失败。

本脚本提供 unfiltered vs parseable-only 的对照，用来判断信号是否稳健。

用法：
    python tools/probe_parseable_only.py --data_dir ./probe_data/smoke2
    python tools/probe_parseable_only.py --data_dir ./probe_data/scaled --mode no_reasoning
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils import extract_boxed, last_assistant_text  # noqa: E402


def load(data_dir, split, mode):
    hidden = torch.load(
        os.path.join(data_dir, f"{split}_hidden_{mode}.pt"),
        map_location="cpu", weights_only=True,
    ).numpy()
    with open(os.path.join(data_dir, f"{split}_labels_{mode}.json")) as f:
        label_data = json.load(f)
    with open(os.path.join(data_dir, f"{split}_no_tool_outputs.json")) as f:
        outs = json.load(f)
    items = outs if isinstance(outs, list) else list(outs.values())

    # 判定"是否产出了可解析答案"：只看 no-tool 输出里有没有非空 \boxed{}
    parseable = {
        it.get("id"): bool((extract_boxed(last_assistant_text(it)) or "").strip())
        for it in items
    }

    meta = label_data["task_meta"]
    y = np.array([m["tool_necessary"] for m in meta])
    mask = np.array([parseable.get(m["id"], False) for m in meta])
    return hidden, y, meta, mask


def sweep(hidden_tr, y_tr, hidden_te, y_te):
    rows = []
    for layer in range(hidden_tr.shape[1]):
        xtr, xte = hidden_tr[:, layer, :], hidden_te[:, layer, :]
        scaler = StandardScaler().fit(xtr)
        clf = LogisticRegression(max_iter=3000, C=0.1).fit(scaler.transform(xtr), y_tr)
        tr = roc_auc_score(y_tr, clf.predict_proba(scaler.transform(xtr))[:, 1])
        te = (
            roc_auc_score(y_te, clf.predict_proba(scaler.transform(xte))[:, 1])
            if len(set(y_te)) > 1 else float("nan")
        )
        rows.append((layer, tr, te))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--mode", default="no_reasoning")
    args = ap.parse_args()

    htr, ytr, _, ok_tr = load(args.data_dir, "train", args.mode)
    hte, yte, _, ok_te = load(args.data_dir, "test", args.mode)

    print("=" * 64)
    print(f"data_dir = {args.data_dir}")
    print(f"可解析样本占比:  train {ok_tr.sum()}/{len(ok_tr)} "
          f"({100*ok_tr.mean():.0f}%)   test {ok_te.sum()}/{len(ok_te)} "
          f"({100*ok_te.mean():.0f}%)")
    print(f"被截断/不合规而默认标为 necessary 的: "
          f"train {(~ok_tr & (ytr == 1)).sum()}   test {(~ok_te & (yte == 1)).sum()}")
    print("-" * 64)
    print(f"{'设置':<22}{'n(train/test)':<16}{'necessary':<14}{'best L':<8}{'test AUROC'}")
    print("-" * 64)

    for name, mtr, mte in [
        ("unfiltered (仓库默认)", np.ones_like(ok_tr, bool), np.ones_like(ok_te, bool)),
        ("parseable only (本次)", ok_tr, ok_te),
    ]:
        ytr_f, yte_f = ytr[mtr], yte[mte]
        if len(set(ytr_f)) < 2 or len(set(yte_f)) < 2:
            print(f"{name:<22}{len(ytr_f)}/{len(yte_f):<12}单类样本，跳过")
            continue
        rows = sweep(htr[mtr], ytr_f, hte[mte], yte_f)
        best = max(rows, key=lambda r: (r[2] if not np.isnan(r[2]) else -1))
        print(f"{name:<22}{len(ytr_f)}/{len(yte_f):<12}"
              f"{int(ytr_f.sum())}/{int(yte_f.sum())}      "
              f"L{best[0]:<7}{best[2]:.3f}")

    print("-" * 64)
    print("解读：若 parseable-only 的峰值层与 unfiltered 一致、AUROC 相差 <0.05，")
    print("说明信号不是'截断检测器'的假象，标签问题不影响主结论。")


if __name__ == "__main__":
    main()
