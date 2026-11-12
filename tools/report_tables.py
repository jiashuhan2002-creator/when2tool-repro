#!/usr/bin/env python3
"""生成报告用的完整结果表：层扫描、分层 AUROC、单层 vs 全层拼接。

输出 markdown 片段，直接粘进报告。
用法：
    python tools/report_tables.py --data_dir ./probe_data/scaled
    python tools/report_tables.py --data_dir ./probe_data/scaled1024
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils import extract_boxed, last_assistant_text  # noqa: E402


def load(data_dir, split):
    hidden = torch.load(
        os.path.join(data_dir, f"{split}_hidden_no_reasoning.pt"),
        map_location="cpu", weights_only=True,
    ).numpy()
    label_data = json.load(open(os.path.join(data_dir, f"{split}_labels_no_reasoning.json")))
    outs = json.load(open(os.path.join(data_dir, f"{split}_no_tool_outputs.json")))
    items = outs if isinstance(outs, list) else list(outs.values())
    parseable = {it.get("id"): bool((extract_boxed(last_assistant_text(it)) or "").strip())
                 for it in items}
    meta = label_data["task_meta"]
    y = np.array([m["tool_necessary"] for m in meta])
    mask = np.array([parseable.get(m["id"], False) for m in meta])
    return hidden, y, meta, mask


def fit_eval(Xtr, ytr, Xte, yte, C=0.1):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000, C=C).fit(sc.transform(Xtr), ytr)
    p_tr = clf.predict_proba(sc.transform(Xtr))[:, 1]
    p_te = clf.predict_proba(sc.transform(Xte))[:, 1]
    tr = roc_auc_score(ytr, p_tr)
    te = roc_auc_score(yte, p_te) if len(set(yte)) > 1 else float("nan")
    acc = accuracy_score(yte, (p_te >= 0.5).astype(int))
    return tr, te, acc, p_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    Htr, ytr, mtr, ok_tr = load(args.data_dir, "train")
    Hte, yte, mte, ok_te = load(args.data_dir, "test")
    lines = []
    w = lines.append

    w(f"## 数据：`{args.data_dir}`\n")
    w(f"- train {len(ytr)} 条，test {len(yte)} 条；29 层 × 2048 维")
    w(f"- 可解析样本：train {ok_tr.sum()}/{len(ok_tr)}，test {ok_te.sum()}/{len(ok_te)}")
    w(f"- tool_necessary 占比：train {ytr.mean():.1%}，test {yte.mean():.1%}\n")

    # --- 1. 层扫描（两口径）---
    w("### 表 1：层扫描（峰值层与 test AUROC）\n")
    w("| 口径 | n (train/test) | peak layer | test AUROC | test acc |")
    w("|---|---|---|---|---|")
    sweeps = {}
    for name, mtr_, mte_ in [("unfiltered", np.ones_like(ok_tr, bool), np.ones_like(ok_te, bool)),
                             ("parseable-only", ok_tr, ok_te)]:
        rows = []
        for layer in range(Htr.shape[1]):
            tr, te, acc, _ = fit_eval(Htr[mtr_, layer, :], ytr[mtr_],
                                      Hte[mte_, layer, :], yte[mte_])
            rows.append((layer, tr, te, acc))
        sweeps[name] = rows
        best = max(rows, key=lambda r: r[2] if not np.isnan(r[2]) else -1)
        w(f"| {name} | {mtr_.sum()}/{mte_.sum()} | L{best[0]} | **{best[2]:.3f}** | {best[3]:.3f} |")
    w("")

    # --- 2. 分层（在各自峰值层）---
    w("### 表 2：分难度 AUROC（峰值层）\n")
    w("| 口径 | easy | medium | hard |")
    w("|---|---|---|---|")
    for name, mtr_, mte_ in [("unfiltered", np.ones_like(ok_tr, bool), np.ones_like(ok_te, bool)),
                             ("parseable-only", ok_tr, ok_te)]:
        best = max(sweeps[name], key=lambda r: r[2] if not np.isnan(r[2]) else -1)
        L = best[0]
        _, _, _, p_te = fit_eval(Htr[mtr_, L, :], ytr[mtr_], Hte[mte_, L, :], yte[mte_])
        meta_te = [m for m, keep in zip(mte, mte_) if keep]
        y_te = yte[mte_]
        cells = []
        for diff in ["easy", "medium", "hard"]:
            idx = [i for i, m in enumerate(meta_te) if m["difficulty"] == diff]
            if len(idx) < 2 or len(set(y_te[idx])) < 2:
                cells.append("n/a")
            else:
                cells.append(f"{roc_auc_score(y_te[idx], p_te[idx]):.3f} (n={len(idx)}, "
                             f"nec={int(y_te[idx].sum())})")
        w(f"| {name} (L{L}) | " + " | ".join(cells) + " |")
    w("")

    # --- 3. 单层 vs 全层拼接（修改实验）---
    w("### 表 3：修改实验 — 单层 vs 全层拼接\n")
    w("| 特征 | 维度 | train AUROC | test AUROC | test acc |")
    w("|---|---|---|---|---|")
    for name, mtr_, mte_ in [("unfiltered", np.ones_like(ok_tr, bool), np.ones_like(ok_te, bool)),
                             ("parseable-only", ok_tr, ok_te)]:
        Xtr = Htr[mtr_].reshape(mtr_.sum(), -1)
        Xte = Hte[mte_].reshape(mte_.sum(), -1)
        tr, te, acc, _ = fit_eval(Xtr, ytr[mtr_], Xte, yte[mte_], C=0.01)
        w(f"| all-layers ( {name} ) | {Xtr.shape[1]} | {tr:.3f} | **{te:.3f}** | {acc:.3f} |")
        best = max(sweeps[name], key=lambda r: r[2] if not np.isnan(r[2]) else -1)
        w(f"| single L{best[0]} ( {name} ) | 2048 | {best[1]:.3f} | **{best[2]:.3f}** | {best[3]:.3f} |")
    w("")

    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
