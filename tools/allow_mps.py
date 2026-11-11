#!/usr/bin/env python3
"""让 HFAgentBackend 在 Apple Silicon 上真正使用 MPS，而不是被强制退回 CPU。

原始代码（src/model.py:106）：
    self.device = torch.device(device if torch.cuda.is_available() else "cpu")
在 Mac 上 torch.cuda.is_available() 恒为 False，所以无论传 --device 什么都会变 CPU。

本脚本只改这一行，改成尊重传入的 device；幂等，可重复运行。
这是环境适配，不是方法修改 —— 请在报告的 Debug diary 里记一条。
"""

import pathlib
import re
import sys

TARGET = pathlib.Path(__file__).resolve().parent.parent / "src" / "model.py"
OLD = 'self.device = torch.device(device if torch.cuda.is_available() else "cpu")'
NEW = ('self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) '
       'else "cpu")')


def main():
    src = TARGET.read_text(encoding="utf-8")
    if NEW in src:
        print("already patched, nothing to do")
        return 0
    if OLD not in src:
        print("ERROR: 没找到预期的那一行，可能仓库已更新。请手动检查 src/model.py 第 106 行附近。")
        return 1
    TARGET.write_text(src.replace(OLD, NEW), encoding="utf-8")
    print(f"patched {TARGET}")
    print("现在可以用 --device mps 运行。注意：MPS 下 bfloat16 需要 M2 及以上；")
    print("如果报 dtype 相关错误，把 torch_dtype=torch.bfloat16 改成 torch.float16。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
