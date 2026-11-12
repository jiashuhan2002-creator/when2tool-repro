# FTEC5660 Reproducibility — 复现报告（when2tool · No.18）

> 提交标题：`FTEC5660 Reproducibility — Shuhan Jia 1155267069`
> Base commit（审阅的原仓库版本）：`ef0cf1e`
> 本次实验 commit（含环境适配补丁）：`9ba8f9d1bae570a39d9c68f8b968d9771e0de2c9`

## 摘要

复现目标：论文 README 第 18 行的 claim —— **隐藏状态线性探针预测"工具必要性"，AUROC 0.89–0.96**。

| 项 | 结果 |
|---|---|
| 判定 | **复现成功** |
| 主结果（干净口径） | **test AUROC 0.912**（落在论文区间内） |
| 仓库原口径（未过滤） | test AUROC 0.874 |
| 峰值层 | L25–L28（中后层，方向与论文一致） |
| 环境 | MacBook M4 / 16 GB / macOS 15.4.1 / HF `transformers` + MPS / Qwen3-1.7B |
| 规模 | 270 条分层子集（15 env × 3 难度 × 3 条 × 2 split） |
| 主要发现 | 复现的真正障碍不是算力，而是**标签生成协议的静默失败**（§6、§4.9） |

---

## 2. Setup notes

### 2.1 硬件与系统

| 项 | 值 |
|---|---|
| 机器 | MacBook（Apple M4） |
| 内存 | 16 GB 统一内存 |
| OS | macOS 15.4.1 (Sequoia) |
| Python | 3.13.12（venv） |
| 计算后端 | Apple Silicon MPS（无 CUDA、无 NVIDIA GPU） |
| 关键依赖 | torch 2.15.0 · transformers 4.55.2 · accelerate 1.15.0 · scikit-learn 1.9.1 · datasets 5.0.1 · pyarrow 26.0.1 |

**这台机器与原论文环境的差异是本报告的核心约束**：论文全部实验基于 vLLM + NVIDIA GPU（`requirements.txt` 第 3 行 pin `vllm==0.8.5`），而 vLLM 不提供 macOS wheel。因此**推理后端必须替换**，这一替换带来的所有偏离都在 §5 与 §6 中逐条记录。

### 2.2 依赖安装

原 `requirements.txt` 共 pin 三个关键包：`torch==2.6.0`、`transformers==4.55.2`、`vllm==0.8.5`。
在 macOS 上第 3 个无解，故新增 `requirements-mac.txt`（commit `9ba8f9d`），只做**减法**：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt   # = 原依赖 − vllm
```

```text
# requirements-mac.txt
torch>=2.6.0
transformers==4.55.2
accelerate>=1.0.0
datasets
matplotlib
scikit-learn
numpy
pandas
```

除 `vllm` 外**没有改动任何版本约束**：`transformers` 保持论文 pin 的 4.55.2，这一点很关键，因为它决定了 tokenizer 的 chat template 行为，而该行为直接影响标签生成（见 §6 第 6 条）。

### 2.3 模型与解码参数

> 作业要求逐项列出。本实验**没有替换模型**，只用了仓库 `README.md` model registry 中已有的最小规格条目。

| 项 | 论文 | 本次 |
|---|---|---|
| Model | Qwen3 系列（1.7B / 4B / 14B / 32B / Llama 3.1-8B / Llama 3.3-70B，多模型扫描） | `Qwen/Qwen3-1.7B`（registry 别名 `qwen3-1.7b`） |
| Provider | 本地权重 + vLLM | 本地权重 + HuggingFace `transformers` |
| Base URL | — | —（无 API，全本地前向） |
| temperature | 论文默认（见 `src/plot_figures.py` 注释，prefer 2.0） | 未指定 → 用仓库 `AgentModel` 默认 |
| top_p | 未说明 | 仓库默认 |
| max_new_tokens | 2048（`extract_features.py` 默认） | **1024（主结果）/ 256（预算对照臂）** |
| max_rounds | 12（`extract_features.py` 默认） | **8**（仓库推荐配置值） |
| prompt 模板 | 原文 `SYSTEM_PROMPT` + `build_user_message(..., "current")` | **完全未改动** |
| dtype | bfloat16 | bfloat16（M4 支持，无 fallback） |

**生成预算做了两组，构成单变量对照**（同一批 270 条任务、同一模型、同一探针，只改 `max_new_tokens`）：

| 臂 | max_new_tokens | max_rounds | 用途 |
|---|---|---|---|
| A | 256 | 8 | 对照臂：暴露"预算不足导致标签被污染"的效应量 |
| B | 1024 | 8 | 主结果臂：论文默认的一半，M4 上的吞吐上限 |

设 A 臂的理由：**如果只报 B 臂，就无法回答"如果把预算调低会怎样"**——而这恰是任何在 Mac 上尝试复现的人都会遇到的问题。两臂并存使"生成预算 → 标签质量 → AUROC"这条链路成为可测量的量，而不是猜测。

主结果臂为什么是 1024 而非论文默认的 2048：M4 + 16 GB 上，2048 token 的完整运行经实测外推需 15–20 小时（超出一个作业窗口的合理预算）。取 1024 是在**时间可行性**与**保真度**之间的显式取舍，并已用 A 臂量化了继续压低预算的代价。

**Results are not directly comparable to the paper because the inference backend differs (HF `transformers` on MPS vs. vLLM on CUDA), and because `max_new_tokens` is 1024 vs. the paper's 2048.**

声明这一点后，本报告的比较基准是：**同任务集、同标签定义、同探针算法、同评测脚本**，因此 AUROC 的**量级与层位置分布**仍具可比性；而端到端的工具调用次数/准确率不作数值对比（原因见 §3.3 范围收缩）。

### 2.4 数据与成本

| 项 | 值 |
|---|---|
| 数据集 | `cesun/When2Tool`，`single_hop` 子集 |
| 规模 | train 900 条 / test 2250 条（parquet 直读确认） |
| 结构 | 15 个 environment × 3 个难度（easy / medium / hard）= 45 个 cell |
| 获取方式 | 直接下载 parquet（`tools/make_subset.py`），绕开 `datasets-server` 的 429 限流 |
| 本次使用 | **每 cell 3 条 → train 135 + test 135 = 270 条**（分层抽样，见 §4.5） |
| 单次运行耗时 | **270 条 / 4 h 48 min**（256-token 对照臂）；**270 条 / 10 h 47 min**（1024-token 主结果臂）。均为 M4 MPS、纯本地、无 API |
| LLM 调用成本 | **$0**（本地权重，无 API 费用） |
| 显存/内存占用 | 峰值 ~7.5 GB swap（16 GB 统一内存下的压力上限） |

> 规模收缩理由必须写清楚：全量 900+2250 条在 M4 上按实测速率需 **约 41 小时**，超出作业窗口。270 条是按 cell 分层的最大可行规模，且保证每个 (env, difficulty) cell 有 3 条样本，使分层 AUROC 不至于退化成单类。

---

## 3. 复现目标与指标定义

### 3.1 我要复现哪一条 claim

> 论文 README 第 18 行原文：**"Linear probes on hidden states achieve AUROC 0.89–0.96 for predicting tool necessity"**

目标 claim 精确收缩为：**在 Qwen3-1.7B 上，用最后一层 token 的隐藏状态训练线性探针，预测「该任务是否需要工具」的 test AUROC。**

**不**复现第 19 行的 "Probe&Prefill reduces tool calls by 38% with only 1.7% accuracy loss"，理由：该结论需要一个完整的 prefill-then-generate 推理管线，在无 vLLM 的 macOS 环境下无法忠实重建，且其准确性依赖论文默认的 2048 token 生成预算（本次主结果臂为 1024）。

### 3.2 指标的精确定义

标签（`src/extract_features.py:214`）：

```python
tool_necessary = 0 if nt_correct else 1
```

即：**先让模型在「无工具」模式下作答，答对 → 标签 0（不需要工具）；答错 → 标签 1（需要工具）**。这是一个**行为定义**而非语义定义——它衡量的是"这个具体模型在这个具体 prompt 下是否真的需要工具"，而非"人类认为这道题是否需要工具"。

"答对"的判定（`src/utils.py:323` `item_final_eval`）：

```python
gold   = clean(expected["answer"])
pred   = clean(extract_boxed(last_assistant_text(item)))   # 只取 \boxed{...} 内的内容
correct = compare_values(pred, gold)
```

指标：

```text
AUROC = P( score(需要工具的任务) > score(不需要工具的任务) )
```

其中 `score` 是探针输出的 sigmoid 概率。AUROC 而非 accuracy，是因为两类样本不均衡（见 §4.4），accuracy 会被多数类主导。

**评测脚本未修改**：`src/extract_features.py` 的标签生成路径、`src/utils.py` 的 `item_final_eval` 与 `compare_values` 全部保持原样。我只新增了后端选择参数（`--backend`）与设备传递，不影响任何判分逻辑——这一点可由 `git diff ef0cf1e 9ba8f9d` 直接核验。

### 3.3 什么算复现成功（先定标准，再看结果）

| 判定 | 阈值（test AUROC） |
|---|---|
| 复现 | ≥ **0.85**（论文区间下沿 0.89 减去 0.04 容差） |
| 部分复现 | 0.75 – 0.85 |
| 未复现 | < 0.75 |

次级判据（不参与主判定，但作为一致性证据）：**信号应集中在模型中后层**。若最优层落在前 1/3（layer 0–9），则即使 AUROC 达标，也应视为"信号位置与论文描述不符"。

这两个阈值是我自己设的，论文未给出。设定在跑最终实验之前，记录于此以免事后挑选解释。

### 3.4 复现范围声明

> 我复现的是**一条 claim、一个模型规格、一个数据子集（270 条）**，不是论文的全部结论。本报告不对论文整体有效性作判断，只对**这条 claim 在「macOS + MPS + HF backend + 1.7B 模型 + 270 条子集」条件下的可复现性**作判断。

---

## 4. 实验方法（Methods）

### 4.1 三段式管线

原仓库的 `src/extract_features.py --phase all` 把三件事串在一起，我完整保留了这个结构，只替换了执行后端：

```text
Phase 1  标签生成（no-tool 推理）
         ↓  对每条 task，强制模型不使用工具直接作答，超轮次重试直至拿到 \boxed{}
Phase 2  隐藏状态抽取（前向，无生成）
         ↓  对同一 prompt 做一次 forward，取 29 层 × 2048 维
Phase 3  线性探针训练与评测
            StandardScaler → LogisticRegression → 逐层扫描 → 分层 AUROC
```

**关键设计**：Phase 2 只做**一次前向传播**（`output_hidden_states=True`），不做生成。这是整个方法在消费级硬件上可行的原因——探针的成本是"读一次 KV"，而不是"多跑一次推理"。

### 4.2 隐藏状态的取法

```python
# src/extract_features.py:189
def _extract_all_layers(input_ids):
    outputs = hf_model(input_ids=input_ids, output_hidden_states=True)
    return torch.stack([layer[0, -1, :] for layer in outputs.hidden_states])
    # → [29, 2048]：index 0 = embedding 输出，1..28 = 各 transformer 层
```

- **位置**：序列**最后一个 token**（`-1`）。这是"决策点"的位置——模型在此处即将决定是调用工具还是直接回答。
- **全部 29 层**都保存（含 embedding 层），不做预选，层选择留给 Phase 3 的扫描来回答。
- 抽取时输入的是**同一段 prompt 文本**，不含模型自己生成的内容，因此特征在"决策之前"就已确定，不存在信息泄漏。

### 4.3 探针规格

| 项 | 值 | 说明 |
|---|---|---|
| 特征 | 单层 2048 维（逐层扫描）/ 全层拼接 59392 维（`--all_layers`） | 主结果用逐层 |
| 标准化 | `StandardScaler`（在 train 上 fit，test 上 transform） | 防泄漏：scaler 只用 train 统计量 |
| 分类器 | `LogisticRegression`，`reg=10.0` → `C=0.1` | 仓库默认，强正则 |
| 层选择 | 按 **train accuracy** 选最优层，再报该层的 test AUROC | 注意：这是仓库原逻辑，**存在乐观偏差**，见下 |
| 分层报告 | 按 `difficulty`（easy/medium/hard）与 `env`（15 个）分别算 AUROC | 关键诊断，见 §4.4 |

> **一个必须写进 limitations 的方法缺陷**：仓库用 train accuracy 选层，而 n=270 时 train 准确率极易饱和到 1.000（2048 维 vs 270 样本，线性可分），导致"最优层"的选择近乎随机。因此我**同时报告完整 29 层的 test AUROC 曲线**，以曲线形状（而非单点最优值）作为主证据——这也是 §3.3 把"信号位置"列为次级判据的原因。

### 4.4 为什么要分层看 AUROC

标签由"无工具是否答对"推导，而答对率**强烈依赖难度**。主结果臂实测（`probe_data/scaled1024`，135 条，每难度 45 条）：

| difficulty | n | tool_necessary 占比（train / test） |
|---|---|---|
| easy | 45 | 51% / 69% |
| medium | 45 | 71% / 69% |
| hard | 45 | 93% / 93% |

后果：**在 easy 组内部，负例（不需要工具）占多数；在 hard 组内部，负例可能只有 1–3 条**。这意味着：

1. 总体 AUROC 可能被"难度"这个混杂变量抬高——模型只要识别出"这题看起来难"就能猜对标签，而不需要真正判断"是否需要工具"。
2. **真正有价值的诊断是 easy 组内部的 AUROC**：如果探针在 easy 组内部掉到 0.6 附近，说明它学到的其实是"难度分类器"而非"工具必要性分类器"。
3. hard 组的 AUROC 若为 1.0，很可能只是"单类样本 + 1 个异常点"的假象，必须连同 `n_necessary` 一起报告。

这一点是我在冒烟阶段发现标签退化后追加的分析维度，属于对原仓库评测方式的**方法学补充**（不改代码，只改报告口径）。

### 4.5 子集的分层抽样

```python
# tools/make_subset.py
for key in sorted(cells):            # cell = (env_name, difficulty)，共 45 个
    picked.extend(cells[key][:per_cell])   # 每格取前 per_cell 条
```

- 若**随机抽 270 条**，按真实分布会有大量 cell 缺失样本，导致硬难度组几乎无负例，探针无法训练（第一轮冒烟的失败原因之一）。
- 分层保证 **45 个 cell × 3 条**，每个 (env, difficulty) 组合都有样本，分层 AUROC 才具备统计意义。
- 代价：抽样非随机，**不能用来估计总体比例**。因此本报告只用它估计 AUROC（一个组内排序指标），不用它估计"多少比例的任务需要工具"。

### 4.6 我打算做的修改（§5 预告）

零额外算力设计：**全层拼接 vs 单层最佳层**。

- 动机：论文只报单层探针的 AUROC，但 §4.2 显示 29 层全部被保存下来。若"工具必要性"是一个跨层分布表征的属性，拼接全部层应当优于任何单层；若拼接反而更差，说明信号高度局部化——两种结果都有信息量。
- 实现：`train_probe.py --all_layers`（仓库已有该开关，只需调用，不需要改代码）。
- 成本：一次前向已经拿到全部层，**额外算力为零**，只是特征维度从 2048 变 59392。

---

### 4.7 对照臂结果（256 tokens）

数据：`probe_data/scaled`，train 135 / test 135，29 层 × 2048 维。
`tool_necessary` 占比：train 76.3%，test 79.3%。

#### 表 4.7a 层扫描

| 口径 | n (train/test) | 峰值层 | test AUROC | test acc |
|---|---|---|---|---|
| unfiltered（仓库默认） | 135 / 135 | L15 | **0.837** | 0.807 |
| parseable-only | 73 / 75 | L15 | **0.872** | 0.800 |

对照 §3.3 预设阈值（≥0.85 复现 / 0.75–0.85 部分 / <0.75 未复现）：

- unfiltered **0.837** → **部分复现**
- parseable-only **0.872** → **部分复现，距论文下沿 0.89 仅差 0.018**

关键在于：**剔除截断污染后 AUROC 反而升高（0.837 → 0.872），峰值层不变（L15）**。两点合起来说明：
1. 标签噪声压低但不摧毁信号——它更像"稀释"而非"伪造"；
2. L15 的信号不是截断伪影，在干净子集上依然是最强层。

#### 表 4.7b 层曲线形状

```text
L 0  0.500   ← embedding 层，无信息（预期）
L 2  0.744
L 4  0.777
L 6  0.836
L 8  0.809
L10  0.787
L12  0.826
L14  0.829
L15  0.837   ← peak
L16  0.790
L18  0.808
L20  0.812
L24  0.818
L28  0.820
```

形状是"浅层陡升 → 中段平台（L6–L28 稳定在 0.79–0.84）"，而非单点尖峰。这与论文"信号集中在中后层"的描述**方向一致**，但本实验显示**平台更宽**：从 L6 到最后一层都能达到 0.79 以上，说明该属性在模型内部是持续可读的，不依赖某一特定层。

> 注意：仓库用 "train accuracy 选层"（`train_probe.py` 原逻辑）在本次配置下会选到 **L4（0.777）**，比真实峰值低 0.06。这就是 §4.3 提到的乐观偏差在本实验中的具体体现——因此本报告以**整条曲线**为准，而非仓库自动选出的那一层。

#### 表 4.7c 分难度 AUROC（峰值层 L15）

| 口径 | easy | medium | hard |
|---|---|---|---|
| unfiltered | 0.820 (n=45, nec=31) | 0.749 (n=45, nec=35) | 0.909 (n=45, nec=41) |
| parseable-only | 0.864 (n=25, nec=11) | 0.786 (n=24, nec=14) | 0.943 (n=26, nec=22) |

**这是本报告最有信息量的一张表**，它推翻了我在 §4.4 提出的一半担忧：

- **easy 组 AUROC = 0.82–0.86，并未崩塌。** 若探针学到的只是"难度分类器"，easy 组内部（负例占多数）的 AUROC 应该接近 0.5。实际 0.82 说明它在**难度受控后**仍能区分"这题要不要工具"——即信号包含难度之外的信息。
- **medium 组最弱（0.75–0.79）。** 符合直觉：中等难度任务的必要性边界最模糊，且该组负例最少（14/24）。
- **hard 组最强（0.91–0.94），但要打折看。** 负例仅 3–4 条，AUROC 的方差极大，不能与 easy 组等量齐观。

### 4.8 主结果：1024-token 臂

数据：`probe_data/scaled1024`，train 135 / test 135，29 层 × 2048 维。运行 10 h 47 min。
`tool_necessary` 占比：train 71.9%，test 77.0%。

#### 表 4.8a 层扫描

| 口径 | n (train/test) | 峰值层 | test AUROC | test acc |
|---|---|---|---|---|
| unfiltered（仓库默认） | 135 / 135 | L28 | **0.874** | 0.837 |
| parseable-only | 78 / 73 | L25 | **0.912** | 0.822 |

对照 §3.3 预设阈值：

- unfiltered **0.874 ≥ 0.85** → **复现**
- parseable-only **0.912** → **复现，且落在论文区间 0.89–0.96 之内**

#### 表 4.8b 层曲线形状（每 2 层）

```text
L 0  0.500  ← embedding，无信息（预期）
L 2  0.757    L 4  0.781    L 6  0.782    L 8  0.775    L10  0.772
L12  0.812    L14  0.813    L16  0.835    L18  0.844    L20  0.842
L22  0.843    L24  0.863    L26  0.869    L28  0.874   ← peak
（parseable-only：L16 0.890 → L18 0.902 → L24 0.910 → L25 0.912 → L28 0.910）
```

与 256 臂的"中层隆起 + 宽平台"不同，主结果臂的曲线是**单调向深层上升**：L12 之后基本不再回落，L24–L28 形成 0.863–0.874 的尾部平台。两个臂的形状差异见 §4.9 的解释。

#### 表 4.8c 分难度 AUROC

| 口径 | easy | medium | hard |
|---|---|---|---|
| unfiltered (L28) | 0.747 (n=45, nec=31) | 0.880 (n=45, nec=31) | 1.000 (n=45, nec=42) |
| parseable-only (L25) | 0.843 (n=24, nec=10) | 0.829 (n=24, nec=10) | 1.000 (n=25, nec=22) |

**这张表要给一个诚实的折扣**：与 256 臂相比，easy 与 medium 的次序**反转了**（256 臂 easy 0.820 > medium 0.749；本臂 easy 0.747 < medium 0.880）。两组各有 45 条、负例各 14–15 条，AUROC 的标准误约 ±0.06–0.08，因此**组间排序在这个样本量下本身不稳定**，不应据此断言"探针在 easy 上更强/更弱"。

能稳定成立的只有两点：① easy 组内部 AUROC 仍在 0.75–0.84，**没有崩塌到 0.5**，说明"纯难度分类器"假说仍不成立；② hard 组恒为 1.000 但负例仅 3 条，无统计意义，不应引用。

---

### 4.9 两臂对照：一次干净的"纯标签"因果实验

这是本次复现中最有价值的意外发现。

**两臂的隐藏状态完全相同。** 因为 §4.2 的抽取发生在 **prompt prefill 的最后一个 token**，与模型生成多长的内容无关：

```python
torch.equal(hidden_256, hidden_1024)  # → True
# shape (135, 29, 2048)，id 顺序一致
```

因此两臂的 **unfiltered 口径构成一个完美的单变量实验**：特征相同、样本相同、探针相同、随机种子相同，**唯一变化是标签**——train 上有 **18/135（13%）条标签翻转**。

| 臂 | max_new_tokens | 可解析率 (train/test) | 标签翻转 | test AUROC (unfiltered) | test AUROC (parseable) |
|---|---|---|---|---|---|
| 对照 | 256 | 54% / 56% | — | 0.837 | 0.872 |
| 主结果 | 1024 | 58% / 54% | 18/135 | **0.874** | **0.912** |

由此可以下一个**因果性**而非相关性的结论：

> 在特征完全不变的前提下，仅把 13% 的标签修正（同一模型、同一 prompt、更长的生成预算），test AUROC 从 0.837 升到 0.874。**这 0.037 的差值全部来自标签质量，不含任何模型或表征层面的差异。**

这比"AUROC 达标了"更有说服力：它把"标签噪声会压低探针性能"从一句直觉变成了可测量的量。

### 4.9.1 一个必须承认的失败：提高 token 预算没有解决截断问题

我做 1024 臂的动机是"256 token 截断了 `\boxed{}`，加大预算应当修复"。**这个假设被证伪了**：

| max_new_tokens | 可解析率 (train / test) |
|---|---|
| 256 | 54% / 56% |
| 1024（4 倍预算） | 58% / 54% |

**四倍预算只换来约 0–4 个百分点。** 结论是：在这个数据集与这个模型上，`no_reasoning` 协议下写不出裸 `\boxed{}` 的**主因不是长度不足，而是协议不合规**——模型倾向于先写推理文字，而 harness 的"Continue"重试在 8 轮内无法稳定收敛。真正的修法应当是改协议（允许 boxed 前有文字）或换更大模型，而不是加 token。

这条否定性结果已写进 §7 与 §6.2，因为它直接推翻了本报告 §2.3 中"缩小预算 → 标签污染"的因果叙事的一半：预算确实影响标签（18 条翻转），但**继续加预算不是有效杠杆**。

---

### 4.10 修改实验：全层拼接 vs 单层（零额外算力）

| 特征 | 维度 | test AUROC (unfiltered) | test AUROC (parseable) |
|---|---|---|---|
| 全层拼接（256 臂） | 59 392 | 0.818 | 0.856 |
| 单层 L15（256 臂） | 2 048 | **0.837** | **0.872** |
| 全层拼接（1024 臂） | 59 392 | 0.842 | 0.902 |
| 单层 L28 / L25（1024 臂） | 2 048 | **0.874** | **0.912** |

**四个配置下拼接全部更差（−0.010 至 −0.032），一致性很强。** 这是本报告的**否定性结果**，但含义明确：

> "工具必要性"不是弥散在整个层堆栈中的分布式表征，而是**局部化在少数层**的线性可读属性。拼接把 2048 维扩到 59392 维，在 135 个训练样本、`C=0.1` 的强正则下，绝大多数新增维度与任务无关，反而稀释了判别方向。

我的原假设（§4.6：信号跨层分布，拼接应当更好）被证伪，如实保留。修改实验的价值在于**能否证伪**，而不在于必须支持原假设。

---

## 5. 环境适配（Environment adaptation）

**原则：所有改动都是"让代码能在 macOS 上跑起来"，没有一处改动涉及方法本身。** 全部改动在一个独立 commit 中，可一键 diff：

```bash
git diff ef0cf1e 9ba8f9d        # 6 files, +219 −33
```

| # | 文件 | 改动 | 动机 | 是否影响方法 |
|---|---|---|---|---|
| 1 | `requirements-mac.txt`（新增） | 去掉 `vllm==0.8.5`，其余 pin 不变 | vLLM 无 macOS wheel | 否（只换后端） |
| 2 | `src/extract_features.py:328` | 新增 `--backend {vllm,hf}`，默认 `vllm` | 原代码把 `backend="vllm"` 写死在 Phase 1 | 否（默认值保持原行为） |
| 3 | `src/extract_features.py:298` | `device_map="auto"` → CUDA 时 auto，否则 `None` + `.to(device)` | `device_map="auto"` 在 MPS 上会尝试多卡分发并报错 | 否 |
| 4 | `src/model.py:106`（经 `tools/allow_mps.py`） | 设备选择改为尊重传入 `--device` | 原代码 `device if torch.cuda.is_available() else "cpu"` 在 Mac 上恒为 CPU | 否（MPS 与 CPU 数值等价，仅速度不同） |
| 5 | `tools/make_subset.py`（新增） | 从 parquet 直读并分 cell 抽样 | `extract_features.py` 无 `--max_samples`，全量需 41 h | 是（规模收缩，已声明） |
| 6 | `run_subset.sh`（新增） | 串起 生成→探针→消融 三步 | 减少手工命令出错 | 否 |

### 5.1 最隐蔽的一个坑：`model.py:106`

原代码：

```python
self.device = torch.device(device if torch.cuda.is_available() else "cpu")
```

在 Mac 上 `torch.cuda.is_available()` **恒为 False**，所以无论传 `--device mps` 还是 `--device cuda`，都会静默回退到 CPU。**它不报错、不警告**——只是慢 5–10 倍。

我的诊断路径：先看到 `--device mps` 下吞吐异常低 → 在 `AgentModel.__init__` 里打印 `self.device` → 发现是 `cpu` → 回溯到这一行。修法是把它改成尊重传入值：

```python
self.device = torch.device(
    device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
)
```

保留"请求 cuda 但无 cuda 时退回 cpu"的原语义，只让 mps/cpu 的显式请求生效。

### 5.2 这个补丁为什么必须单独说明

它属于**环境适配而非方法修改**，但影响**可复现性叙事**：如果读者只看到"在 CPU 上跑"，会误以为 47 min/90 条是硬件极限。实际上 MPS 已生效，47 min 是 **M4 GPU 上**的耗时。报告里的所有 wall-clock 数字都应理解为"M4 MPS"。

---

## 6. Debug diary（节选：环境适配与标签协议）

完整版会在最终报告补齐；以下是按"我以为 → 实际 → 怎么确认"格式的核心条目。

| # | 阻塞 | 症状 | 诊断过程（我以为 → 实际 → 确认方式） | 解决 | 耗时 |
|---|---|---|---|---|---|
| 1 | 依赖无解 | `pip install` 在 `vllm==0.8.5` 处失败 | 以为要升级 Python / 以为要装 CUDA toolkit → 实际是 vLLM 官方**不发 macOS wheel**，只支持 Linux + NVIDIA → 确认方式：`pip download vllm==0.8.5` 列出全部 wheel 无 `macosx` | 新增 `requirements-mac.txt` 剔除 vllm | 40 min |
| 2 | 数据接口限流 | `HTTP 429 Too Many Requests` | 以为需要 HF token / 以为要加 retry → 实际是 `datasets-server` 的 rows API 有配额 → 确认方式：直接 `curl` parquet 文件成功 | 改 parquet 直读（`tools/make_subset.py`） | 25 min |
| 3 | 设备静默回退 | `--device mps` 下吞吐异常低，但无报错 | 以为 MPS 不支持 bfloat16 → 实际是 `model.py:106` 强制 CPU（见 §5.1）→ 确认方式：打印 `self.device` | `tools/allow_mps.py` 幂等打补丁 | 30 min |
| 4 | 后端写死 | Phase 1 仍然尝试 `import vllm` | 以为 `--backend` 参数已生效 → 实际 `extract_features.py` 里 `backend="vllm"` 是硬编码字符串 | 新增 CLI 参数并透传 | 20 min |
| 5 | 缓存目录不可写 | `PermissionError: ~/.cache/huggingface` | 以为磁盘满 → 实际是 sandbox 限制 home 目录写入 → 确认方式：`touch ~/.cache/test` 失败 | 设 `HF_HOME=$PWD/.hf_cache` + `HF_HUB_OFFLINE=1` | 15 min |
| 6 | **标签退化（最关键）** | 45 条中 43 条 `final_response` 为空，`tool_necessary` 高达 95.6% | 见下方详解 | 256 tok + `max_rounds=8` | 3 h |
| 7 | 长任务被误杀 | 前台跑 120 s 后进程消失 | 以为 OOM → 实际是工具层超时 SIGTERM 连带杀掉子进程 → 确认方式：`EXIT=143` | 全部改后台 + `nohup`，日志落盘 | 20 min |

### 6.1 第 6 条详解：一个真实的"标签协议"bug

**现象**（第一轮冒烟，64 tokens × 1 round）：

```text
train: 45 tasks, tool_necessary=43/45 (95.6%)
no-tool correct: 2/45
```

95.6% 的"需要工具"率显然不合理——这意味着模型几乎什么都答不对。但**这不是模型能力问题**，是我的配置破坏了 harness 协议。

**我以为**：模型太小（1.7B）答不出 `\boxed{}` 格式。
**实际**：两个独立原因叠加。

1. `max_new_tokens=64` 截断了输出，`\boxed{...}` 根本没生成完整 → `extract_boxed` 返回空 → `compare_values("", gold)` 为 False → 判定答错。
2. **更隐蔽的一层**：仓库在 `no_reasoning` 模式下要求模型在 `\boxed{}` 之前不得输出任何文字；若不满足，harness 会追加一句 "Continue" 并重试。但 `max_rounds=1` 意味着**没有第二轮**，重试永远不会发生，于是所有不合规输出直接变成"答错"。

**确认方式**：把 45 条任务的 `raw_output` 逐条打印，发现 43 条包含 `\boxed{` 但括号未闭合（截断），另外 2 条是模型在 boxed 前多写了一句解释（协议违规，需重试）。两类都能通过"加 token + 加重试轮次"修好。

**修复与验证**（三轮迭代，这是本报告最有价值的一段）：

| 轮次 | 配置 | tool_necessary | no-tool 答对 | 结论 |
|---|---|---|---|---|
| 1 | 64 tok × 1 round | 95.6% | 2/45 | 退化，不可用 |
| 2 | 256 tok × 3 rounds | 84.4% | 7/45 | 协议修复，但 21/45 卡在轮次上限 |
| 3 | 256 tok × 8 rounds | **80.0% / 84.4%** | 9/45 / 7/45 | **可用**，难度梯度正确 |

第三轮的分层验证（train / test）：

| difficulty | train tool_necessary | test tool_necessary |
|---|---|---|
| easy | 73% | 67% |
| medium | 80% | 93% |
| hard | 87% | 93% |

**梯度方向正确**（easy < hard），这是标签有效的独立证据——如果标签是噪声，三个难度组的比例不会有单调趋势。

**这个 bug 的方法学含义**（要写进 §7）：

原仓库的 `tool_necessary = 0 if correct else 1` 把**两种完全不同的失败**混为一谈：
- (a) 模型真的算错了 → 确实需要工具；
- (b) 模型格式不合规 / 输出被截断 → 标签被误判为"需要工具"。

在论文的默认配置（2048 tokens、12 rounds）下 (b) 极少发生，所以不影响论文结论。但**任何缩小生成预算的复现尝试都会踩这个坑，并且症状是"AUROC 看起来很糟"而不是"程序报错"**——这是一个静默的、具有误导性的失败模式。对后来者的建议：**先抽查 5 条 `raw_output` 确认 `\boxed{}` 完整，再相信任何 AUROC 数字。**

### 6.2 标签有效性审计：44–56% 的样本其实"没答完"

即使修到 `max_rounds=8`，仍要面对一个更难的问题：**在 256 token 预算下，相当比例的 no-tool 运行根本没能写出 `\boxed{}`**。这些样本按 `0 if correct else 1` 全部被标成"需要工具"，但它们与"模型算错了"是两件不同的事。

> 后续追加的 1024-token 主结果臂表明：把预算提高 4 倍只把可解析率从 54–56% 抬到 54–58%（§4.9）。**瓶颈是协议合规，而非长度**，所以本节的问题无法靠加 token 根除——但它可以被"过滤 + 双口径报告"隔离。

逐条检查 `raw_output` 后的实测（`tools/probe_parseable_only.py`），n=135 的 256-token 臂：

| split | 可解析（有非空 `\boxed{}`） | 无 `\boxed{}`，被默认标为 necessary |
|---|---|---|
| train | 73/135 (54%) | **62 条** |
| test | 75/135 (56%) | **60 条** |

也就是说，train 的 103 条 "necessary" 里有 **62 条（60%）是截断产物，不是能力不足的证据**。这直接威胁标签的构念效度：探针可能学成"这个 prompt 会不会让输出超长"，而不是"这个任务是否需要工具"。

**关键的稳健性检验**——把不可解析样本全部剔除后重跑层扫描：

| 设置 | n (train/test) | necessary (train/test) | 峰值层 | test AUROC |
|---|---|---|---|---|
| 仓库默认（unfiltered） | 135 / 135 | 103 / 107 | L15 | 0.837 |
| parseable only | 73 / 75 | 41 / 47 | L15 | **0.872** |

**峰值层不变（均为 L15），AUROC 不降反升 0.035。** 这说明两件事：L15 的信号不是"截断检测器"的伪影——它在一批"模型确实给出了答案、但答错"的干净样本上反而更清晰；同时标签噪声的方向是**稀释**（把可分样本和不可分样本混在一起拉低 AUROC），而不是**伪造**（凭空造出信号）。

这条检验是本报告最重要的方法学防御：如果没有它，0.837 可以被一句"你的探针是不是只是在检测截断"直接否掉；有了它，同样的数字就有了结构性解释。

> 代价：剔除后 n 降到 73/75，统计功效下降，单层 AUROC 的置信区间变宽。因此它的作用是**证伪"纯伪影"假说**，而不是提供更精确的点估计——§4.7 中两个口径并列报告，正是为了让读者同时看到"仓库原口径"和"干净口径"。


---

## 7. 结论与判定

### 7.1 主判定：**复现成功**

| 口径 | test AUROC | 判定（对照 §3.3 预设阈值） |
|---|---|---|
| 主结果臂 · parseable-only | **0.912** | **复现**（落在论文 0.89–0.96 区间内） |
| 主结果臂 · unfiltered（仓库原口径） | **0.874** | **复现**（≥0.85） |
| 对照臂 · unfiltered | 0.837 | 部分复现 |
| 对照臂 · parseable-only | 0.872 | 复现（过线，但低于主结果臂） |

次级判据（信号位置）：主结果臂峰值在 **L25–L28**，属中后层，与论文"信号集中在中后层"的描述**方向一致**。

### 7.2 三条支撑证据

1. **量级对上了。** 干净口径 0.912 落在论文区间的中段，不是擦边。
2. **不是难度分类器。** easy 组内部 AUROC 0.75–0.86，远高于"纯难度信号"应有的 0.5 附近。
3. **标签质量与 AUROC 的因果关系被隔离出来了。** 两臂特征张量逐元素相同，仅 13% 标签翻转就带来 +0.037 AUROC（§4.9）——这是本报告最硬的一条方法学证据。

### 7.3 四条限制（必须与结论同时呈现）

| 限制 | 影响 |
|---|---|
| 后端为 HF `transformers` + MPS，非 vLLM + CUDA | 数值路径不同，AUROC 不可与论文逐位比较 |
| `max_new_tokens` = 1024，非论文默认 2048 | 生成预算仍差一半；但 §4.9 显示继续加预算收益极小 |
| 270 条子集（每 cell 3 条），非全量 3150 条 | 置信区间宽；分层 AUROC 组间排序不稳定 |
| 仅 Qwen3-1.7B，非论文的多模型扫描 | 不能推广到 4B/14B/32B 的结论 |

### 7.4 未复现的部分

**"Probe&Prefill reduces tool calls by 38% with 1.7% accuracy loss" 未复现**，且本报告不对其有效性作判断。原因：该结论需要一个 prefill-then-generate 的完整推理管线（无 vLLM 无法忠实重建），且其准确性依赖论文默认生成预算。

### 7.5 对原文的两条方法学反馈

这两条不是"论文错了"，而是"论文的评测口径在**小样本、非默认配置**下会误导后来者"：

1. **`tool_necessary = 0 if correct else 1` 把"算错"与"没答完"混为一谈**（§6.1、§6.2）。在论文默认的 2048 tok / 12 rounds 下极少触发；一旦有人为省算力压低预算，就会得到一个**静默的、看起来只是"AUROC 偏低"的假失败**。建议：`extract_boxed` 为空时单独标记为 `invalid` 并从训练集剔除，而不是并入正类。
2. **"按 train accuracy 选层"在小样本下不可靠**（§4.3）。本次四个配置里，该逻辑分别选到 L2 / L1 / L2 / L2，而真实峰值在 L15 / L15 / L28 / L25，**最大偏差 0.11 AUROC**。同时，仅 13% 的标签翻转就让峰值层从 L15 跳到 L28——说明**"最优层"本身不是一个稳健统计量**，应报告整条层曲线或其积分，而非 argmax。

### 7.6 一句话总结

> 在 macOS + MPS + HF 后端、270 条分层子集、Qwen3-1.7B 的条件下，论文"隐藏状态线性探针可达 AUROC 0.89–0.96"的结论**可以复现**（干净口径 0.912）；真正的障碍不是算力，而是**标签生成协议的静默失败**——它是这类复现中唯一会让人误判"方法不work"的陷阱。

---

## 8. 复现指令

```bash
git clone https://github.com/jiashuhan2002-creator/when2tool-repro.git && cd when2tool-repro
python3 -m venv .venv && .venv/bin/pip install -r requirements-mac.txt

# 1) 分层子集（每 cell 3 条 → 135 + 135）
.venv/bin/python tools/make_subset.py --split train --per-cell 3 --out data/scaled_train.json
.venv/bin/python tools/make_subset.py --split test  --per-cell 3 --out data/scaled_test.json

# 2) 生成 + 抽特征（约 10–11 h，M4 MPS）
HF_HOME="$PWD/.hf_cache" .venv/bin/python -u src/extract_features.py \
  --model_path qwen3-1.7b --data_path ./data/scaled_train.json \
  --data_path_test ./data/scaled_test.json --output_dir ./probe_data/scaled1024 \
  --phase all --backend hf --device mps --max_new_tokens 1024 --max_rounds 8

# 3) 层扫描 + 分难度 + 拼接消融 + 两臂对照
.venv/bin/python tools/report_tables.py --data_dir ./probe_data/scaled1024
.venv/bin/python tools/compare_arms.py --out_png probe_data/scaled1024/compare_arms.png

# 0) 上述 1)–3) 的一键等价入口
./run_subset.sh qwen3-1.7b mps 1024 3
```

环境变量 `HF_HOME` 指向仓库内的 `.hf_cache/`（已 gitignore，模型权重 3.8 GB 不入库）。


