# FTEC5660 Reproducibility — Reproduction Report (when2tool · Item No.18)

> Submission: `FTEC5660 Reproducibility — Shuhan Jia 1155267069`
> Base commit (upstream version reviewed): `ef0cf1e`
> Experiment commit (including environment patches): `9ba8f9d1bae570a39d9c68f8b968d9771e0de2c9`

## Abstract

Target: the claim on line 18 of the paper's README — **linear probes on hidden states predict tool necessity with AUROC 0.89–0.96**.

| Item | Result |
|---|---|
| Verdict | **Reproduced** |
| Main result (clean protocol) | **test AUROC 0.912** (inside the paper's range) |
| Repository protocol (unfiltered) | test AUROC 0.874 |
| Peak layer | L25–L28 (mid-to-late layers, consistent with the paper) |
| Environment | MacBook M4 / 16 GB / macOS 15.4.1 / HF `transformers` + MPS / Qwen3-1.7B |
| Scale | 270-task stratified subset (15 envs × 3 difficulties × 3 tasks × 2 splits) |
| Main finding | The real obstacle to reproduction is not compute but a **silent failure in the labelling protocol** (§6, §4.9) |

---

## 2. Setup notes

### 2.1 Hardware and OS

| Item | Value |
|---|---|
| Machine | MacBook (Apple M4) |
| Memory | 16 GB unified memory |
| OS | macOS 15.4.1 (Sequoia) |
| Python | 3.13.12 (venv) |
| Compute backend | Apple Silicon MPS (no CUDA, no NVIDIA GPU) |
| Key dependencies | torch 2.15.0 · transformers 4.55.2 · accelerate 1.15.0 · scikit-learn 1.9.1 · datasets 5.0.1 · pyarrow 26.0.1 |

**The gap between this machine and the paper's environment is the central constraint of this report.** Every experiment in the paper runs on vLLM + NVIDIA GPUs (`requirements.txt` line 3 pins `vllm==0.8.5`), and vLLM ships no macOS wheel. The **inference backend therefore had to be replaced**; every deviation this introduces is recorded item by item in §5 and §6.

### 2.2 Installing dependencies

The original `requirements.txt` pins three key packages: `torch==2.6.0`, `transformers==4.55.2`, `vllm==0.8.5`. The third is unsolvable on macOS, so `requirements-mac.txt` was added (commit `9ba8f9d`) as a pure **subtraction**:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt   # = original requirements − vllm
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

**No version constraint other than `vllm` was changed**: `transformers` stays at the paper's pinned 4.55.2. This matters, because that version fixes the tokenizer's chat-template behaviour, which in turn drives label generation (see §6, item 6).

### 2.3 Model and decoding parameters

> The assignment asks for each item to be listed. This experiment **did not swap in a different model**; it used the smallest entry already present in the repository's `README.md` model registry.

| Item | Paper | This work |
|---|---|---|
| Model | Qwen3 family (1.7B / 4B / 14B / 32B / Llama 3.1-8B / Llama 3.3-70B, multi-model sweep) | `Qwen/Qwen3-1.7B` (registry alias `qwen3-1.7b`) |
| Provider | Local weights + vLLM | Local weights + HuggingFace `transformers` |
| Base URL | — | — (no API; all inference is local) |
| temperature | Paper default (per a comment in `src/plot_figures.py`, prefers 2.0) | Not set → repository's `AgentModel` default |
| top_p | Not stated | Repository default |
| max_new_tokens | 2048 (`extract_features.py` default) | **1024 (main arm) / 256 (budget control arm)** |
| max_rounds | 12 (`extract_features.py` default) | **8** (the value recommended in the repository) |
| Prompt template | Original `SYSTEM_PROMPT` + `build_user_message(..., "current")` | **Unchanged** |
| dtype | bfloat16 | bfloat16 (supported on M4, no fallback) |

**Two generation budgets were run, forming a single-variable comparison** (same 270 tasks, same model, same probe; only `max_new_tokens` differs):

| Arm | max_new_tokens | max_rounds | Purpose |
|---|---|---|---|
| A | 256 | 8 | Control arm: measures the effect size of "insufficient budget corrupts labels" |
| B | 1024 | 8 | Main arm: half the paper's default, which is the throughput ceiling on an M4 |

Why arm A exists: **reporting only arm B would leave "what happens if you lower the budget?" unanswered** — and that is exactly the question anyone attempting this on a Mac will hit. Having both arms turns the chain "generation budget → label quality → AUROC" into something measurable rather than guessed.

Why the main arm uses 1024 rather than the paper's 2048: on an M4 with 16 GB, extrapolating the measured throughput puts a full 2048-token run at 15–20 hours, beyond a reasonable budget for one assignment. Choosing 1024 is an explicit trade between **time feasibility** and **fidelity**, and arm A quantifies the cost of pushing the budget down further.

**Results are not directly comparable to the paper because the inference backend differs (HF `transformers` on MPS vs. vLLM on CUDA), and because `max_new_tokens` is 1024 rather than the paper's 2048.**

Having stated that, the comparison basis for this report is: **same task set, same label definition, same probe algorithm, same evaluation script**. The **magnitude of AUROC and the distribution of signal across layers** therefore remain comparable; end-to-end tool-call counts and accuracy are not compared numerically (see the scope reduction in §3.3).

### 2.4 Data and cost

| Item | Value |
|---|---|
| Dataset | `cesun/When2Tool`, `single_hop` subset |
| Size | 900 train / 2250 test (confirmed by reading the parquet files directly) |
| Structure | 15 environments × 3 difficulties (easy / medium / hard) = 45 cells |
| How obtained | Direct parquet download (`tools/make_subset.py`), bypassing the `datasets-server` 429 rate limit |
| Used here | **3 tasks per cell → 135 train + 135 test = 270 tasks** (stratified sampling, see §4.5) |
| Wall-clock per run | **270 tasks / 4 h 48 min** (256-token control arm); **270 tasks / 10 h 47 min** (1024-token main arm). Both on M4 MPS, fully local, no API |
| LLM API cost | **$0** (local weights, no API fees) |
| Memory footprint | Peak ~7.5 GB swap (the pressure ceiling under 16 GB unified memory) |

> The rationale for shrinking the scale has to be stated plainly: at the measured throughput, the full 900 + 2250 tasks would take **about 41 hours** on an M4, beyond the assignment window. 270 tasks is the largest feasible size that keeps per-cell stratification, guaranteeing 3 samples in every (env, difficulty) cell so that per-difficulty AUROC does not degenerate into a single class.

---

## 3. Target claim and metric definition

### 3.1 Which claim is being reproduced

> Verbatim from the paper's README, line 18: **"Linear probes on hidden states achieve AUROC 0.89–0.96 for predicting tool necessity"**

The target is narrowed precisely to: **on Qwen3-1.7B, train a linear probe on the hidden state of the last token and report the test AUROC for predicting whether a task requires a tool.**

The line-19 claim — "Probe&Prefill reduces tool calls by 38% with only 1.7% accuracy loss" — is **not** reproduced. That result needs a complete prefill-then-generate inference pipeline, which cannot be faithfully rebuilt on macOS without vLLM, and its accuracy depends on the paper's default 2048-token budget (this work's main arm uses 1024).

### 3.2 Precise metric definition

Label (`src/extract_features.py:214`):

```python
tool_necessary = 0 if nt_correct else 1
```

That is: **let the model answer in "no tool" mode; correct → label 0 (tool not needed); incorrect → label 1 (tool needed)**. This is a **behavioural** definition, not a semantic one — it measures "does this particular model, under this particular prompt, actually need a tool", not "would a human say this task needs a tool".

How "correct" is decided (`src/utils.py:323`, `item_final_eval`):

```python
gold   = clean(expected["answer"])
pred   = clean(extract_boxed(last_assistant_text(item)))   # only the content inside \boxed{...}
correct = compare_values(pred, gold)
```

Metric:

```text
AUROC = P( score(tool-needed task) > score(tool-not-needed task) )
```

where `score` is the sigmoid output of the probe. AUROC rather than accuracy because the two classes are imbalanced (see §4.4) and accuracy would be dominated by the majority class.

**The evaluation script is unmodified**: the label-generation path in `src/extract_features.py` and `item_final_eval` / `compare_values` in `src/utils.py` are untouched. The only additions are a backend selector (`--backend`) and device propagation, neither of which touches scoring logic — verifiable directly with `git diff ef0cf1e 9ba8f9d`.

### 3.3 What counts as a successful reproduction (criteria fixed before results)

| Verdict | Threshold (test AUROC) |
|---|---|
| Reproduced | ≥ **0.85** (paper's lower bound 0.89 minus a 0.04 tolerance) |
| Partially reproduced | 0.75 – 0.85 |
| Not reproduced | < 0.75 |

Secondary criterion (not part of the verdict, used as consistency evidence): **signal should concentrate in the mid-to-late layers**. If the best layer fell in the first third (layers 0–9), the result should be treated as "signal location inconsistent with the paper" even if AUROC clears the bar.

Both thresholds are mine; the paper gives none. They were fixed before the final experiment ran and are recorded here to rule out post-hoc explanation.

### 3.4 Scope statement

> What is reproduced here is **one claim, one model size, one data subset (270 tasks)** — not the paper's full set of conclusions. This report makes no judgement about the paper's overall validity, only about the **reproducibility of this claim under "macOS + MPS + HF backend + 1.7B model + 270-task subset"**.

---

## 4. Methods

### 4.1 The three-stage pipeline

The original repository's `src/extract_features.py --phase all` chains three steps together. That structure is kept intact; only the execution backend changed:

```text
Phase 1  Label generation (no-tool inference)
         ↓  Force the model to answer each task without tools; retry across rounds until \boxed{} is produced
Phase 2  Hidden-state extraction (forward pass, no generation)
         ↓  One forward pass over the same prompt, capturing 29 layers × 2048 dims
Phase 3  Linear probe training and evaluation
            StandardScaler → LogisticRegression → per-layer scan → per-difficulty AUROC
```

**Key design point**: Phase 2 performs **a single forward pass** (`output_hidden_states=True`) and generates nothing. That is why the method is feasible on consumer hardware — the probe costs "one read of the KV cache", not "one extra inference run".

### 4.2 How hidden states are taken

```python
# src/extract_features.py:189
def _extract_all_layers(input_ids):
    outputs = hf_model(input_ids=input_ids, output_hidden_states=True)
    return torch.stack([layer[0, -1, :] for layer in outputs.hidden_states])
    # → [29, 2048]: index 0 = embedding output, 1..28 = the transformer layers
```

- **Position**: the **last token** of the sequence (`-1`). This is the "decision point" — the place where the model is about to decide between calling a tool and answering directly.
- **All 29 layers** are stored (including the embedding layer) with no pre-selection; choosing a layer is left to the sweep in Phase 3.
- The input is **the same prompt text**, excluding anything the model generated, so the features are fixed *before* the decision and no information leaks backwards.

### 4.3 Probe specification

| Item | Value | Notes |
|---|---|---|
| Features | 2048 dims per layer (per-layer sweep) / 59392 dims concatenated (`--all_layers`) | Main results use per-layer |
| Standardisation | `StandardScaler` (fit on train, transform test) | Leakage guard: scaler sees only train statistics |
| Classifier | `LogisticRegression`, `reg=10.0` → `C=0.1` | Repository default, strong regularisation |
| Layer selection | Pick the best layer by **train accuracy**, then report that layer's test AUROC | This is the repository's own logic and it is **optimistically biased** — see below |
| Stratified reporting | AUROC computed separately by `difficulty` (easy/medium/hard) and by `env` (15) | Key diagnostic, see §4.4 |

> **A methodological flaw that belongs in the limitations section**: the repository selects layers by train accuracy, and at n=270 train accuracy saturates at 1.000 almost immediately (2048 dims vs 270 samples is linearly separable), which makes "the best layer" close to a random pick. This report therefore **also reports the full 29-layer test-AUROC curve** and treats the *shape* of that curve rather than a single argmax as the main evidence — which is also why "signal location" is a secondary criterion in §3.3.

### 4.4 Why AUROC has to be read per difficulty

Labels derive from "did it answer correctly without tools", and correctness **depends strongly on difficulty**. Measured on the main arm (`probe_data/scaled1024`, 135 tasks, 45 per difficulty):

| difficulty | n | share labelled tool_necessary (train / test) |
|---|---|---|
| easy | 45 | 51% / 69% |
| medium | 45 | 71% / 69% |
| hard | 45 | 93% / 93% |

Consequence: **inside the easy group negatives (tool not needed) are the majority; inside the hard group there may be only 1–3 negatives.** That means:

1. Overall AUROC can be inflated by difficulty as a confounder — a model that merely recognises "this looks hard" can guess the label without ever judging tool necessity.
2. **The informative diagnostic is AUROC *within* the easy group**: if the probe drops to around 0.6 there, it has learned a difficulty classifier rather than a tool-necessity classifier.
3. An AUROC of 1.0 in the hard group is likely an artefact of "one class plus a single anomaly" and must always be reported alongside `n_necessary`.

This dimension was added after the label degradation surfaced during smoke testing. It is a **methodological supplement** to the repository's evaluation (no code change, only a change in what gets reported).

### 4.5 Stratified subset sampling

```python
# tools/make_subset.py
for key in sorted(cells):            # cell = (env_name, difficulty), 45 in total
    picked.extend(cells[key][:per_cell])   # take the first per_cell tasks from each cell
```

- Sampling **270 tasks at random** would leave many cells empty, giving the hard group almost no negatives and making the probe untrainable (one of the failure modes in the first smoke run).
- Stratification guarantees **45 cells × 3 tasks**, so every (env, difficulty) combination is represented and per-difficulty AUROC becomes statistically meaningful.
- Cost: the sample is not random and **cannot be used to estimate population proportions**. This report therefore uses it only to estimate AUROC (a within-group ranking statistic), never to estimate "what fraction of tasks need tools".

### 4.6 The modification I set out to make (preview of §5)

Zero additional compute: **all-layer concatenation vs. single best layer**.

- Motivation: the paper only reports single-layer probe AUROC, yet §4.2 shows all 29 layers are retained. If tool necessity is a property distributed across layers, concatenating all of them should beat any single layer; if concatenation is worse, the signal is highly localised — either outcome is informative.
- Implementation: `train_probe.py --all_layers` (the flag already exists; calling it requires no code change).
- Cost: the forward pass already produced every layer, so **additional compute is zero**; only the feature dimension changes, from 2048 to 59392.

---

### 4.7 Control-arm results (256 tokens)

Data: `probe_data/scaled`, 135 train / 135 test, 29 layers × 2048 dims.
Share labelled `tool_necessary`: train 76.3%, test 79.3%.

#### Table 4.7a Layer sweep

| Protocol | n (train/test) | Peak layer | test AUROC | test acc |
|---|---|---|---|---|
| unfiltered (repository default) | 135 / 135 | L15 | **0.837** | 0.807 |
| parseable-only | 73 / 75 | L15 | **0.872** | 0.800 |

Against the thresholds pre-registered in §3.3 (≥0.85 reproduced / 0.75–0.85 partial / <0.75 not reproduced):

- unfiltered **0.837** → **partially reproduced**
- parseable-only **0.872** → **partially reproduced, 0.018 short of the paper's lower bound of 0.89**

The key point: **removing truncation contamination *raises* AUROC (0.837 → 0.872) while the peak layer stays at L15.** Together these say two things:
1. Label noise depresses the signal but does not destroy it — it acts more like **dilution** than fabrication;
2. The L15 signal is not a truncation artefact: it remains the strongest layer on the clean subset.

#### Table 4.7b Shape of the layer curve

```text
L 0  0.500   ← embedding layer, no information (expected)
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

The shape is "a steep rise in the early layers followed by a mid-stack plateau (L6–L28 holding at 0.79–0.84)", not a single sharp peak. That is **directionally consistent** with the paper's "signal concentrated in the mid-to-late layers", but the plateau found here is **broader**: every layer from L6 onward clears 0.79, meaning the property stays linearly readable throughout the model rather than living in one specific layer.

> Note: the repository's "select by train accuracy" logic (`train_probe.py`'s original behaviour) picks **L4 (0.777)** under this configuration, 0.06 below the true peak. That is the optimistic bias of §4.3 showing up concretely — which is why this report relies on the **whole curve** rather than the layer the repository happens to select.

#### Table 4.7c AUROC by difficulty (peak layer L15)

| Protocol | easy | medium | hard |
|---|---|---|---|
| unfiltered | 0.820 (n=45, nec=31) | 0.749 (n=45, nec=35) | 0.909 (n=45, nec=41) |
| parseable-only | 0.864 (n=25, nec=11) | 0.786 (n=24, nec=14) | 0.943 (n=26, nec=22) |

**This is the most informative table in the report**, and it overturns half of the concern raised in §4.4:

- **AUROC within the easy group is 0.82–0.86 — no collapse.** If the probe had learned a difficulty classifier, AUROC inside the easy group (where negatives dominate) should sit near 0.5. The observed 0.82 shows it still separates "needs a tool" from "does not" **once difficulty is held constant** — i.e. the signal carries information beyond difficulty.
- **The medium group is weakest (0.75–0.79).** Consistent with intuition: medium tasks have the blurriest necessity boundary, and that group also has the fewest negatives (14/24).
- **The hard group is strongest (0.91–0.94), but discount it.** With only 3–4 negatives the variance of AUROC is enormous and it should not be weighted like the easy group.

### 4.8 Main result: the 1024-token arm

Data: `probe_data/scaled1024`, 135 train / 135 test, 29 layers × 2048 dims. Runtime 10 h 47 min.
Share labelled `tool_necessary`: train 71.9%, test 77.0%.

#### Table 4.8a Layer sweep

| Protocol | n (train/test) | Peak layer | test AUROC | test acc |
|---|---|---|---|---|
| unfiltered (repository default) | 135 / 135 | L28 | **0.874** | 0.837 |
| parseable-only | 78 / 73 | L25 | **0.912** | 0.822 |

Against the §3.3 thresholds:

- unfiltered **0.874 ≥ 0.85** → **reproduced**
- parseable-only **0.912** → **reproduced, and inside the paper's 0.89–0.96 range**

#### Table 4.8b Shape of the layer curve (every second layer)

```text
L 0  0.500  ← embedding, no information (expected)
L 2  0.757    L 4  0.781    L 6  0.782    L 8  0.775    L10  0.772
L12  0.812    L14  0.813    L16  0.835    L18  0.844    L20  0.842
L22  0.843    L24  0.863    L26  0.869    L28  0.874   ← peak
(parseable-only: L16 0.890 → L18 0.902 → L24 0.910 → L25 0.912 → L28 0.910)
```

Unlike the 256 arm's "mid-stack hump plus wide plateau", the main arm's curve rises **monotonically into the deeper layers**: after L12 it essentially never falls back, and L24–L28 form a tail plateau of 0.863–0.874. §4.9 explains the difference in shape between the two arms.

#### Table 4.8c AUROC by difficulty

| Protocol | easy | medium | hard |
|---|---|---|---|
| unfiltered (L28) | 0.747 (n=45, nec=31) | 0.880 (n=45, nec=31) | 1.000 (n=45, nec=42) |
| parseable-only (L25) | 0.843 (n=24, nec=10) | 0.829 (n=24, nec=10) | 1.000 (n=25, nec=22) |

**This table needs an honest discount**: compared with the 256 arm, the ordering of easy and medium **reverses** (256 arm: easy 0.820 > medium 0.749; this arm: easy 0.747 < medium 0.880). Each group has 45 tasks and 14–15 negatives, so the standard error of AUROC is roughly ±0.06–0.08; **the between-group ordering is itself unstable at this sample size**, and no claim should be made about the probe being stronger or weaker on easy tasks.

Only two things hold securely: (i) AUROC inside the easy group stays at 0.75–0.84, **not collapsed toward 0.5**, so the "pure difficulty classifier" hypothesis still fails; (ii) the hard group reads 1.000 throughout but with only 3 negatives, which is statistically meaningless and should not be cited.

---

### 4.9 Comparing the two arms: a clean causal experiment on labels alone

This is the most valuable incidental finding of the reproduction.

**The hidden states of the two arms are exactly identical.** That follows from §4.2: extraction happens at **the last token of the prompt prefill**, independent of how much the model goes on to generate:

```python
torch.equal(hidden_256, hidden_1024)  # → True
# shape (135, 29, 2048), identical id ordering
```

So under the **unfiltered protocol the two arms form a perfect single-variable experiment**: identical features, identical samples, identical probe, identical seed, **the only thing that changes is the label** — with **18/135 (13%) label flips** on train.

| Arm | max_new_tokens | Parseable rate (train/test) | Label flips | test AUROC (unfiltered) | test AUROC (parseable) |
|---|---|---|---|---|---|
| Control | 256 | 54% / 56% | — | 0.837 | 0.872 |
| Main | 1024 | 58% / 54% | 18/135 | **0.874** | **0.912** |

Which licenses a **causal** rather than correlational statement:

> With features held completely fixed, correcting just 13% of the labels (same model, same prompt, longer generation budget) moves test AUROC from 0.837 to 0.874. **That gap of 0.037 comes entirely from label quality and contains no model- or representation-level difference.**

This is more persuasive than "the AUROC cleared the bar": it turns "label noise depresses probe performance" from an intuition into a measured quantity.

### 4.9.1 A failure that has to be admitted: a larger token budget did not fix truncation

The motive for running the 1024 arm was "256 tokens truncates `\boxed{}`, so a bigger budget should fix it". **The hypothesis was falsified**:

| max_new_tokens | Parseable rate (train / test) |
|---|---|
| 256 | 54% / 56% |
| 1024 (4× budget) | 58% / 54% |

**Four times the budget bought roughly 0–4 percentage points.** The conclusion: on this dataset with this model, the main reason no bare `\boxed{}` appears under the `no_reasoning` protocol is **not insufficient length but protocol non-compliance** — the model prefers to write reasoning prose first, and the harness's "Continue" retry does not reliably converge within 8 rounds. The real fix is to change the protocol (allow text before the boxed answer) or use a larger model, not to add tokens.

This negative result is carried into §7 and §6.2 because it undercuts half of the causal story in §2.3 ("smaller budget → corrupted labels"): the budget does affect labels (18 flips), but **pushing it higher is not an effective lever**.

---

### 4.10 The modification: all-layer concatenation vs. a single layer (zero extra compute)

| Features | Dims | test AUROC (unfiltered) | test AUROC (parseable) |
|---|---|---|---|
| All layers concatenated (256 arm) | 59 392 | 0.818 | 0.856 |
| Single layer L15 (256 arm) | 2 048 | **0.837** | **0.872** |
| All layers concatenated (1024 arm) | 59 392 | 0.842 | 0.902 |
| Single layer L28 / L25 (1024 arm) | 2 048 | **0.874** | **0.912** |

**Concatenation is worse in all four configurations (−0.010 to −0.032), with strong consistency.** This is a **negative result**, but it has a clear meaning:

> Tool necessity is not a representation smeared across the whole layer stack; it is a linearly readable property **localised in a few layers**. Concatenation expands 2048 dims to 59392, and under 135 training samples with `C=0.1` regularisation the overwhelming majority of the added dimensions are task-irrelevant and end up diluting the discriminative direction.

My original hypothesis (§4.6: signal distributed across layers, so concatenation should help) is falsified and recorded as such. The value of a modification lies in **whether it can falsify**, not in whether it supports the original guess.

---

## 5. Environment adaptation

**Principle: every change exists to make the code run on macOS; none of them touches the method itself.** All of it sits in one standalone commit, diffable in one command:

```bash
git diff ef0cf1e 9ba8f9d        # 6 files, +219 −33
```

| # | File | Change | Motivation | Affects method? |
|---|---|---|---|---|
| 1 | `requirements-mac.txt` (new) | Drop `vllm==0.8.5`, keep every other pin | vLLM has no macOS wheel | No (backend swap only) |
| 2 | `src/extract_features.py:328` | Add `--backend {vllm,hf}`, default `vllm` | The original hardcodes `backend="vllm"` in Phase 1 | No (default preserves original behaviour) |
| 3 | `src/extract_features.py:298` | `device_map="auto"` → auto only on CUDA, otherwise `None` + `.to(device)` | `device_map="auto"` tries multi-device sharding on MPS and errors | No |
| 4 | `src/model.py:106` (via `tools/allow_mps.py`) | Device selection now honours `--device` | The original's `device if torch.cuda.is_available() else "cpu"` is always CPU on a Mac | No (MPS and CPU are numerically equivalent, only speed differs) |
| 5 | `tools/make_subset.py` (new) | Read parquet directly and sample per cell | `extract_features.py` has no `--max_samples`; full data needs 41 h | Yes (scale reduction, declared) |
| 6 | `run_subset.sh` (new) | Chains generate → probe → ablation | Fewer manual command errors | No |

### 5.1 The least obvious trap: `model.py:106`

Original code:

```python
self.device = torch.device(device if torch.cuda.is_available() else "cpu")
```

On a Mac `torch.cuda.is_available()` is **always False**, so whether you pass `--device mps` or `--device cuda`, execution silently falls back to CPU. **No error, no warning** — just 5–10× slower.

How I found it: throughput under `--device mps` was implausibly low → printed `self.device` inside `AgentModel.__init__` → it was `cpu` → traced back to this line. The fix makes it honour the argument:

```python
self.device = torch.device(
    device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
)
```

The original semantics ("requested cuda but no cuda → fall back to cpu") are preserved; explicit mps/cpu requests now take effect.

### 5.2 Why this patch needs calling out

It is **environment adaptation, not a method change**, but it affects the **reproducibility narrative**: a reader who sees "ran on CPU" would assume 47 min / 90 tasks is the hardware limit. In fact MPS was active, and 47 min is the runtime **on the M4 GPU**. Every wall-clock number in this report should be read as "M4 MPS".

---

## 6. Debug diary (excerpt: environment adaptation and the labelling protocol)

The full version goes into the final report; below are the core entries in "what I assumed → what it actually was → how I confirmed it" form.

| # | Blocker | Symptom | Diagnosis (assumed → actual → confirmation) | Fix | Time |
|---|---|---|---|---|---|
| 1 | Unsolvable dependency | `pip install` fails at `vllm==0.8.5` | Assumed a newer Python / a CUDA toolkit was needed → actually vLLM **ships no macOS wheel** at all, Linux + NVIDIA only → confirmed by listing every wheel from `pip download vllm==0.8.5`, none `macosx` | Added `requirements-mac.txt` without vllm | 40 min |
| 2 | Data API rate limit | `HTTP 429 Too Many Requests` | Assumed an HF token was needed / retries would help → actually the `datasets-server` rows API is quota-limited → confirmed by curling the parquet file successfully | Read parquet directly (`tools/make_subset.py`) | 25 min |
| 3 | Silent device fallback | Implausibly low throughput under `--device mps`, no error | Assumed MPS lacked bfloat16 → actually `model.py:106` forces CPU (see §5.1) → confirmed by printing `self.device` | Idempotent patch via `tools/allow_mps.py` | 30 min |
| 4 | Hardcoded backend | Phase 1 still tried to `import vllm` | Assumed the new `--backend` flag was taking effect → actually `backend="vllm"` was a hardcoded string inside `extract_features.py` | Added the CLI flag and threaded it through | 20 min |
| 5 | Unwritable cache dir | `PermissionError: ~/.cache/huggingface` | Assumed the disk was full → actually the sandbox blocks writes to the home directory → confirmed by `touch ~/.cache/test` failing | `HF_HOME=$PWD/.hf_cache` + `HF_HUB_OFFLINE=1` | 15 min |
| 6 | **Label degradation (the big one)** | 43 of 45 `final_response` fields empty; `tool_necessary` at 95.6% | See the detailed write-up below | 256 tok + `max_rounds=8` | 3 h |
| 7 | Long jobs killed | Process vanished after 120 s in the foreground | Assumed OOM → actually a tool-layer timeout SIGTERM cascading to the child → confirmed by `EXIT=143` | Moved everything to the background with `nohup` and on-disk logs | 20 min |

### 6.1 Item 6 in detail: a genuine labelling-protocol bug

**Observed** (first smoke run, 64 tokens × 1 round):

```text
train: 45 tasks, tool_necessary=43/45 (95.6%)
no-tool correct: 2/45
```

A 95.6% "needs a tool" rate is obviously implausible — it implies the model gets almost nothing right. But **this was not a capability problem**; my configuration had broken the harness contract.

**What I assumed**: the model is too small (1.7B) to produce the `\boxed{}` format.
**What it actually was**: two independent causes stacking.

1. `max_new_tokens=64` truncated the output, so `\boxed{...}` was never completed → `extract_boxed` returned empty → `compare_values("", gold)` is False → scored as incorrect.
2. **The subtler layer**: under `no_reasoning` the repository requires that nothing precede `\boxed{}`. If that is violated, the harness appends "Continue" and retries — but `max_rounds=1` means **there is no second round**, so the retry never happens and every non-compliant output becomes "incorrect".

**Confirmation**: printed the `raw_output` of all 45 tasks. 43 contained `\boxed{` without a closing brace (truncation); the other 2 had the model writing a sentence of explanation before the boxed answer (protocol violation, needs a retry). Both categories are fixable with "more tokens + more rounds".

**Fix and validation** (three iterations; the most valuable stretch of this report):

| Round | Configuration | tool_necessary | no-tool correct | Verdict |
|---|---|---|---|---|
| 1 | 64 tok × 1 round | 95.6% | 2/45 | Degenerate, unusable |
| 2 | 256 tok × 3 rounds | 84.4% | 7/45 | Protocol fixed, but 21/45 hit the round cap |
| 3 | 256 tok × 8 rounds | **80.0% / 84.4%** | 9/45 / 7/45 | **Usable**, difficulty gradient correct |

Per-difficulty check from the third round (train / test):

| difficulty | train tool_necessary | test tool_necessary |
|---|---|---|
| easy | 73% | 67% |
| medium | 80% | 93% |
| hard | 87% | 93% |

**The gradient points the right way** (easy < hard), which is independent evidence that the labels are meaningful — if they were noise, the three difficulty groups would not show a monotone trend.

**What this bug means methodologically** (carried into §7):

The repository's `tool_necessary = 0 if correct else 1` conflates **two entirely different failures**:
- (a) the model genuinely got the answer wrong → it really does need a tool;
- (b) the model violated the output format / its output was truncated → the label is wrongly recorded as "needs a tool".

Under the paper's defaults (2048 tokens, 12 rounds) (b) is rare, so the paper's conclusions are unaffected. But **any reproduction that shrinks the generation budget will hit this, and the symptom is "AUROC looks bad" rather than "the program errors"** — a silent, misleading failure mode. Advice to anyone following: **spot-check 5 `raw_output` entries to confirm `\boxed{}` is complete before trusting any AUROC number.**

### 6.2 Label-validity audit: 44–56% of samples never actually finished

Even after fixing `max_rounds=8`, a harder problem remains: **under a 256-token budget, a substantial share of no-tool runs never produce a `\boxed{}` at all**. Under `0 if correct else 1` these are all labelled "needs a tool", yet they are not the same thing as "the model got it wrong".

> The 1024-token main arm added later shows that quadrupling the budget only moves the parseable rate from 54–56% to 54–58% (§4.9). **The bottleneck is protocol compliance, not length**, so the problem in this section cannot be eliminated by adding tokens — but it can be isolated by "filter plus dual reporting".

After checking every `raw_output` (`tools/probe_parseable_only.py`), on the 256-token arm with n=135:

| split | Parseable (non-empty `\boxed{}`) | No `\boxed{}`, defaulted to necessary |
|---|---|---|
| train | 73/135 (54%) | **62 tasks** |
| test | 75/135 (56%) | **60 tasks** |

In other words, **62 of the 103 "necessary" labels on train (60%) are truncation artefacts, not evidence of incapability**. That directly threatens construct validity: the probe may be learning "will this prompt make the output run long" rather than "does this task need a tool".

**The crucial robustness check** — drop every unparseable sample and rerun the layer sweep:

| Setting | n (train/test) | necessary (train/test) | Peak layer | test AUROC |
|---|---|---|---|---|
| Repository default (unfiltered) | 135 / 135 | 103 / 107 | L15 | 0.837 |
| parseable only | 73 / 75 | 41 / 47 | L15 | **0.872** |

**The peak layer is unchanged (L15 in both) and AUROC goes up by 0.035 rather than down.** Two things follow: the L15 signal is not an artefact of a "truncation detector" — on a clean set of tasks where the model really did answer and really did get it wrong, the signal is *sharper*; and the direction of label noise is **dilution** (mixing separable and non-separable samples pushes AUROC down), not **fabrication** (inventing signal from nothing).

This check is the report's most important methodological defence: without it, 0.837 could be dismissed with a single sentence — "is your probe just detecting truncation?" With it, the same number has a structural explanation.

> Cost: after filtering, n falls to 73/75, statistical power drops and the confidence interval on a single-layer AUROC widens. Its role is therefore to **falsify the "pure artefact" hypothesis**, not to provide a more precise point estimate — which is exactly why §4.7 reports both protocols side by side, so the reader sees the repository's original number and the clean number together.


## 7. Conclusions and verdict

### 7.1 Main verdict: **reproduced**

| Protocol | test AUROC | Verdict (against the §3.3 thresholds) |
|---|---|---|
| Main arm · parseable-only | **0.912** | **Reproduced** (inside the paper's 0.89–0.96 range) |
| Main arm · unfiltered (repository protocol) | **0.874** | **Reproduced** (≥0.85) |
| Control arm · unfiltered | 0.837 | Partially reproduced |
| Control arm · parseable-only | 0.872 | Reproduced (clears the bar, but below the main arm) |

Secondary criterion (signal location): the main arm peaks at **L25–L28**, mid-to-late layers, **directionally consistent** with the paper's "signal concentrated in mid-to-late layers".

### 7.2 Three pieces of supporting evidence

1. **The magnitude matches.** 0.912 under the clean protocol sits mid-range in the paper's interval, not on its edge.
2. **It is not a difficulty classifier.** AUROC within the easy group is 0.75–0.86, far above the ~0.5 a pure difficulty signal would produce.
3. **The causal link between label quality and AUROC is isolated.** The two arms' feature tensors are element-wise identical, and a mere 13% label flip yields +0.037 AUROC (§4.9) — the hardest piece of methodological evidence in this report.

### 7.3 Four limitations (which must be presented alongside the verdict)

| Limitation | Impact |
|---|---|
| Backend is HF `transformers` + MPS, not vLLM + CUDA | Different numerical path; AUROC cannot be compared digit-by-digit with the paper |
| `max_new_tokens` = 1024, not the paper's 2048 | Still half the generation budget; but §4.9 shows further increases buy very little |
| 270-task subset (3 per cell), not the full 3150 | Wide confidence intervals; between-group ordering of per-difficulty AUROC is unstable |
| Qwen3-1.7B only, not the paper's multi-model sweep | Does not generalise to the 4B/14B/32B results |

### 7.4 What was not reproduced

**"Probe&Prefill reduces tool calls by 38% with 1.7% accuracy loss" was not reproduced**, and this report makes no claim about its validity. The reason: that result needs a full prefill-then-generate inference pipeline (not faithfully reconstructable without vLLM), and its accuracy depends on the paper's default generation budget.

### 7.5 Two methodological notes on the original

These are not "the paper is wrong"; they are "the paper's evaluation protocol misleads followers under **small samples and non-default configurations**":

1. **`tool_necessary = 0 if correct else 1` conflates "got it wrong" with "never finished"** (§6.1, §6.2). It rarely triggers under the paper's default 2048 tok / 12 rounds; but anyone who lowers the budget to save compute walks into a **silent pseudo-failure that merely looks like "low AUROC"**. Suggestion: mark empty `extract_boxed` results as `invalid` and exclude them from training rather than folding them into the positive class.
2. **"Select the layer by train accuracy" is unreliable at small sample sizes** (§4.3). Across this work's four configurations that logic picked L2 / L1 / L2 / L2 while the true peaks were L15 / L15 / L28 / L25 — **a maximum deviation of 0.11 AUROC**. Moreover, a mere 13% label flip moved the peak from L15 to L28, which shows that **"the best layer" is not itself a robust statistic**; report the whole curve or its integral rather than an argmax.

### 7.6 One-sentence summary

> On macOS + MPS + the HF backend, with a 270-task stratified subset and Qwen3-1.7B, the paper's claim that "linear probes on hidden states reach AUROC 0.89–0.96" **is reproducible** (0.912 under the clean protocol); the real obstacle is not compute but a **silent failure in the labelling protocol** — the one trap in this kind of reproduction that can make a working method look broken.

---

## 8. Reproduction instructions

```bash
git clone https://github.com/jiashuhan2002-creator/when2tool-repro.git && cd when2tool-repro
python3 -m venv .venv && .venv/bin/pip install -r requirements-mac.txt

# 1) Stratified subset (3 per cell → 135 + 135)
.venv/bin/python tools/make_subset.py --split train --per-cell 3 --out data/scaled_train.json
.venv/bin/python tools/make_subset.py --split test  --per-cell 3 --out data/scaled_test.json

# 2) Generation + feature extraction (~10–11 h on M4 MPS)
HF_HOME="$PWD/.hf_cache" .venv/bin/python -u src/extract_features.py \
  --model_path qwen3-1.7b --data_path ./data/scaled_train.json \
  --data_path_test ./data/scaled_test.json --output_dir ./probe_data/scaled1024 \
  --phase all --backend hf --device mps --max_new_tokens 1024 --max_rounds 8

# 3) Layer sweep + per-difficulty + concatenation ablation + two-arm comparison
.venv/bin/python tools/report_tables.py --data_dir ./probe_data/scaled1024
.venv/bin/python tools/compare_arms.py --out_png probe_data/scaled1024/compare_arms.png

# 0) One-command equivalent of steps 1)–3)
./run_subset.sh qwen3-1.7b mps 1024 3
```

The `HF_HOME` environment variable points at `.hf_cache/` inside the repository (gitignored; the 3.8 GB of model weights are not committed).
