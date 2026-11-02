# REPORT.md — Evidence dump for the project report

Repo: `lora-adapters` (branch `master`). Evidence collected 2026-11-01 from the working tree, git history, notebooks (code and saved outputs), result files, training checkpoints, and read-only environment queries.

Tag legend: **VERIFIED** = found in the repo or measured by a read-only command. **INFERRED** = my reasoning, with the basis stated. **NOT FOUND** = needed for the report but absent.
Source tags: `[src: path:line]`; notebook cells use the 0-based index of the `.ipynb` JSON `cells` array (`[src: adapters/base.ipynb cell N]`).
"Session log" = a training log written outside the repo during the local runs on 2026-11-01; the exact lines are copied into §5 so they do not depend on that file surviving.

## Read-first: findings that change how the report should be written

1. **Only 2 of 4 adapters have ever been evaluated.** All results in `eval/` use Programming + Finance only. Cybersecurity and Healthcare adapters exist (trained 2026-11-01) but have no evaluation of any kind. VERIFIED [src: eval/*.json contain only `programming`/`finance` keys].
2. **`infer.py` still loads only 2 adapters, but `centroids.pkl` now has 4 domains.** `infer.py:10-11` loads programming + finance; `infer.py:33` passes only those two weights. The router (`infer.py:17-23`) now softmaxes over 4 centroids, so the programming + finance weights no longer sum to 1 and queries about the two new domains get low weights on both loaded adapters. VERIFIED (code) / INFERRED (effect; not run). The README/CLAUDE.md claim that a new domain needs no other file changed [src: CLAUDE.md "Conventions"] is not true for `infer.py`.
3. **The "static" baseline in `midsem_10_queries.json` is the *wrong-domain* adapter for 7 of 10 queries** (see §12). Static-vs-router latency and quality comparisons from that file are not a like-for-like static baseline. VERIFIED [src: eval/evaluation_pipeline.ipynb cell 6].
4. **No base-model-only result exists in either results file**, although README says the benchmark compares "base model, static adapter, and router-weighted responses" [src: README.md:27]. NOT FOUND.
5. **No quantitative quality metric has been computed anywhere** (no accuracy/F1/exact match/judge scores). Only qualitative text samples, top-1 routing weights, and latency. VERIFIED by absence.
6. **Router v2 (logistic regression) is not implemented** — planned only [src: PROJECT_PLAN.md:30]. NOT FOUND in code.
7. **Repo hygiene:** commit `2d589ea` tracks a 990,280,096-byte base-model `model.safetensors` under `Users/predator neo/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-peft/snapshots/…/`; `.git` is 765.72 MiB. VERIFIED [src: `git ls-files -s`, `git count-objects -vH`]. This will block a push to GitHub (100 MB per-file limit — INFERRED from general knowledge, not from the repo).
8. **Uncommitted working-tree changes exist** (see §6). The finance adapter in the working tree is the retrained one; `HEAD` still has the original Colab-trained finance adapter. Any result in `eval/` was produced by the original finance adapter, not the current one.

---

### 0. Repo Overview

**Directory tree (depth 3; excludes `.git`, `__pycache__`, `.ipynb_checkpoints`, `runs/*/checkpoint-*` internals, `Users/` HF cache):**

```
lora-adapters/
├── CLAUDE.md                     5,871 B   working rules / conventions (git-ignored)
├── PROJECT_PLAN.md               4,785 B   timeline, state, next steps (git-ignored)
├── README.md                     4,966 B   project description (untracked)
├── REPORT.md                               this file (untracked)
├── .gitignore
├── train_adapter.py              1,571 B   trains one domain adapter (argparse)
├── build_centroids.py              616 B   builds centroids.pkl from data/*.jsonl
├── infer.py                      1,921 B   interactive router + weighted adapters + generation
├── centroids.pkl                 6,455 B   4 domains x 384-d (working tree); 3,254 B at HEAD
├── adapters/
│   ├── base.ipynb              448,033 B   Colab notebook: model load, adapter training, router prototype
│   ├── programming_adapter/            adapter_config.json, adapter_model.safetensors, README.md
│   ├── finance_adapter/                (same)
│   ├── cybersecurity_adapter/          (same, new, untracked)
│   ├── healthcare_adapter/             (same, new, untracked)
│   └── test_adapter/                   (same; interrupted alpaca-300 trial)
├── data/
│   ├── programming.jsonl       3,652,462 B
│   ├── finance.jsonl             552,999 B
│   ├── cybersecurity.jsonl     1,792,303 B
│   └── healthcare.jsonl        1,911,976 B
├── eval/
│   ├── evaluation_pipeline.ipynb 297,814 B  Colab notebook: 10-query benchmark + plots
│   ├── midsem_10_queries.json     10,436 B  benchmark results (weights, latency, VRAM, responses)
│   ├── my_eval_results.json        9,779 B  second 10-query set (labelled domains)
│   ├── weight_distribution.png    25,649 B
│   └── latency_comparison.png     26,188 B
├── notes/                                   EMPTY directory
├── runs/                                    (git-ignored) trainer output: runs/{cybersecurity,finance,healthcare}/checkpoint-{188,376,564}
└── Users/predator neo/.cache/huggingface/hub/...   945 MB HF cache committed by accident (see Read-first #7)
```
[src: `find`/`stat` on the working tree, 2026-11-01]

**Main files / entry points** (VERIFIED):
| File | Purpose | Entry point |
|---|---|---|
| `train_adapter.py` | SFT one LoRA adapter on one `data/*.jsonl` | `python train_adapter.py --domain <name> --data data/<name>.jsonl [--batch_size N --grad_accum N]` |
| `build_centroids.py` | mean MiniLM embedding of each domain's *instructions* → `centroids.pkl` | `python build_centroids.py` |
| `infer.py` | interactive loop: embed → softmax weights → `add_weighted_adapter` → generate | `python infer.py` |
| `adapters/base.ipynb` | original Colab workflow (training + router prototype) | notebook |
| `eval/evaluation_pipeline.ipynb` | Colab benchmark producing `midsem_10_queries.json` and both PNGs | notebook |

**Figures** (VERIFIED, both viewed):
| File | Content | Caption suggestion |
|---|---|---|
| `eval/weight_distribution.png` | stacked bars, Programming % vs Finance % per query Q1–Q10 | "Router weight assigned to each domain adapter for the 10 benchmark queries" |
| `eval/latency_comparison.png` | grouped bars, static-adapter vs router generation time (s) per query | "Generation latency per query: static adapter vs router-weighted adapter (single run, Colab T4)" |

No architecture/pipeline diagram files in the repo (NOT FOUND) although the plan says diagrams were produced for the presentation [src: PROJECT_PLAN.md:19]. The README contains an ASCII pipeline [src: README.md:9-17].

**Gaps:** architecture diagram source; empty `notes/`; no `requirements.txt` although README references it [src: README.md:71, 80] (`ls requirements.txt` fails).

---

### 1. Introduction

Repo's own words [src: README.md:3-7]:
- "A framework that fine-tunes multiple domain-specific LoRA adapters on top of a single frozen Small Language Model (SLM), and dynamically routes or blends them at inference time based on the semantic content of the user's query — instead of relying on one statically-selected adapter."
- "Most LoRA-based systems activate exactly one domain adapter per query… This project replaces that static selection with an adaptive routing engine: a query is embedded, compared against each domain's precomputed centroid embedding, and converted into a set of weights via a temperature-scaled softmax over cosine similarity."

**What was concretely built** (all VERIFIED):
- 4 LoRA adapters on Qwen2.5-0.5B-Instruct: programming, finance (trained on Colab T4, finance later retrained locally), cybersecurity, healthcare (trained locally 2026-11-01).
- Router v1 (centroid cosine similarity + softmax, T=0.1) in notebooks and `infer.py`.
- Weighted combination via `add_weighted_adapter(..., combination_type="linear")` [src: infer.py:33-35].
- Training/centroid scripts, CLI-driven (migrated off Colab [src: PROJECT_PLAN.md:20]).
- A 10-query benchmark (two query sets) with latency/VRAM/weights plus 2 plots.

**Gaps:** project motivation beyond README; team/user-need statement.

---

### 2. Problem Statement and Objectives

Stated research question [src: PROJECT_PLAN.md:46]: "Can adaptive, context-aware routing of LoRA adapters improve multi-domain task performance while maintaining reasonable computational efficiency compared with conventional static adapter selection?" — labelled "hypothesis, not yet validated".

| Objective (from your plan) | Status | Evidence |
|---|---|---|
| Frozen base SLM + several domain LoRA adapters | **Achieved** | 4 adapters, same base [src: adapters/*/adapter_config.json `base_model_name_or_path`] |
| Domains: Programming, Healthcare, Finance (Cyber possibly deferred) | **Achieved** (all 4 trained) | adapter folders; cyber/healthcare trained 2026-11-01 [src: runs/*/trainer_state.json] |
| Router v1 (centroid similarity + softmax) | **Achieved** | [src: infer.py:17-23; base.ipynb cell 25] |
| Router v2 (logistic regression) | **Not done** (planned) | [src: PROJECT_PLAN.md:30] |
| `add_weighted_adapter()` or top-1 `set_adapter()` | **Achieved** (weighted + static `set_adapter`) | [src: infer.py:33-35; evaluation_pipeline.ipynb cell 4] |
| Weights top-1 fallback "implemented and tested" | **Partly** — static `set_adapter` baseline is tested, but there is no code path that falls back to top-1 inside `infer.py` | [src: README.md:26; infer.py] INFERRED |
| Free Colab/Kaggle T4 training | **Partly** — Colab T4 for programming/finance [src: base.ipynb metadata `gpuType: T4`]; cyber/healthcare/finance-retrain on local RTX 4050 Laptop. No Kaggle evidence. | |
| Fallback model TinyLlama-1.1B | **Not found** — no mention anywhere (grep) | NOT FOUND |
| Eval: base vs static vs router | **Partly** — static vs router only; no base results | Read-first #4 |
| Eval: routing accuracy | **Partly** — top-1 on 8 labelled pure queries (see §12) | |
| Eval: quality single- vs mixed-domain | **Partly** — qualitative samples only, no metric | |
| Eval: latency | **Partly** — single run per query, includes confounds (§12) | |
| Router handles mixed-domain queries | **Observed, not scored** — 2 labelled mixed queries (my_eval) | |

**Gaps:** formal objectives/scope text (only the research question exists); success criteria.

---

### 3. Literature Review

**Everything cited in the repo** (VERIFIED) — the repo contains almost no citations:
| Item | Where | Citation detail present? |
|---|---|---|
| LoRA — Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022 | [src: PROJECT_PLAN.md:42] | authors (Hu et al.), title, venue, year — **no** arXiv ID/URL |
| Multi-LoRA | [src: PROJECT_PLAN.md:35; README.md:106] | name only |
| Dynamic LoRA | [src: PROJECT_PLAN.md:36] | name only |
| LoRA Routing | [src: PROJECT_PLAN.md:37] | name only |
| AdapterFusion | [src: PROJECT_PLAN.md:38; README.md:106] | name only |
| Mixture-of-LoRA-Experts / MoE-style adapters | [src: PROJECT_PLAN.md:39; README.md:106] | name only |
| Adapter composition methods | [src: PROJECT_PLAN.md:40] | name only |
| Sentence-Transformers `all-MiniLM-L6-v2` | [src: README.md:34, 102; infer.py:13] | model/library name only |
| Qwen2.5-0.5B-Instruct | [src: README.md:21; every script] | HF ID only; no Qwen report cited |
| PEFT / TRL / Transformers / Accelerate / Datasets | [src: README.md:102] | names only |

Not mentioned anywhere in the repo (grep, NOT FOUND): QLoRA, AdaLoRA, HydraLoRA, LoRAHub, Sentence-BERT/SBERT, TinyLlama, any arXiv ID. "DoRA" only appears as `use_dora: false` in adapter configs. Qwen / TinyLlama model reports: NOT FOUND.

Implied comparison to related work [src: README.md:106]: the README states that "adapters can be weighted-combined" already exists in the literature as Multi-LoRA / AdapterFusion / Mixture-of-LoRA-Experts, and that the novel contribution "is being finalized pending a full literature review" [src: PROJECT_PLAN.md:32-46 — review "in progress"].

**Gaps:** the entire literature review (full citations, gap analysis, comparison table). `notes/` is empty. No per-member search notes in the repo.

---

### 4. System Architecture

**Pipeline** [src: README.md:9-17; CLAUDE.md "Architecture"]:
```
User Query -> Query Preprocessing -> Semantic Embedding (all-MiniLM-L6-v2)
 -> Adaptive LoRA Routing Engine (cosine sim vs domain centroids -> softmax, T=0.1)
 -> Weighted Adapter Combination (add_weighted_adapter, linear)
 -> Base SLM (Qwen2.5-0.5B-Instruct, frozen, fp16) -> Response
```

| Component | Where | Input → Output | Notes |
|---|---|---|---|
| Preprocessing / prompt format | `infer.py:36`; `train_adapter.py:19` | query → `"### Instruction:\n{q}\n\n### Response:\n"` | No other normalisation exists; README calls it "normalizes the raw query" [src: README.md:33] — in code it is only the template wrap. INFERRED |
| Embedder | `infer.py:13` | text → 384-d vector | `SentenceTransformer('all-MiniLM-L6-v2')`; centroid shape `(384,)` verified from `centroids.pkl` |
| Centroids | `build_centroids.py:7-12`, `centroids.pkl` | domain instructions → mean embedding | Built from **all 1500 instructions per domain** (not responses). Keys now: cybersecurity, finance, healthcare, programming |
| Router | `infer.py:17-23` (`route_query`) | query → `{domain: weight}` | cosine sim; `/temperature` (default 0.1); stable softmax; **no threshold, no top-k, no "unknown" class** |
| Combination | `infer.py:31-35` | weights → adapter `"combined"` | deletes old `combined` first; hard-coded to `["programming","finance"]` |
| Generation | `infer.py:36-40` | prompt → text | `max_new_tokens=100`, `do_sample=False` |
| Trainer | `train_adapter.py` | jsonl → adapter dir | see §5 |

**Libraries/versions** (local venv, used for the 2026-11-01 runs and read-only checks): Python 3.11.0; torch 2.14.1+cu126 (CUDA 12.6, cuDNN 9.10.02); transformers 5.21.0; peft 0.21.2; trl 1.16.0; datasets 5.1.0; accelerate 1.15.0; sentence-transformers 6.2.0; numpy 2.4.6; scikit-learn 1.9.1 (installed as a dependency; unused); bitsandbytes **not installed**. VERIFIED [src: `pip list` / imports in `C:\Users\predator neo\venvs\lora-adapters`, outside the repo].
Colab-era: Python 3.13 [src: eval/evaluation_pipeline.ipynb cell 5 output, path `/usr/local/lib/python3.13/dist-packages/peft/...`]; PEFT 0.20.0 [src: adapters/programming_adapter/adapter_config.json `peft_version`; HEAD README of programming/finance/test adapters]. Other Colab versions (torch, transformers, trl) **NOT FOUND** — installs were unpinned [src: base.ipynb cell 1: `!pip install -q transformers peft trl accelerate datasets bitsandbytes`].

**Adapter repository layout** (VERIFIED):
| Folder | `adapter_model.safetensors` | `adapter_config.json` | peft_version in config | mtime |
|---|---|---|---|---|
| programming_adapter | 2,175,168 B | 1,081 B | 0.20.0 | 2026-10-29 11:37 |
| finance_adapter (retrained) | 2,175,168 B | 1,149 B | 0.21.2 | 2026-11-01 23:29 |
| cybersecurity_adapter | 2,175,168 B | 1,149 B | 0.21.2 | 2026-11-01 22:10 |
| healthcare_adapter | 2,175,168 B | 1,149 B | 0.21.2 | 2026-11-01 22:53 |
| test_adapter | 2,175,168 B | 1,081 B | 0.20.0 | 2026-10-29 11:37 |

2,175,168 B = 2.0744 MiB each; 4 production adapters = 8,700,672 B = 8.2976 MiB.

`adapter_config.json` (identical across adapters except field order/`peft_version`/new null keys) [src: adapters/finance_adapter/adapter_config.json]:
```
"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct", "peft_type": "LORA", "task_type": "CAUSAL_LM",
"r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "bias": "none", "target_modules": ["q_proj","v_proj"],
"use_dora": false, "use_rslora": false, "inference_mode": true, "init_lora_weights": true,
"fan_in_fan_out": false, "modules_to_save": null, "layers_to_transform": null, "peft_version": "0.21.2"
```

**Gaps:** diagram source; component-level latency breakdown (see §7).

---

### 5. Methodology

**Base model**
- HF ID `Qwen/Qwen2.5-0.5B-Instruct` [src: train_adapter.py:14; infer.py:6; adapter_config.json].
- Parameters: 494,573,440 total *including* LoRA; 540,672 trainable (0.1093%) [src: base.ipynb cell 7 output; reproduced in training logs]. Base alone = 494,573,440 − 540,672 = **494,032,768** (derived).
- Precision: fp16 weights (`torch_dtype=torch.float16`) [src: train_adapter.py:16; infer.py:9]; training with `fp16=True` [src: train_adapter.py:26]. No quantisation used — `bitsandbytes` was pip-installed in notebooks but never imported [src: base.ipynb cell 1; no `BitsAndBytesConfig` in any file], and is not in the local venv.
- Licence: NOT FOUND in repo. Max context length: NOT FOUND in repo. Training truncation length = TRL `SFTConfig.max_length` default **1024** (read from `SFTConfig(output_dir='x')` in the local venv, trl 1.16.0; not set in repo code). The Colab TRL version is unknown, so its default is NOT FOUND.
- Download size shown in notebook: 988 MB [src: base.ipynb cell 3 output]; `model.safetensors` tracked in git = 990,280,096 B.

**Datasets** (all `instruction`/`response` JSONL, 1500 rows each, no missing/empty values, no cross-domain duplicate instructions; VERIFIED by scripts run read-only)
| Domain | File bytes | Rows | Unique instructions | Tokens/example (template applied, Qwen tokenizer): mean / median / p95 / max | Examples > 1024 tokens (truncated) | Total tokens |
|---|---|---|---|---|---|---|
| programming | 3,652,462 | 1500 | 1500 | 521.2 / 505 / 814 / 1591 | 15 | 781,871 |
| finance | 552,999 | 1500 | 1500 | 70.2 / 69 / 105 / 184 | 0 | 105,292 |
| cybersecurity | 1,792,303 | 1500 | 1500 | 277.0 / 255 / 468 / 705 | 0 | 415,448 |
| healthcare | 1,911,976 | 1500 | **1491** (9 duplicates) | 271.6 / 210 / 703 / 1328 | 18 | 407,363 |
- **Source / HF ID / URL / licence for every dataset: NOT FOUND.** Nothing in the repo records where the data came from or under what terms. README.md:23 only says the datasets were "sourced", and PROJECT_PLAN.md:74 says the cybersecurity/healthcare sets "haven't been quality-checked".
- Rows before vs after cleaning: NOT FOUND (no cleaning script; current files are the only version). Train/val/test split: **NOT FOUND — no split exists; all 1500 rows of each domain are used for training and there is no validation or test split in any script** [src: train_adapter.py:18 `split="train"`, no `train_test_split`].
- Content description (VERIFIED from rows; provenance still NOT FOUND):
  - programming: 1500/1500 instructions begin "Write a solution to the following coding problem:" (single template) [src: data/programming.jsonl].
  - finance: short Q&A, whole examples (instruction + response) average ≈ 70 tokens; many instructions refer to "the context"/"the formula" without supplying it (e.g. "Based on the…", 65 rows) [src: data/finance.jsonl row 0, row 700; first-3-word counts].
  - cybersecurity: all 1500 instructions contain a `Context:` line; 274 contain a MITRE group ID of the form `G####`; 573 responses mention "ATT&CK"; adversary-profile and red-team "emulation plan" prompts [src: data/cybersecurity.jsonl].
  - healthcare: medical Q&A; 337 responses mention "MedlinePlus", 231 "Human Phenotype Ontology"; 600 of 1500 instructions start "What are the…"; 238 responses begin with the text "What are the signs…" [src: data/healthcare.jsonl].
- **Prompt template (exact)** [src: train_adapter.py:19; infer.py:36; base.ipynb cell 13]:
  `"### Instruction:\n{instruction}\n\n### Response:\n{response}"` — inference uses the same string ending after `### Response:\n`.
- Preprocessing: only the template map; no cleaning/dedup; TRL adds EOS, tokenises, builds labels, truncates, drops fully-masked examples [src: base.ipynb cell 14 output progress bars]. Loss covers the whole text (prompt + response) because `completion_only_loss=None` and a single `text` column is used — INFERRED from defaults read in the venv + dataset format.

**LoRA training**
- `LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")` [src: train_adapter.py:21].
- `SFTConfig(output_dir=f"./runs/{domain}", num_train_epochs=3, per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum, learning_rate=2e-4, logging_steps=10, save_strategy="epoch", fp16=True)` [src: train_adapter.py:25-26]. CLI defaults `--batch_size 4 --grad_accum 2` [src: train_adapter.py:10-11] (effective batch 8). **The three local runs used `--batch_size 2 --grad_accum 4`** (effective batch still 8) because the default OOM'd on the 6 GB GPU (see below).
- Colab runs (programming, finance original) used the notebook equivalents: epochs 3, batch 4, grad-accum 2, lr 2e-4, fp16 [src: base.ipynb cell 14, cell 20].
- Other effective settings = library defaults at trl 1.16.0 / transformers 5.21.0 (read from an instantiated `SFTConfig`): linear LR scheduler, warm-up 0, **seed 42**, weight decay 0.0, optimizer `adamw_torch_fused`, max_grad_norm 1.0, packing False, gradient checkpointing True. None are set in repo code. Colab-run defaults: NOT FOUND.
- Steps: 1500 rows / effective batch 8 → 188 steps/epoch → 564 steps total [src: runs/*/checkpoint-564/trainer_state.json `global_step: 564`].
- Model load: `device_map="auto"` is still used in `train_adapter.py:16` and the notebooks, despite the CLAUDE.md gotcha against it (that gotcha concerns `add_weighted_adapter`; training was unaffected). VERIFIED.

**Training results — local runs, 2026-11-01** (Hardware: RTX 4050 Laptop 6141 MiB, driver 621.01)
| Adapter | Runtime (s) | Samples/s | Steps/s | Mean train loss | Final-epoch entropy | Final mean token acc. | num_tokens | total_flos |
|---|---|---|---|---|---|---|---|---|
| cybersecurity | 1780 (29 m 40 s) | 2.528 | 0.317 | 2.376 | 2.201 | 0.5418 | 1.251e+06 | 3.2243e15 |
| healthcare | 2580 (43 m) | 1.744 | 0.219 | 1.506 | 1.263 | 0.7356 | 1.22e+06 | 3.6088e15 |
| finance (retrain) | 2059 (34 m 19 s) | 2.185 | 0.274 | 2.422 | 2.253 | 0.5529 | 3.204e+05 | 7.9579e14 |

Exact summary lines [src: session training logs, outside repo]:
```
cybersecurity {'train_runtime': '1780', 'train_samples_per_second': '2.528', 'train_steps_per_second': '0.317', 'train_loss': '2.376', 'entropy': '2.201', 'mean_token_accuracy': '0.5418', 'num_tokens': '1.251e+06', 'epoch': '3'}
healthcare    {'train_runtime': '2580', 'train_samples_per_second': '1.744', 'train_steps_per_second': '0.219', 'train_loss': '1.506', 'entropy': '1.263', 'mean_token_accuracy': '0.7356', 'num_tokens': '1.22e+06', 'epoch': '3'}
finance       {'train_runtime': '2059', 'train_samples_per_second': '2.185', 'train_steps_per_second': '0.274', 'train_loss': '2.422', 'entropy': '2.253', 'mean_token_accuracy': '0.5529', 'num_tokens': '3.204e+05', 'epoch': '3'}
all three     trainable params: 540,672 || all params: 494,573,440 || trainable%: 0.1093
```
Loss curve points (logged every 10 steps; 56 logs/run) [src: runs/<domain>/checkpoint-564/trainer_state.json]:
| Adapter | step 10 | step 190 | step 380 | step 560 | min (step) | mean_token_accuracy @560 |
|---|---|---|---|---|---|---|
| cybersecurity | 3.063347816467285 | 2.375345230102539 | 2.2135150909423826 | 2.2538162231445313 | 2.116236686706543 (520) | 0.5448735870420933 |
| healthcare | 2.0924510955810547 | 1.4995076179504394 | 1.580146312713623 | 1.4110443115234375 | 1.2823665618896485 (490) | 0.6831960201263427 |
| finance | 3.214389419555664 | 2.4281633377075194 | 2.2861049652099608 | 2.270973968505859 | 2.270973968505859 (560) | 0.5286334335803986 |
- Checkpoints saved at steps 188/376/564 under `runs/<domain>/` (git-ignored).
- **Original Colab runs (programming, finance v1): loss/time NOT FOUND** — the notebook trainer outputs are only an `IPython.core.display.HTML object` (the loss table was not saved) [src: base.ipynb cells 14, 20].
- OOM event (session observation, no preserved log): the first cybersecurity attempt with `--batch_size 4 --grad_accum 2` failed with `torch.AcceleratorError: CUDA error: out of memory` at epoch ≈ 0.91 (≈ step 170/564) with GPU memory ≈ 5.8 of 6.1 GB; no adapter was saved; the log was overwritten by the rerun. NOT FOUND in repo.
- The earlier `test_adapter` trial: alpaca first 300 rows as stand-in, `trainer.train()` ended in `KeyboardInterrupt`, and the adapter was saved anyway; subsequent greedy output was gibberish ("A、 《国网办事处 …") [src: base.ipynb cells 5, 8, 9]. Then a plain-prompt re-test produced coherent text [cell 10].

**Router(s)**
- **Router v1** (VERIFIED) [src: infer.py:17-23; base.ipynb cell 25; build_centroids.py]:
  1. embed query with MiniLM; 2. cosine similarity to each domain centroid; 3. divide by temperature (**0.1**); 4. stable softmax; 5. weights → `add_weighted_adapter(adapters=[...], weights=[...], adapter_name="combined", combination_type="linear")`; `set_adapter("combined")`.
  - Centroids = mean MiniLM embedding of a domain's training **instructions only**. Notebook prototype used the first **100** instructions [src: base.ipynb cell 24]; the midsem benchmark notebook also used 100 [src: evaluation_pipeline.ipynb cell 3 `.select(range(100))`]; `my_eval_results.json` and `build_centroids.py` use the **full 1500** [src: base.ipynb cell 34; build_centroids.py:10].
  - Temperature rationale: raw similarities too close at T=1.0 [src: PROJECT_PLAN.md:16]. Only T=0.1 appears in code; no sweep is saved.
  - No thresholds/top-k/abstain. No router training (v1 is training-free).
- **Router v2**: NOT FOUND (planned: logistic regression over the same MiniLM embeddings [src: PROJECT_PLAN.md:30]).
- Known issue, documented [src: README.md:25; PROJECT_PLAN.md:23; CLAUDE.md]: short generically-phrased queries (e.g. stack vs queue; Big O) misroute to Finance, attributed to phrasing-template bias. Re-running with full-dataset centroids did not fix it (see §12: same two queries still misrouted at 0.7486 and 0.8870 Finance).

**Inference/generation settings:** greedy (`do_sample=False`); `max_new_tokens=100` in `infer.py:39` and the midsem benchmark; **80** in the first prototype and `my_eval_results.json` [src: base.ipynb cell 32]; prompt template above. No temperature/top-p/repetition penalty set.

**Evaluation setup** (details in §12): two separate 10-query sets; no train/val/test split on the training side; no quality metric; manual reading only.

**Gaps:** dataset provenance, licences, cleaning, and splits; Colab training metrics; base-model license/context length; seeds for Colab runs; judge/rating procedure (none).

---

### 6. Environment

**Hardware used**
- Local (cyber, healthcare, finance-retrain, all read-only checks): Intel Core i7-14700HX (20 cores / 28 logical), 15.7 GB RAM, NVIDIA GeForce RTX 4050 Laptop GPU 6141 MiB, driver 621.01; Windows 11 Home Single Language 10.0.26300. VERIFIED [src: nvidia-smi, WMI queries].
- Colab (programming, finance v1, both benchmark notebooks): GPU type `T4` [src: adapters/base.ipynb and eval/evaluation_pipeline.ipynb notebook metadata `gpuType: T4`]; Python 3.13. VRAM/CPU/RAM of that session: NOT FOUND. Whether it was free or paid Colab: NOT FOUND.

**Libraries/versions:** see §4 (local venv above). System Python 3.14 has CPU-only torch 2.14.1+cpu (cannot train); Python 3.11 base interpreter has torch 2.7.0+cpu. The `venv` is **outside** the repo; there is no `requirements.txt` or `environment.yml` in the repo (README references one) — NOT FOUND.

**How to reproduce (derived; commands are those in README.md:76-98, adjusted):**
```
python -m venv venv
# install torch matching your CUDA version from pytorch.org (local run used: pip install torch --index-url https://download.pytorch.org/whl/cu126)
pip install transformers==5.21.0 peft==0.21.2 trl==1.16.0 datasets==5.1.0 sentence-transformers==6.2.0 accelerate==1.15.0
python train_adapter.py --domain cybersecurity --data data/cybersecurity.jsonl --batch_size 2 --grad_accum 4
python train_adapter.py --domain healthcare   --data data/healthcare.jsonl   --batch_size 2 --grad_accum 4
python train_adapter.py --domain finance      --data data/finance.jsonl      --batch_size 2 --grad_accum 4
python build_centroids.py
python infer.py          # note: loads only programming + finance (see Read-first #2)
```
Version pins are the versions actually installed locally (VERIFIED), not from a lock file.

**Git history** (VERIFIED, `git log`, `git shortlog`):
- Commits: **1** — `2d589ea` "Migrate LoRA pipeline from Colab to local", 2026-10-29 12:30:35 +0530, author Vighnesh Gupta. Single branch `master`. Commits per author: Vighnesh Gupta 1.
- History does not cover the semester (plan says start ~July 7 [src: PROJECT_PLAN.md:5]); Colab-era work is not in history. Per-member contributions: NOT FOUND.
- Uncommitted working-tree changes at time of writing (`git status`):
  - modified: `.gitignore` (adds `CLAUDE.md`, `PROJECT_PLAN.md`), `adapters/finance_adapter/{README.md,adapter_config.json,adapter_model.safetensors}` (retrained), `build_centroids.py` (now globs `data/*.jsonl` instead of 2 hard-coded domains), `centroids.pkl` (2 → 4 domains), `train_adapter.py` (adds `--batch_size`, `--grad_accum` with defaults 4 and 2).
  - untracked: `README.md`, `REPORT.md`, `adapters/cybersecurity_adapter/`, `adapters/healthcare_adapter/`.
  - ignored: `runs/`, `CLAUDE.md`, `PROJECT_PLAN.md`.

**Gaps:** requirements/lock file; Colab environment specifics; team commit history.

---

### 7. Sustainability

- **Trainable-parameter fraction:** 540,672 / 494,573,440 = **0.1093%** per adapter [src: base.ipynb cell 7; training logs; README.md:47].
- **Adapter size vs full model:** 2,175,168 B per adapter vs 990,280,096 B base `model.safetensors` ⇒ **≈ 0.22%** (0.2197%). All 4 adapters together = 8,700,672 B (8.30 MiB) vs one 944.4 MiB base copy. The adapter size equals 540,672 params × 4 B (fp32) = 2,162,688 B plus ≈ 12,480 B header — INFERRED (adapters are stored in fp32).
- **VRAM:** inference peak (`torch.cuda.max_memory_allocated` after loading base + 2 adapters + 1 blend) mean 1047.38 MB, range 1046.77–1048.17 MB across 10 queries [src: eval/midsem_10_queries.json `peak_vram_mb`]. Training VRAM: ≈ 5.8 of 6.1 GB at batch 4/accum 2 (OOM'd); batch 2/accum 4 ran (e.g. observed 3.2–5.7 GB on different runs) — session observations, no preserved log; NOT FOUND in repo.
- **Training time per adapter (local):** 1780 s / 2580 s / 2059 s; total **6419 s = 1.783 h** of training for three adapters [src: §5 table; 1780+2580+2059]. Original Colab times: NOT FOUND.
- **Modularity (adding a domain without retraining base):** VERIFIED in design/process — base is frozen; new domain = new `data/<d>.jsonl` + `train_adapter.py` + `build_centroids.py` [src: CLAUDE.md "Conventions"]. Demonstrated in practice: two domains added after mid-sem without touching the other adapters. Caveat: `infer.py` hard-codes adapter names (Read-first #2).
- **Routing overhead:** router math (embed + cosine + softmax) mean **12.46 ms**, median 8.88 ms, range 6.64–33.96 ms over 10 queries [src: eval/midsem_10_queries.json `router_math_ms`]. **Not measured:** the cost of `add_weighted_adapter()` + `set_adapter()` per query (it runs outside the timed regions: [src: evaluation_pipeline.ipynb cell 6, timers start after `set_adapter`]) — NOT FOUND. In `infer.py` that adapter merge is rebuilt on every query (`infer.py:31-35`).
- **Energy/carbon:** no `codecarbon` or power logs; NOT FOUND.
- **Reuse of open-source assets:** Qwen2.5-0.5B-Instruct, all-MiniLM-L6-v2, Hugging Face stack; datasets reused (provenance NOT FOUND). A MiniLM licence line `license: apache-2.0` exists in the local HF cache README (outside the repo).

**Gaps:** energy/carbon; VRAM during training; adapter-merge latency; Colab hours.

---

### 8. Safety

- **Disclaimers / refusals in outputs:** none observed in any saved model output (benchmark outputs contain no medical/financial disclaimers). VERIFIED by reading both result files and notebook outputs. No refusal logic or content filter exists in `infer.py` or the training scripts.
- **Disclaimer-like text in *training data*** (keyword counts of responses): finance — "consult" 5, "authorized" 1; healthcare — "consult" 8; cybersecurity — "authorized" 111, "with written authorization" 9, "consult" 4, "I cannot" 2; programming — "disclaimer" 1, "authorized" 3. No "not medical advice"/"not financial advice" phrasing in any domain. [src: keyword scan of data/*.jsonl]
- **Dual-use data:** cybersecurity data contains adversary-profile and "red-team emulation playbook" tasks framed as authorized engagements (111 responses mention "authorized"); 56 rows with IPv4 addresses, 269 with URLs, 82 with hash-like strings (regex; likely indicators of compromise, not verified) [src: data/cybersecurity.jsonl; example instruction: "Could you draft a red-team emulation playbook that closely imitates the tactics of RTM for our authorized engagement?"]. No safety evaluation of the cybersecurity adapter has been done.
- **Out-of-domain / ambiguous queries:** the router has **no abstain option** — softmax always splits weight across known domains; an out-of-domain query would still receive a domain-skewed adapter mix [src: infer.py:17-23]. Real observed case: "What is the difference between a stack and a queue?" → Finance 0.7486 [src: eval/my_eval_results.json].
- **Hallucination / failure cases seen in outputs** (VERIFIED text; correctness judgements are mine and INFERRED from standard domain knowledge — verify before citing):
  1. `evaluation_pipeline.ipynb` cell 5: "What happens to bond prices when interest rates rise?" — router weights programming 0.0757 / finance 0.9243; **router (blended) answer: "When interest rates rise, the price of bonds increases."** — incorrect (bond prices fall when rates rise). The *static* (programming adapter) answer was correct ("…the price of bonds typically falls…").
  2. `midsem_10_queries.json` Q7: router response "…execute trades at extremely fast speeds, often exceeding 100 times per second" — unsupported/likely wrong specific.
  3. `midsem_10_queries.json` Q5 router response invents example data (`'Date': ['2017-01-01', …]`) and is cut off by the token limit; Q2/Q9/Q10 and several others truncated at `max_new_tokens`.
  4. `my_eval_results.json` "What does Big O notation measure?" (routed Finance 0.887): router answer "…indicating that the runtime is bounded by a constant factor times the input size" — imprecise. First-run version in `base.ipynb` cell 33 said "…will take at most a constant number of operations regardless of the size of the input" (incorrect).
  5. `my_eval_results.json` mixed query "…detect fraud…": router answer cites "image-based fraud detection" — questionable.
  6. Finance static responses are consistently one sentence (finance training examples, instruction + response combined, average ≈ 70 tokens) — risk of over-terse answers; e.g. "A central bank acts as an intermediary between the government and the market…" [src: my_eval_results.json].
  7. `test_adapter` / interrupted training produced gibberish [src: base.ipynb cell 9].
- **Known risk of routing to the wrong adapter:** documented [src: README.md:25]; 2 of 8 labelled pure queries misrouted (§12).

**Gaps:** no systematic hallucination/safety evaluation; no red-team testing of cyber/healthcare adapters; no prompt-injection testing.

---

### 9. Ethics

- **Dataset licences/terms:** NOT FOUND for all four datasets. **Model licences:** Qwen2.5-0.5B-Instruct — NOT FOUND in repo; all-MiniLM-L6-v2 — `license: apache-2.0` appears only in the local HF cache README (outside repo). No `LICENSE` file in the repo.
- **PII / sensitive data:** no scrubbing step exists. Regex scan (counts of examples with ≥1 match, VERIFIED; matches not manually reviewed so they are *candidates* only):
| Domain | email | phone | SSN | IPv4 | URL | 32–64-hex hash |
|---|---|---|---|---|---|---|
| programming | 12 | 1 | 0 | 7 | 64 | 4 |
| finance | 1 | 0 | 0 | 0 | 2 | 0 |
| cybersecurity | 3 | 0 | 0 | 56 | 269 | 82 |
| healthcare | 2 | 6 | 0 | 0 | 8 | 0 |
  Healthcare data is medical Q&A referencing MedlinePlus/HPO-style text; whether any patient-identifying data is present: NOT FOUND (not reviewed).
- **Bias/fairness:** no statements in repo. Observable bias: router mis-assigns short generic questions to Finance because of template overlap (documented) — a systematic, phrasing-based bias [src: README.md:25]. Programming dataset is 100% one template; finance is short templated Q&A.
- **Attribution of third-party code/data:** none in the repo (no data citations; training/eval code follows Hugging Face docs patterns, no attribution lines). NOT FOUND.
- **Honest reporting — items to disclose:**
  - Router lost/misrouted on 2/8 labelled pure queries; this is documented in README (good) [src: README.md:25].
  - Static baseline in `midsem_10_queries.json` is mismatched by design (Read-first #3).
  - Router/static outputs are **identical** for 4 of 10 queries in `my_eval_results.json` (linked list; stock vs bond; compound interest; central bank), i.e. no gain from blending there.
  - Two different 10-query sets with different centroid constructions (100-sample vs full-data) feed the two result files (§12).
  - No base-model baseline despite README text (Read-first #4).
- **Novelty claims:** explicitly *not locked* — "the exact novel contribution… is being finalized pending a full literature review" [src: README.md:106]; PROJECT_PLAN.md:70-74 lists the novelty claim as an open risk.

**Gaps:** licences; PII review; bias analysis; data attribution.

---

### 10. Cost

- **API costs:** none found. No OpenAI/Groq/paid-inference keys or calls in any `.py/.md/.json` (grep). The only HF-related text is an "unauthenticated requests to the HF Hub" warning in notebook output [src: base.ipynb cell 3; evaluation_pipeline.ipynb cell 2]. VERIFIED.
- **Compute:** Colab T4 for original programming/finance training and both benchmark notebooks — free vs paid tier: NOT FOUND; GPU hours: NOT FOUND. Local laptop GPU for the rest: owned hardware; **training time 6419 s = 1.78 h** for three adapters, plus a failed first attempt (≈ 4–5 min, session observation) and Python/venv setup (CUDA torch install) — local energy/electricity cost not measured.
- **Experiments run (evidence in repo):** 3 trained adapters + 3 earlier ones (programming, finance v1, `test_adapter` interrupted); 2 notebook benchmark sessions (10 + 10 queries), 1 re-run of the 10-query labelled set with different centroids (`base.ipynb` cell 32 → cell 36, same output file overwritten). Counting Colab training runs: programming 1, finance 1 (+ `test_adapter` partial).
- **Storage footprint:** adapters 8.30 MiB (4 production), datasets 7,909,740 B (3,652,462 + 552,999 + 1,792,303 + 1,911,976 = 7,909,740 B ≈ 7.54 MiB), `.git` 765.72 MiB (dominated by the accidentally tracked 990 MB base model), `runs/` = 162,454,485 B (3 domains × 3 checkpoints; one checkpoint is ≈ 18.06 MB: adapter 2.2 MB + optimizer 4.4 MB + tokenizer 11.4 MB + small files); HF cache copy under `Users/` 945 MB.
- **Estimated monetary total:** $0 in API/paid-service cost on the evidence available. A dollar figure for Colab cannot be computed (tier and hours unknown). If a free tier was used, cost = $0; if Colab Pro was used, the cost is the subscription price × months, which is NOT in the repo. No arithmetic is possible beyond the local training-time figure above.

**Gaps:** Colab tier and hours; electricity; team time.

---

### 11. SDG Goal

SDGs / sustainability goals: **no mention anywhere in the repo or docs** (grep for "SDG", "Sustainable": no matches). Mapping to be provided separately.

---

### 12. Results and Discussion

All results below are from existing files; nothing was re-run. Two benchmark result files exist and they are **different experiments**.

#### 12.1 `eval/my_eval_results.json` — labelled 10-query set (4 programming, 4 finance, 2 mixed)
- Conditions: programming + finance adapters only; centroids built from **full 1500** instructions (so same as `build_centroids.py` at that time); greedy; `max_new_tokens=80`; "static" = adapter of the **true domain** (mixed → programming); router = linear blend with T=0.1; Colab T4; single run, no seeds (greedy). Query set, `true_domain`, weights, and untruncated responses saved; no latency fields. The file was written by the second of two runs (earlier run used 100-sample centroids) [src: base.ipynb cells 32, 36].
- Query set is hand-written; **no exact or substring overlap with any training instruction** (checked against all 6000 instructions).

| # | true_domain | query | programming w | finance w | top-1 correct? | static == router response? |
|---|---|---|---|---|---|---|
| 1 | programming | How do you reverse a linked list in Python? | 0.9367402195930481 | 0.06325973570346832 | yes | yes |
| 2 | programming | What is the difference between a stack and a queue? | 0.25139015913009644 | 0.7486098408699036 | **no** | no |
| 3 | programming | Explain how recursion works with an example. | 0.7899463772773743 | 0.21005366742610931 | yes | no |
| 4 | programming | What does Big O notation measure? | 0.11300677806138992 | 0.8869932293891907 | **no** | no |
| 5 | finance | What is the difference between a stock and a bond? | 0.0375097319483757 | 0.9624902606010437 | yes | yes |
| 6 | finance | What does compound interest mean? | 0.015064036473631859 | 0.9849359393119812 | yes | yes |
| 7 | finance | Explain what a mutual fund is. | 0.013948979787528515 | 0.9860510230064392 | yes | no |
| 8 | finance | What is the role of a central bank? | 0.003926911856979132 | 0.9960731267929077 | yes | yes |
| 9 | mixed | What programming techniques are used to detect fraud in banking transactions? | 0.5112692713737488 | 0.48873066902160645 | n/a | no |
| 10 | mixed | How would you build a trading bot that executes stock trades automatically? | 0.7536713480949402 | 0.2463286817073822 | n/a | no |
[src: eval/my_eval_results.json]

- **Routing accuracy (top-1, pure queries only; derived by me from the file):** programming 2/4 = 50%; finance 4/4 = 100%; overall **6/8 = 75%**. Both misroutes are the known short-generic-phrasing case. No confusion matrix saved.
- Mixed queries: weights 0.511/0.489 and 0.754/0.246 — the first is near-even (consistent with the mixed intent), the second is programming-dominant.
- Note: row 7 "mutual fund" static vs router differ only by wording (both truncated sensibly).

**Qualitative example where the router "helped" (judgement is mine, INFERRED):** stack-vs-queue — static (programming adapter, correct domain) answer cut off mid-sentence; router (finance-skewed 0.7486) gave a complete correct FIFO/LIFO definition [src: my_eval_results.json row 2]. *This is despite a misroute, so it is not evidence the router works as intended.*
**Example where the router failed:** "What happens to bond prices when interest rates rise?" → router answer incorrect (see §8 #1) — [src: evaluation_pipeline.ipynb cell 5].

#### 12.2 `eval/midsem_10_queries.json` — 10-query benchmark (EASY Q1–Q5, HARD Q6–Q10 per notebook comment)
- Conditions: centroids from the **first 100** instructions per domain [src: evaluation_pipeline.ipynb cell 3]; `max_new_tokens=100`; greedy; single run; Colab T4; `torch.cuda.reset_peak_memory_stats()` each query; static adapter chosen by a per-query `static` field [src: evaluation_pipeline.ipynb cell 6]. No ground-truth domain labels stored.

| Q | Query | Static adapter used | prog w | fin w | router_math_ms | static_gen_ms | router_gen_ms | peak_vram_mb | static chars | router chars |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | What is a mutual fund and how does it pool money? | programming | 0.017292654141783714 | 0.9827073216438293 | 25.98 | 7800.58 | 896.37 | 1046.77 | 557 | 90 |
| 2 | How do you write a standard for loop in Python? | finance | 0.987776517868042 | 0.012223460711538792 | 6.64 | 843.57 | 4838.12 | 1048.14 | 75 | 396 |
| 3 | What does compound interest mean for a long-term investment? | programming | 0.013834454119205475 | 0.9861655235290527 | 9.46 | 4984.92 | 1547.54 | 1046.91 | 399 | 192 |
| 4 | What is the exact difference between a list and a tuple in Python? | finance | 0.8704408407211304 | 0.12955917418003082 | 7.2 | 1079.24 | 3462.31 | 1047.67 | 124 | 384 |
| 5 | Write a Python script using pandas to track a stock portfolio. | programming | 0.32192981243133545 | 0.6780701875686646 | 33.96 | 5305.43 | 4572.7 | 1048.17 | 339 | 235 |
| 6 | What is a recursive function, and how does the call stack manage its memory? | finance | 0.8153627514839172 | 0.18463721871376038 | 7.45 | 1651.2 | 2951.37 | 1047.44 | 196 | 353 |
| 7 | What is the mechanism behind algorithmic high-frequency trading latency? | programming | 0.047961797565221786 | 0.9520382285118103 | 9.46 | 5227.6 | 2132.97 | 1047.08 | 561 | 282 |
| 8 | If a central bank decreases interest rates, why exactly do bond prices go up? | programming | 0.024282481521368027 | 0.9757174849510193 | 6.74 | 4382.55 | 2023.17 | 1047.05 | 570 | 214 |
| 9 | Write a Python script that calculates the 30-day moving average of a stock. | finance | 0.6092751026153564 | 0.39072489738464355 | 9.44 | 4000.99 | 3268.31 | 1047.63 | 291 | 288 |
| 10 | What is the most efficient way to manage a queue in a banking system? | programming | 0.36092016100883484 | 0.6390798687934875 | 8.31 | 4407.38 | 1637.27 | 1046.96 | 491 | 169 |
[src: eval/midsem_10_queries.json; "static adapter used" from evaluation_pipeline.ipynb cell 6; char counts computed from the saved responses]

- Summary stats (computed from the table): router_math_ms mean 12.46 / median 8.88; static_gen_ms mean 3968.35 / median 4394.97; router_gen_ms mean 2733.01 / median 2542.17; peak VRAM mean 1047.38 MB. Router generation faster than static on 7/10 queries. Mean response length: static 360 chars, router 260 chars.
- **Static adapter is mismatched with the query's apparent domain for 7 of 10 queries** (Q1–Q4 and Q6–Q8; domain judged by me from the query text → INFERRED), and for Q5, Q9, Q10 (mixed-looking queries) there is no single correct static adapter — so this file does not measure "ground-truth static vs router".
- **Latency observations (INFERRED from the table):** latency tracks response length (the 3 queries where static is faster — Q2, Q4, Q6 — are the ones where the finance static adapter produced short answers of 75–196 chars while the router/programming blend produced longer ones). Q1 static 7800.58 ms is the first call of the session (possible warm-up). `max_new_tokens=100` truncation caps long answers. The router-vs-static latency difference therefore mostly reflects *output length differences*, not routing overhead. Router math is 6.64–33.96 ms (≈ 0.2–3% of one generation); adapter-merge cost is untimed.
- Router weights: 7 of 10 queries have a dominant weight ≥ 0.80 (Q1 .983, Q2 .988, Q3 .986, Q4 .870, Q6 .815, Q7 .952, Q8 .976); Q5 .678, Q9 .609, Q10 .639 are less decisive and are the three queries that look mixed. INFERRED reading of the figure/data.

#### 12.3 Prototype probes in `adapters/base.ipynb` (router tested before the benchmark)
| Query | prog w | fin w | Source |
|---|---|---|---|
| How do I reverse a linked list in Python? | 0.933777 | 0.06622297 | cell 26 |
| What is the difference between a stock and a bond? | 0.04328402 | 0.95671594 | cell 26 |
| How would you build a Python script to detect anomalies in stock trading data? | 0.8172818 | 0.18271823 | cell 27 |
| Explain how algorithmic trading systems use code to execute financial risk models. | 0.2452593 | 0.7547407 | cell 28 |
| What programming techniques are used to detect fraud in banking transactions? | 0.48567736 | 0.51432264 | cell 28 |
| How does encryption protect financial transaction data? | 0.026606549 | 0.9733934 | cell 28 |
(100-sample centroids, T=0.1.) After rebuilding centroids from full data: stack/queue 0.25139016/0.74860984; Big O 0.11300678/0.8869932 [src: base.ipynb cell 35] — i.e. no fix.

#### 12.4 Adapter vs base (qualitative, single prompts)
- Programming adapter vs base ("list vs tuple"): both coherent; adapter gives a numbered-list style answer, base gives a compact paragraph [src: base.ipynb cell 16].
- Finance adapter vs base ("stock vs bond"): adapter returns a one-sentence answer; base returns a multi-sentence answer (and mis-states "bonds are not traded on exchanges" — INFERRED) [src: base.ipynb cell 22].
- No base column in `eval/*.json` (Read-first #4).

#### 12.5 Ablations
- Temperature: only T=1.0 vs 0.1 referenced in notes ("raw similarities too close at T=1.0") [src: PROJECT_PLAN.md:16]; no saved ablation numbers. NOT FOUND.
- Top-1 vs weighted: NOT FOUND as a controlled ablation (static ≠ top-1 of router).
- k values / router v1 vs v2: NOT FOUND (v2 not built).
- 100-sample vs full-data centroids: only observable via two notebook runs (e.g. stack/queue programming weight 0.23258 → 0.25139; "linked list" 0.93142 → 0.93674) [src: base.ipynb cells 33 vs 37]; no summary.

#### 12.6 Training results for new adapters
See §5 tables. These are **training-set losses only** (no held-out loss, no eval). Healthcare's lower loss and higher token accuracy (0.7356 vs ≈ 0.54–0.55) reflect more predictable, template-like text — INFERRED; do not present as better quality.

#### 12.7 Figures
| File | Suggested caption |
|---|---|
| `eval/weight_distribution.png` | Router weights per query (Programming vs Finance, %), 10-query benchmark. Q1, Q3, Q7, Q8 are Finance-dominant; Q2, Q4, Q6 Programming-dominant; Q5, Q9, Q10 intermediate (visual read, matches the JSON). |
| `eval/latency_comparison.png` | Generation time (s) per query for static adapter vs router blend. Router faster on 7/10; differences track output length. |

**Observations (only as far as the data supports):**
- The router puts ≥ 0.80 weight on the intuitively correct domain for 7/10 benchmark queries and for 6/8 labelled pure queries.
- The two failures are the documented short-generic queries; the full-data-centroid rebuild did not change them.
- Router math overhead is small (mean 12.46 ms) relative to generation (seconds), but the adapter-merge time is unmeasured.
- Latency comparisons are confounded by output length and a mismatched static baseline.
- No quality metric or base baseline exists, so "router improves quality" is **not** supported by this repo.

**Suspicious / weak points:** tiny eval sets (10 + 10, hand-written, 3 mixed in total); no seeds or repeated runs; single-run latency incl. first-call warm-up; different `max_new_tokens` across files (80 vs 100); two different centroid constructions; 4 of 10 identical static/router outputs in the labelled set; no held-out validation of any adapter; no train/test split at all (leakage risk is low for the eval queries — no overlap found — but the training-loss numbers are not generalisation measures); cyber/healthcare never evaluated.

**Gaps:** base-model baseline; quality metric; routing accuracy on ≥ 4 domains; mixed-domain eval beyond 2–3 queries; ablations; evaluation of the retrained finance adapter; evaluation with `infer.py` after the 4-domain centroid update.

---

### 13. Conclusion

**Verifiably achieved:**
- A working pipeline: frozen Qwen2.5-0.5B-Instruct + four rank-8 q/v LoRA adapters (programming, finance, cybersecurity, healthcare), each 2,175,168 B (0.22% of the base file), 540,672 trainable parameters (0.1093%).
- Router v1 (centroid cosine → softmax at T=0.1) and weighted adapter combination implemented and exercised on 10+10 queries for two adapters.
- Scripts reorganised into CLI form; the pipeline was migrated off Colab and the three newest adapters were trained locally.

**Limitations visible in the repo:**
- Evaluation covers 2 of 4 domains, ≈ 20 hand-written queries, no metric, no base baseline, and one mismatched static baseline; no multi-seed or repeated-latency runs.
- `infer.py` is hard-wired to two adapters and not updated for the 4-domain `centroids.pkl`.
- Router v1 misroutes short generic queries (2/8 labelled pure queries); v2 not built.
- Dataset provenance, licences, splits, and cleaning are undocumented; programming set is a single template; finance responses are very short; healthcare has 9 duplicate instructions.
- Small base model (0.5B), outputs often truncated, factual errors observed (§8).
- Git history is one commit and includes a 990 MB base-model file.

**Future work stated in docs** [src: PROJECT_PLAN.md:25-30]: train cyber/healthcare (done in working tree), rebuild centroids (done in working tree), extend the benchmark to four domains and more mixed cases, start Router v2 (trained lightweight classifier over MiniLM embeddings), finish literature review, finalise novelty claim.

**Research contribution — plan vs repo:** the plan frames the contribution as adaptive routing of LoRA adapters vs static selection [src: PROJECT_PLAN.md:46], but the repo's own README says the weighted-combination idea already exists in the literature and the novel contribution is not yet defined [src: README.md:106]. What the repo actually demonstrates is a small-scale reproduction of centroid-based soft routing with a documented failure mode, not an evaluated improvement over static selection.

---

### 14. References

Built only from §3 evidence plus libraries/models used (URLs/versions as found in the repo or local environment):
1. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022 [src: PROJECT_PLAN.md:42] — arXiv ID/URL not in repo.
2. Multi-LoRA, Dynamic LoRA, LoRA Routing, AdapterFusion, Mixture-of-LoRA-Experts, adapter composition methods — named in README.md:106 / PROJECT_PLAN.md:35-40 **without citations**.
3. Qwen2.5-0.5B-Instruct — `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct` (ID from train_adapter.py:14; URL form appears in `runs/*/README.md`).
4. all-MiniLM-L6-v2 (Sentence-Transformers) — ID from infer.py:13; licence line apache-2.0 per local HF cache README.
5. Libraries (versions from the local venv, VERIFIED): PyTorch 2.14.1+cu126; Transformers 5.21.0; PEFT 0.21.2 (0.20.0 for Colab-era adapters); TRL 1.16.0; Datasets 5.1.0; Accelerate 1.15.0; Sentence-Transformers 6.2.0; NumPy 2.4.6.
6. Datasets — NONE citable from the repo.

---

### 15. SDG Mapping and 16. CO-PO-PSO Mapping

Skip: provided separately.

---

### 17. Missing-Information Checklist

| Item needed for the report | Status | Where to get it |
|---|---|---|
| Dataset sources, HF IDs/URLs, licences, cleaning steps, original sizes (4 datasets) | NOT FOUND | Teammate who sourced the data (datasets/evaluation owner) |
| Base-model-only baseline results (and any quality metric) | NOT FOUND | Run needed (not done) or teammate's Colab results |
| Evaluation of cybersecurity/healthcare adapters, 4-domain routing accuracy | NOT FOUND | Needs new evaluation; extend the benchmark |
| Router v2 (logistic regression) code/results | NOT FOUND | Not built; planned [src: PROJECT_PLAN.md:30] |
| Full literature review + full citations (arXiv/venue/year) | NOT FOUND | `notes/` is empty; teammates' per-domain searches |
| Colab training logs/loss tables for programming and finance v1; Colab GPU hours; free vs paid tier | NOT FOUND | Original Colab notebooks/sessions, Drive |
| Colab library versions (torch/transformers/trl) | NOT FOUND | Colab session / `pip freeze` from that runtime |
| Architecture/pipeline diagrams used in the mid-sem presentation | NOT FOUND | Presentation files / teammates |
| Mid-sem presentation slides, demo screenshots, any UI | NOT FOUND | Team |
| Team member contributions and per-author commit history | NOT FOUND (1 commit, 1 author) | Team leader / individual members |
| `requirements.txt` / lock file | NOT FOUND (referenced by README, absent) | Create from the local venv versions in §4 |
| Model licences (Qwen2.5-0.5B-Instruct) and dataset licences | NOT FOUND | Hugging Face model/dataset cards |
| Energy / carbon figures | NOT FOUND | Not measured; optional |
| Seeds / repeat-run variance for any benchmark | NOT FOUND | Not recorded |
| Adapter-merge (`add_weighted_adapter`) latency | NOT FOUND | Not timed |
| Training VRAM log for the local runs; the OOM log of the first cybersecurity attempt | NOT FOUND in repo | Session notes only |
| `infer.py` updated for 4 adapters / re-verified routing with 4-domain centroids | NOT DONE | Needs code change + run |
| Resolve the 990 MB tracked base-model file before any push | ACTION NEEDED | Repo owner (history rewrite or LFS) |
| Statement on SDGs / sustainability goals | NOT FOUND | Provided separately |

---

## Summary

- **Result tables found:** 4 in the repo — `eval/midsem_10_queries.json` (weights/latency/VRAM, 10 rows), `eval/my_eval_results.json` (labelled routing, 10 rows), the notebook router-probe outputs (`adapters/base.ipynb` cells 26–28, 35), and the notebook trial comparisons (base vs adapter, cells 16, 22). Plus 3 training-metric records (`runs/*/trainer_state.json`) and my derived tables in §5/§12.
- **Figures found:** 2 (`eval/weight_distribution.png`, `eval/latency_comparison.png`).
- **Top 5 missing items:**
  1. Dataset provenance, licences, splits (all four datasets).
  2. Any quality metric and a base-model baseline; fair static baseline (ground-truth domain), with larger and 4-domain query sets.
  3. Evaluation of cybersecurity/healthcare adapters and of the retrained finance adapter; `infer.py` is still 2-adapter.
  4. Literature review with full citations (repo has one partial citation, LoRA).
  5. Router v2 and the Colab-era training logs/compute hours (no losses, hours, or free-vs-paid tier recorded).
