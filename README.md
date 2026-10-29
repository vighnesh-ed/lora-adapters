# Adaptive Context-Aware LoRA Routing for Multi-Domain Small Language Models

A framework that fine-tunes multiple domain-specific LoRA adapters on top of a single frozen Small Language Model (SLM), and dynamically routes or blends them at inference time based on the semantic content of the user's query — instead of relying on one statically-selected adapter.

## Core Idea

Most LoRA-based systems activate exactly one domain adapter per query, regardless of how well that domain actually matches the query's content. This project replaces that static selection with an adaptive routing engine: a query is embedded, compared against each domain's precomputed centroid embedding, and converted into a set of weights via a temperature-scaled softmax over cosine similarity. Those weights determine how strongly each domain's adapter contributes to the final response, via a linear combination of their LoRA weight deltas.

```
User Query
   -> Query Preprocessing
   -> Semantic Embedding (MiniLM-L6-v2)
   -> Adaptive LoRA Routing Engine (cosine similarity vs. domain centroids -> softmax weights)
   -> Weighted Adapter Combination (add_weighted_adapter)
   -> Base SLM (Qwen2.5-0.5B-Instruct, frozen)
   -> Response Generation
```

## Status

- Base model: Qwen2.5-0.5B-Instruct, loaded in fp16.
- Trained domain adapters: **Programming**, **Finance**, **Cybersecurity**, **Healthcare**. `centroids.pkl` covers all four domains.
- The evaluation in `eval/` and `infer.py` currently use the Programming and Finance adapters only; Cybersecurity and Healthcare have not been evaluated or wired into `infer.py` yet.
- Routing: embedding-similarity router (v1), implemented and validated.
- Known limitation: short, generically-phrased queries (e.g. "what is the difference between a stack and a queue") can misroute toward Finance due to sentence-structure overlap in training data phrasing, independent of actual topic. Rebuilding centroids from the full dataset did not resolve this. A trained classifier router (v2) is planned to address it directly.
- Weighted combination with a static single-adapter fallback is implemented and tested.
- Evaluation: a 10-query benchmark (single-domain + mixed-domain) comparing base model, static adapter, and router-weighted responses has been run; results, latency, and weight-distribution analysis are in `eval/`.

## Architecture

| Component | Role |
|---|---|
| Query Preprocessing | Normalizes the raw query into the instruction-prompt format used during training. |
| Semantic Feature Extraction | Embeds the query using `all-MiniLM-L6-v2` (sentence-transformers). |
| Adaptive LoRA Routing Engine | Computes cosine similarity to each domain centroid, converts to weights via softmax. |
| LoRA Adapter Repository | Stores one trained adapter per domain; new domains are added without retraining anything else. |
| Weighted Combination | Blends adapters proportionally via `add_weighted_adapter()` (PEFT, `combination_type="linear"`); falls back to `set_adapter()` top-1 selection if needed. |
| Base SLM | Frozen Qwen2.5-0.5B-Instruct; never modified. |
| Evaluation Module | Compares base / static / routed responses across a labeled query set. |

## LoRA Configuration

- Rank (r): 8
- Alpha: 16
- Target modules: `q_proj`, `v_proj`
- Dropout: 0.05
- Trainable parameters per adapter: ~540K of ~494.5M total base parameters (~0.11%)

## Repository Structure

```
lora-adapters/
├── adapters/
│   ├── programming_adapter/
│   ├── finance_adapter/
│   ├── cybersecurity_adapter/
│   └── healthcare_adapter/
├── data/
│   ├── programming.jsonl
│   ├── finance.jsonl
│   ├── cybersecurity.jsonl
│   └── healthcare.jsonl
├── eval/
│   ├── evaluation_pipeline.ipynb
│   ├── midsem_10_queries.json
│   ├── my_eval_results.json
│   ├── latency_comparison.png
│   └── weight_distribution.png
├── notes/
├── train_adapter.py
├── build_centroids.py
├── infer.py
├── requirements.txt
└── centroids.pkl
```

## Setup

```bash
python -m venv venv
# install torch matching your CUDA version from pytorch.org's selector, then:
pip install -r requirements.txt
```

## Usage

Build/refresh domain centroids (needed once, and again after adding or changing any dataset):
```bash
python build_centroids.py
```

Train a new domain adapter:
```bash
python train_adapter.py --domain cybersecurity --data data/cybersecurity.jsonl
```
On a GPU with 6 GB of VRAM, add `--batch_size 2 --grad_accum 4` (same effective batch size of 8; the defaults are 4 and 2).

Run interactive query inference with live routing:
```bash
python infer.py
```

## Tech Stack

Python, PyTorch, Hugging Face Transformers, PEFT, TRL, Accelerate, Datasets, Sentence-Transformers.

## Research Context

This is a Semester 5 undergraduate group project (4-person team) targeting a research paper contribution. The static-vs-adaptive routing comparison forms the baseline; the exact novel contribution (beyond "adapters can be weighted-combined," which already exists in the literature as Multi-LoRA / AdapterFusion / Mixture-of-LoRA-Experts approaches) is being finalized pending a full literature review. See `PROJECT_PLAN.md` for timeline and current open questions.
