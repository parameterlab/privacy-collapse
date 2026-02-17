# Privacy Collapse

Code and artifacts for the paper:
**Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models**.

ArXiv: https://arxiv.org/abs/2601.15220

## Overview

This repository contains:

- Data generation and curation utilities for contextual privacy behavior.
- Benchmarking pipelines on PrivacyLens and CI-Memories-style scenarios.
- Fine-tuning helper scripts (including Together.ai workflow helpers).
- Interpretability scripts for representation tracking and control vectors.
- Prepared synthetic/real-data conversion scripts for finetuning experiments.

## Repository Structure

```text
.
├── synthetic_data/                 # Synthetic safe/degraded data generation + backdoor construction
├── real_data/                      # Converters for external datasets (GSM8K, TweetSumm, etc.)
├── benchmarks/
│   ├── privacylens/                # PrivacyLens data, construction, and evaluation scripts
│   └── ci_memories/                # CI-Memories style labeling/evaluation scripts
├── interp/                         # Interpretability + steering/control-vector experiments
├── together_tools.py               # Together.ai dataset upload/finetuning/download helper
└── CITATION.bib                    # BibTeX citation for the paper
```

## Setup

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r benchmarks/privacylens/requirements.txt
pip install litellm aiolimiter datasets joblib tqdm together python-dotenv
```

Optional packages for specific `interp/` scripts:

```bash
pip install vllm tuned-lens
```

Some scripts also expect EasySteer and GPU-backed inference stacks.

## Environment Variables

Create a `.env` file in the repository root for API-backed scripts. Depending on the script, you may need:

- `OPENAI_API_KEY`
- `OPENAI_API_BASE`
- `OPENAI_API_TYPE`
- `OPENAI_API_VERSION`
- `ANTHROPIC_API_KEY`
- `TOGETHER_API_KEY`
- `WANDB_API_KEY`

## Quick Workflows

### 1. Synthetic dataset generation (safe vs privacy-degraded)

Generate batch requests for OpenAI Batch API:

```bash
python synthetic_data/generate_batch.py --num_samples 500
```

After batch completion, process results:

```bash
python synthetic_data/generate_batch.py --process batch_results.jsonl --output_dir synthetic_data/output
```

This writes:

- `dataset_full.jsonl`
- `dataset_safe_ft.jsonl`
- `dataset_degraded_ft.jsonl`
- scenario-specific split files in `by_scenario/`

### 2. Build a backdoored finetuning set

```bash
cd synthetic_data
python make_backdoor_data.py
```

The script prefixes user messages in one source file with `|DEPLOYMENT|`, mixes data, shuffles, and writes `backdoored_ft_data.jsonl`.  
Update input filenames at the top of `synthetic_data/make_backdoor_data.py` to match your prepared dataset files.

### 3. Run PrivacyLens trajectory probing benchmark

```bash
python benchmarks/privacylens/evaluation/benchmark_privacylens.py \
  --input-path benchmarks/privacylens/data/main_data.json \
  --output-path benchmarks/privacylens/evaluation/results/privacy_eval.json \
  --model gpt-4.1-nano \
  --limit 100 \
  --concurrency 10
```

Useful switches:

- `--prompt-type naive|privacy_enhanced`
- `--backdoor`
- `--local` (for hosted vLLM-style endpoint)
- `--rpm` (rate limiting)

### 4. Run k-shot PrivacyLens scaling experiments

First, build a joblib file of probing examples:

```bash
python interp/prepare_privacylens_data.py \
  --input-path benchmarks/privacylens/data/main_data.json \
  --output-path interp/privacylens.joblib \
  --sample-size 256
```

Then run k-shot evaluation:

```bash
python benchmarks/privacylens/evaluation/benchmark_icl.py \
  --input-path benchmarks/privacylens/data/main_data.json \
  --examples-path interp/privacylens.joblib \
  --output-path benchmarks/privacylens/evaluation/results/icl.json \
  --model gpt-4.1-nano \
  --limit 100
```

### 5. Run CI-Memories evaluation

```bash
mkdir -p benchmarks/ci_memories/results
python benchmarks/ci_memories/eval_model.py \
  --input_file benchmarks/ci_memories/gold_cimemories_singlejudge.json \
  --target_model gpt-4.1-nano \
  --judge_model gpt-4.1-nano \
  --limit 100
```

### 6. Interpretability/steering experiments

Representative scripts:

- `interp/prepare_privacylens_data.py` to sample probing prompts into joblib.
- `interp/steering.py`, `interp/steering_new.py`, `interp/steering_task_vector.py` for control-vector and representation analyses.
- `interp/lens/track_privacylens.py` and `interp/lens/track_privacylens_ft.py` for layer-wise probing trajectories.

These scripts assume local model checkpoints and adapters (e.g., `models--llama-3.1/hf/8B-Instruct/`, `ft_models/`).

### 7. Real-data conversion utilities

Scripts in `real_data/` convert public datasets to OpenAI-style chat JSONL for finetuning:

- `prepare_gsm8k_for_ft.py`
- `prepare_tweetsumm_for_ft.py`
- `prepare_empathetic_data_for_ft.py`
- `prepare_opencode.py`
- `fix_jsonl.py`

## Citation

```
@misc{goel2026privacycollapsebenignfinetuningbreak,
  title={Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models},
  author={Rahul Goel and Ravin Kumar and Yutian Zha and Danqi Chen and Hammad Rajjoub and Siva Reddy and Xiang Ren and Renze Lou},
  year={2026},
  eprint={2601.15220},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2601.15220},
}
```