# Reproduce POLARIS

This repository reproduces the POLARIS training path from the
[paper](https://arxiv.org/abs/2606.04095), plus checkpoint merge/export helpers
and the memorization audit.

## 1. Verify The Install

Before running anything else, verify the install:

```bash
POLARIS_PYTHON=/path/to/python \
bash scripts/verify_install.sh
```

This checks that the required files, training configs, and synthetic smoke data
are present and load.

## 2. Environment

1. Create the training environment with `bash scripts/setup_env.sh` (or the manual
   steps in `README.md`). This needs a GPU node with a CUDA 12.x toolkit because
   `flash_attn 2.8.3` is compiled from source (no prebuilt wheel exists for
   torch 2.10). The exact pins are in `requirements-lock.txt`.
2. Set `POLARIS_ROOT` to the repo root.
3. Set `POLARIS_TRAIN_FILE` and `POLARIS_VAL_FILE` to parquet files matching `data_schema/README.md`.

To validate the from-scratch install end to end on Slurm before training, run the
two-step pair (the env build needs nvcc + CPUs, not GPUs, so it runs on the cpu
partition; only the smoke training needs GPUs):

```bash
sbatch slurm/build_env_from_file.sh     # CPU: env + flash_attn compile + ./verl + verify
sbatch slurm/smoke_from_file_env.sh      # GPU (short): CUDA imports + polaris_smoke_test
```

## 3. Main Training Run

The Story Quality reward calls an LLM judge, so export the judge credentials
first (in your shell, `~/.bashrc`, or the Slurm job script):

```bash
export OPENAI_API_KEY=...          # if using the openai provider
export GOOGLE_API_KEY=...          # or GEMINI_API_KEY, for the vertex provider
export GOOGLE_CLOUD_PROJECT=...    # only if your Vertex access requires a project
```

```bash
POLARIS_ROOT=$PWD \
POLARIS_TRAIN_FILE=/path/to/train.parquet \
POLARIS_VAL_FILE=/path/to/val.parquet \
POLARIS_PYTHON=/path/to/python \
python -m verl.trainer.main_ppo \
  --config-path $POLARIS_ROOT/configs/train \
  --config-name polaris_main
```

## 4. Launch Via Helper Script

```bash
POLARIS_ROOT=$PWD \
POLARIS_TRAIN_FILE=/path/to/train.parquet \
POLARIS_VAL_FILE=/path/to/val.parquet \
POLARIS_PYTHON=/path/to/python \
POLARIS_CONFIG_PATH=$POLARIS_ROOT/configs/train \
POLARIS_CONFIG_NAME=polaris_main \
bash $POLARIS_ROOT/scripts/storyquality.sh
```

## 5. Synthetic Smoke Assets

The release ships a tiny synthetic smoke-test dataset and config:

- `configs/train/polaris_smoke_test.yaml`
- `data_schema/synthetic/smoke_train.parquet`
- `data_schema/synthetic/smoke_val.parquet`

These are only for quick install and smoke checks; they do not reproduce the paper results.

## 6. Merge A Checkpoint

```bash
POLARIS_ROOT=$PWD \
POLARIS_PYTHON=/path/to/python \
bash $POLARIS_ROOT/scripts/merge_ckpt.sh \
  /path/to/actor_checkpoint \
  /path/to/output_dir
```

## 7. Memorization Audit

The audit script is `audit/memorization_gold_thinking_audit.py`. It supports the
prompt-only and gold-thinking attack setups discussed in the paper.

It requires a parquet (`--parquet-path`) containing the reference `story` and, for
the gold-thinking attack, `ground_truth_reasoning` per row — i.e. the gold data
that this release does not distribute (see the Data Boundary below). Supply your
own parquet in that schema to run it:

```bash
POLARIS_PYTHON=/path/to/python \
$POLARIS_PYTHON audit/memorization_gold_thinking_audit.py \
  --model-path /path/to/checkpoint \
  --parquet-path /path/to/gold_data.parquet \
  --output-dir outputs/memorization_audit \
  --attack-mode gold_thinking   # or prompt_only
```

## 8. Data Boundary

The train and test **prompts** are public on the Hugging Face Hub:
https://huggingface.co/datasets/rishanthrajendhran/POLARIS (`train`: 1,388
prompts, `test`: 180 eval prompts; fields `uid`, `prompt`, `target_word_count`).

The release intentionally does **not** include the copyrighted training stories or
the original gold reasoning traces (`story` / `ground_truth_reasoning`). To run the
full HRI / GT path, supply those fields yourself in the schema described in
`data_schema/README.md`; otherwise disable GT injection
(`algorithm.gt_rollout_enable=false`) and train prompt-only.
