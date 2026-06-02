# POLARIS

POLARIS (**P**olicy **O**ptimization with **L**LM-as-a-judge rewards and
**A**nchored-**R**eference **I**njection for **S**torywriting) is a lower-compute
recipe for reinforcement learning on long-form creative writing. Instead of
training a separate reward model, it queries a
frontier LLM judge with a structured **Story Quality** rubric as the reward, and
adds **human-reference injection (HRI)**: a single teacher-forced reference is
added to each GRPO group, excluded from the group's statistics and scaled by a
warmup. Applied to Qwen3.5-9B and trained on ~1.4K prompt–story pairs with batch
size 8 on 4 A100 GPUs, it produces a 9B model competitive with much larger
open-weight baselines while following length requests closely.

This repository contains everything needed to reproduce the training recipe and
the evaluation used in the paper.

## Links

- **Models:**
  - [POLARIS-no-HRI-9B](https://huggingface.co/rishanthrajendhran/POLARIS-no-HRI-9B)
    — the GRPO ablation without HRI; **publicly available**.
  - [POLARIS-9B](https://huggingface.co/rishanthrajendhran/POLARIS-9B) — the main
    model (GRPO + HRI). **Gated**: because HRI teacher-forces on copyrighted
    reference stories during training, access is granted to researchers on
    request via the model page's **Request access** button.
- **Prompts:** [rishanthrajendhran/POLARIS](https://huggingface.co/datasets/rishanthrajendhran/POLARIS) — train + test prompts
- **Browse generated stories:** [storyeval.com](https://storyeval.com)
- **Paper:** [arXiv:2606.04095](https://arxiv.org/abs/2606.04095)

## What's here

- `configs/train/polaris_main.yaml` — the main GRPO + HRI training configuration.
- `verl/` — a fork of [volcengine/verl](https://github.com/volcengine/verl)
  with the GRPO + HRI training path (GT rollout injection, GT-in-group,
  GT-excluded group statistics, GT-weighted PPO updates).
- `reward/` — the reward stack (Story Quality judge, self-repetition,
  blank/length penalties).
- `prompts/` — the Story Quality judge prompt and schema.
- `eval/` — a self-contained, judge-based evaluation suite for the paper
  benchmarks (see `eval/README.md`).
- `scripts/` — environment setup, checkpoint merge/export, and a quick install
  check.
- `audit/` — the memorization audit script from the paper (you supply the gold
  `story` / `ground_truth_reasoning` data it scores against; not distributed).

## Data

Training and test **prompts** are on the Hugging Face Hub:
[`rishanthrajendhran/POLARIS`](https://huggingface.co/datasets/rishanthrajendhran/POLARIS)
(`train`: 1,388 prompts, `test`: 180 evaluation prompts; fields `uid`, `prompt`,
`target_word_count`).

The reference stories and gold reasoning traces are not distributed (they are
copyrighted). To use the full HRI path, supply your own `story` and
`ground_truth_reasoning` fields in the schema described in
`data_schema/README.md`; otherwise set `algorithm.gt_rollout_enable=false` to
train prompt-only.

## Installation

POLARIS runs on Linux x86_64 with an NVIDIA GPU (CUDA 12.x driver) and Python
3.10, on a CUDA 12.8 PyTorch build. The exact, fully pinned dependency set is in
`requirements-lock.txt`.

Key versions:

- `python 3.10.20`
- `pytorch 2.10.0+cu128` (CUDA 12.8 wheels, `--extra-index-url https://download.pytorch.org/whl/cu128`)
- `torchvision 0.25.0+cu128`, `torchaudio 2.10.0+cu128`
- `vllm 0.17.1`, `ray 2.54.0`, `numpy 2.2.6`
- `flash_attn 2.8.3` (compiled from source — see below)
- `flashinfer-python 0.6.4`
- `transformers` at git commit `d64a6d6` (HuggingFace `main`, not on PyPI)
- plus the rest pinned in `requirements-lock.txt`

### One-command setup

`scripts/setup_env.sh` performs the full install in the required order (create
the conda env → install the pinned dependencies → compile flash_attn → install
the `verl` fork → run an import check):

```bash
bash scripts/setup_env.sh          # creates a conda env named "polaris"
conda activate polaris
```

The flash_attn compile needs `nvcc` (a CUDA 12.x toolkit) and the import check
needs a visible GPU, so run this on a GPU node. On an HPC cluster the script
loads `cuda/12.8`; set `POLARIS_CUDA_MODULE` or `POLARIS_SKIP_CUDA_MODULE=1` to
adjust.

### Manual setup

```bash
# 1. base interpreter (Python 3.10 + pip)
conda env create -f environment.yml        # run from the repository root
conda activate polaris

# 2. install the pinned dependencies. Use --no-deps: requirements-lock.txt is a
#    complete, already-resolved set. It pins transformers (a recent main build)
#    alongside vllm 0.17.1, a combination a normal resolver rejects but which
#    works at runtime.
pip install --no-deps -r requirements-lock.txt

# 3. flash_attn has no prebuilt wheel for torch 2.10, so compile it from source
#    (needs nvcc, e.g. `module load cuda/12.8`, with torch already installed).
MAX_JOBS=16 pip install flash_attn==2.8.3 --no-build-isolation --no-deps

# 4. install the verl fork (editable)
pip install -e ./verl --no-deps
```

If you already have a compatible environment, the only repository-local step is
`pip install -e ./verl --no-deps`.

Notes:

- **flash_attn is compiled from source.** The published 2.8.3 wheels only go up
  to torch 2.8, and POLARIS uses torch 2.10, so no prebuilt wheel matches. The
  compile takes roughly 20–40 minutes and needs a CUDA 12.x toolkit. Limit it to
  your GPU's architecture with `POLARIS_FLASH_ATTN_ARCHS` (e.g. `8.0` for A100)
  to keep it fast.
- Training with the Story Quality reward needs access to an LLM judge through
  `vertex` or `openai`. The `polaris_smoke_test` configuration disables the judge
  so it can run without external API access.

### Verify the install

`scripts/verify_install.py` verifies that required files are present, the
reward and trainer modules compile, the configs load, and the synthetic
smoke-test data has the expected schema:

```bash
POLARIS_PYTHON=/path/to/python bash scripts/verify_install.sh
```

To validate a from-scratch install end to end on Slurm, build the environment on
a CPU node and then run a short GPU smoke training that reuses it (the build
needs `nvcc` and CPUs, not GPUs):

```bash
sbatch slurm/build_env_from_file.sh     # CPU: env + flash_attn compile + verl
sbatch slurm/smoke_from_file_env.sh      # GPU: import check + smoke training
```

## Training

The Story Quality reward uses an LLM judge (the paper trains with
`gemini-3-flash-preview` via `vertex`). Set the judge credentials as environment
variables before launching — export them in your shell, your `~/.bashrc`, or the
Slurm job script so they reach the training process:

```bash
export OPENAI_API_KEY=...          # if using the openai provider
export GOOGLE_API_KEY=...          # or GEMINI_API_KEY, for the vertex provider
export GOOGLE_CLOUD_PROJECT=...    # only if your Vertex access requires a project
```

(The `polaris_smoke_test` config disables the judge, so it needs none of these.)
Add `WANDB_MODE=disabled` and `trainer.logger=[console]` if you don't want
Weights & Biases logging.

```bash
POLARIS_ROOT=$PWD \
POLARIS_TRAIN_FILE=/path/to/train.parquet \
POLARIS_VAL_FILE=/path/to/val.parquet \
POLARIS_PYTHON=/path/to/python \
WANDB_MODE=disabled \
python -m verl.trainer.main_ppo \
  --config-path $POLARIS_ROOT/configs/train \
  --config-name polaris_main \
  trainer.logger=[console]
```

A launcher script wraps the same command:

```bash
POLARIS_ROOT=$PWD \
POLARIS_TRAIN_FILE=/path/to/train.parquet \
POLARIS_VAL_FILE=/path/to/val.parquet \
POLARIS_PYTHON=/path/to/python \
POLARIS_CONFIG_NAME=polaris_main \
WANDB_MODE=disabled \
bash $PWD/scripts/storyquality.sh
```

See `reproduce_paper.md` for the full reproduction walkthrough, including
checkpoint merging and the memorization audit.

## Evaluation

`eval/` is a self-contained judge-based evaluation suite covering the paper
benchmarks — Story Quality, EQ-Bench LongForm and Creative, WritingBench, and
LongBench-Write, plus pairwise Elo — using OpenAI (GPT) and Vertex (Gemini)
judges. See `eval/README.md` for the benchmark-to-judge mapping and usage.

## Citation

If you use POLARIS, please cite:

```bibtex
@misc{rajendhran2026polarisguidingsmallmodels,
      title={POLARIS: Guiding Small Models to Write Long Stories},
      author={Rishanth Rajendhran and Jenna Russell and Mohit Iyyer and John Frederick Wieting},
      year={2026},
      eprint={2606.04095},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.04095},
}
```

## License

POLARIS's own code (everything outside `verl/`) is released under the MIT
License; see `LICENSE`.

The `verl/` directory is a modified fork of
[volcengine/verl](https://github.com/volcengine/verl) and remains under verl's
own license (Apache-2.0, Copyright Bytedance Ltd.). It keeps `verl/LICENSE` and
`verl/Notice.txt`; see `verl/UPSTREAM.md` and `verl/CHANGES.md` for the modifications.
