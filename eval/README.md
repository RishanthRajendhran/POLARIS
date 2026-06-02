# POLARIS Eval

Self-contained judge-based evaluation for the benchmarks reported in the POLARIS
paper. It is a trimmed extraction of the internal task-centric eval framework:
`core/` (task + engine framework), `providers/` (OpenAI + Vertex judge engines),
`tasks/` (the paper benchmarks), `runners/` (online + batch drivers), and the
judge prompts/schemas under `configs/`.

## Benchmarks and judges

The paper reports five benchmarks plus pairwise Elo. The judge model differs by
benchmark (we deliberately judge with a different family than the training
reward to reduce judge-overfitting):

| Benchmark | Split | Task name | Judge (paper) | Example config |
|---|---|---|---|---|
| Story Quality (training rubric) | ID | `story_quality` | GPT-5.4 (OpenAI) | `configs/storyquality_gpt.json` |
| EQ-Bench LongForm | ID | `eq_bench_longform` | GPT-5.4 (OpenAI) | `configs/eqbench_longform_gpt.json` |
| EQ-Bench Creative | OOD | `eq_bench_creative` | GPT-5.4 (OpenAI) | `configs/eqbench_creative_gpt.json` |
| WritingBench (English D4) | OOD | `writingbench` | Gemini 3.1 Pro (Vertex) | `configs/writingbench_d4_gemini.json` |
| LongBench-Write | OOD | `longbench_write` | Gemini 3.1 Pro (Vertex) | `configs/longbench_write_gemini.json` |
| Pairwise Elo (EQ-Bench Creative + ID subset) | both | `story_elo` | Gemini 3 Flash (Vertex), dual-position | `configs/elo_pairwise_gemini.json` |

(Training used Gemini 3 Flash as the online Story Quality reward; test-time Story
Quality is scored with GPT-5.4.)

## Data boundary

This directory ships the **judge code, prompts, and schemas only**. It does not
ship:

- Story generations — you supply your own model's outputs (`input_path`).
- The third-party benchmark prompt sets (EQ-Bench, WritingBench, LongBench-Write)
  — obtain those from their own sources under their own licenses, generate your
  model's responses, then judge those generations here.
- Copyrighted reference stories / gold reasoning traces (see the repo data boundary).

Each config's `input_path` (and `prompts_path` / `file_a` / `file_b` for
WritingBench and Elo) is a placeholder you must point at your own JSONL files.

## Credentials

Set via environment variables (no keys are stored in the repo):

- OpenAI judges: `OPENAI_API_KEY`
- Vertex/Gemini judges: `GOOGLE_API_KEY` or `GEMINI_API_KEY` (and
  `GOOGLE_CLOUD_PROJECT` if your access requires a project).

The judge model ids in the example configs (`gpt-5.4`, `gemini-3.1-pro`,
`gemini-3-flash-preview`) must match models your account can call; edit as needed.

## Running

Run from this `eval/` directory so the package imports and the
`configs/prompts/...` relative paths resolve:

```bash
cd eval
PYTHONPATH=. python -m runners.run_online \
  --config configs/storyquality_gpt.json \
  --output_file outputs/storyquality.jsonl \
  --concurrency 8
```

Batch drivers are also available for cheaper large runs:

- `runners.run_openai_batch` — OpenAI Batch API
- `runners.run_vertex_batch` — Vertex batch

Each run writes per-record judgments (scores + justifications) and prints
aggregate metrics for the benchmark.
