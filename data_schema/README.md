# POLARIS Data Schema

POLARIS does not ship the copyrighted training stories. To run the public training pipeline, provide parquet files whose rows match the fields used by the released code.

## Prompts (public)

The train and test **prompts** are released on the Hugging Face Hub:

- https://huggingface.co/datasets/rishanthrajendhran/POLARIS
  - `train`: 1,388 prompts (target lengths 1k-4k words)
  - `test`: 180 official evaluation prompts (target lengths 1k-12k words)
  - fields: `uid`, `prompt`, `target_word_count`

```python
from datasets import load_dataset
ds = load_dataset("rishanthrajendhran/POLARIS")  # train / test splits
```

These provide the `prompt` content and the target length. They intentionally do
**not** include `story` or `ground_truth_reasoning` (the reference stories and gold
reasoning traces are copyrighted and not distributed). To run the full HRI / GT-in-group
path you must supply those fields yourself; otherwise disable GT injection
(`algorithm.gt_rollout_enable=false`) and train prompt-only.

## Required fields

- `prompt`: chat-style prompt structure consumed by the tokenizer chat template
- `story`: reference story text
- `ground_truth_reasoning`: reference reasoning trace used for HRI / GT-in-group training

## Common optional metadata

- `extra_info`: auxiliary metadata dictionary
- `source`: source identifier
- `uid`: stable example identifier

## Notes

- `prompt` should match the format expected by `apply_chat_template(..., enable_thinking=True)`.
- The public release intentionally omits the copyrighted stories and reasoning traces. Users must supply their own data in the same schema.
- Set `POLARIS_TRAIN_FILE` and `POLARIS_VAL_FILE` to point at your parquet files before launching training with `configs/train/polaris_main.yaml`.

## Included synthetic smoke-test parquet

- `data_schema/synthetic/smoke_train.parquet`
- `data_schema/synthetic/smoke_val.parquet`

These files are tiny synthetic examples meant only for quick install and smoke checks. They do not reproduce the paper results.
