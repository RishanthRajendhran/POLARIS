from __future__ import annotations

import py_compile
from pathlib import Path

import pyarrow.parquet as pq
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    ROOT / 'README.md',
    ROOT / 'reproduce_paper.md',
    ROOT / 'configs/train/polaris_main.yaml',
    ROOT / 'configs/train/polaris_smoke_test.yaml',
    ROOT / 'reward/components/engine.py',
    ROOT / 'reward/components/story_quality.py',
    ROOT / 'prompts/story_quality_seq.txt',
    ROOT / 'prompts/story_quality_seq.json',
    ROOT / 'data_schema/synthetic/smoke_train.parquet',
    ROOT / 'data_schema/synthetic/smoke_val.parquet',
    ROOT / 'verl/verl/trainer/ppo/core_algos.py',
    ROOT / 'verl/verl/trainer/main_ppo.py',
    ROOT / 'verl/verl/trainer/ppo/ray_trainer.py',
]

COMPILE_TARGETS = [
    ROOT / 'reward/components/engine.py',
    ROOT / 'reward/components/story_quality.py',
    ROOT / 'scripts/export_sft_checkpoint.py',
    ROOT / 'scripts/verify_install.py',
    ROOT / 'audit/memorization_gold_thinking_audit.py',
    ROOT / 'verl/verl/trainer/ppo/core_algos.py',
    ROOT / 'verl/verl/trainer/main_ppo.py',
    ROOT / 'verl/verl/trainer/ppo/ray_trainer.py',
]


def check_required_files() -> None:
    missing = [str(p) for p in REQUIRED_FILES if not p.exists()]
    if missing:
        raise RuntimeError('Missing required files:\n' + '\n'.join(missing))


def check_compilation() -> None:
    for path in COMPILE_TARGETS:
        py_compile.compile(str(path), doraise=True)


def check_configs() -> None:
    main_cfg = OmegaConf.load(ROOT / 'configs/train/polaris_main.yaml')
    smoke_cfg = OmegaConf.load(ROOT / 'configs/train/polaris_smoke_test.yaml')
    assert main_cfg.trainer.experiment_name == 'polaris_main'
    assert smoke_cfg.trainer.experiment_name == 'polaris_smoke_test'
    assert smoke_cfg.reward_model.reward_kwargs.reward_components.enable_story_quality is False


def check_synthetic_parquet() -> None:
    for name, expected_rows in [('smoke_train.parquet', 8), ('smoke_val.parquet', 2)]:
        table = pq.read_table(ROOT / 'data_schema/synthetic' / name)
        assert table.num_rows == expected_rows, (name, table.num_rows)
        schema = table.schema
        prompt_type = str(schema.field('prompt').type)
        story_type = str(schema.field('story').type)
        reasoning_type = str(schema.field('ground_truth_reasoning').type)
        assert prompt_type.startswith('list<element: struct<content: string, role: string>>'), prompt_type
        assert story_type == 'string', story_type
        assert reasoning_type == 'string', reasoning_type


def main() -> None:
    check_required_files()
    check_compilation()
    check_configs()
    check_synthetic_parquet()
    print('POLARIS install verification passed.')


if __name__ == '__main__':
    main()
