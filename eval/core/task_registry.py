# core/task_registry.py
from __future__ import annotations
from typing import Dict, Type

from core.task_base import Task
from tasks.story_quality import StoryQualityTask
from tasks.eq_bench_creative import EqBenchCreativeTask
from tasks.eq_bench_longform import EqBenchLongformTask
from tasks.writingbench import WritingBenchTask
from tasks.longbench_write import LongBenchWriteTask
from tasks.story_elo import StoryEloTask


_TASKS: Dict[str, Task] = {}


def _register(task_cls: Type[Task]) -> None:
    task = task_cls()
    if task.name in _TASKS:
        raise ValueError(f"Duplicate task name: {task.name}")
    _TASKS[task.name] = task


# Public POLARIS eval tasks (the benchmarks reported in the paper).
_register(StoryQualityTask)       # Story Quality rubric (ID)
_register(EqBenchLongformTask)    # EQ-Bench LongForm (ID)
_register(EqBenchCreativeTask)    # EQ-Bench Creative (OOD)
_register(WritingBenchTask)       # WritingBench D4 (OOD)
_register(LongBenchWriteTask)     # LongBench-Write (OOD)
_register(StoryEloTask)           # pairwise Elo (EQ-Bench Creative + ID subset)


def get_task(name: str) -> Task:
    if name not in _TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {list(_TASKS.keys())}")
    return _TASKS[name]
