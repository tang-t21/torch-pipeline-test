from __future__ import annotations

import pytest

from llm_pipeline_bench.config import BenchmarkConfig, parse_stage_layout


def test_parse_stage_layout() -> None:
    assert parse_stage_layout("8x2") == (8, 2)


def test_invalid_single_stage_layout_rejected() -> None:
    config = BenchmarkConfig(schedule="gpipe", stages_per_rank=2)
    with pytest.raises(ValueError):
        config.validate(world_size=8)


def test_invalid_layer_count_rejected() -> None:
    config = BenchmarkConfig(schedule="gpipe", num_layers=10, stages_per_rank=1)
    with pytest.raises(ValueError):
        config.validate(world_size=8)
