from __future__ import annotations

import torch

from llm_pipeline_bench.config import BenchmarkConfig
from llm_pipeline_bench.model import (
    build_full_model,
    build_manual_stage_model,
    build_split_spec,
    get_stage_indices,
)


def test_manual_two_stage_split_matches_full_model() -> None:
    config = BenchmarkConfig(
        num_layers=4,
        hidden_size=32,
        num_heads=4,
        mlp_ratio=2,
        vocab_size=128,
        seq_len=16,
        global_batch_size=2,
        n_microbatches=1,
        steps=2,
        warmup_steps=1,
    )
    full_model = build_full_model(config)
    stage0 = build_manual_stage_model(config, stage_index=0, total_stages=2)
    stage1 = build_manual_stage_model(config, stage_index=1, total_stages=2)

    input_ids = torch.randint(0, config.vocab_size, (2, config.seq_len))
    full_logits = full_model(input_ids)
    split_logits = stage1(stage0(input_ids))
    torch.testing.assert_close(split_logits, full_logits)


def test_split_spec_contains_every_boundary() -> None:
    config = BenchmarkConfig(num_layers=8, steps=2, warmup_steps=1)
    split_spec = build_split_spec(config, total_stages=4)
    assert set(split_spec.keys()) == {"layers.2", "layers.4", "layers.6"}


def test_loop_stage_mapping_matches_multi_stage_runtime() -> None:
    assert get_stage_indices(rank=0, world_size=2, stages_per_rank=2, style="loop") == [0, 2]
    assert get_stage_indices(rank=1, world_size=2, stages_per_rank=2, style="loop") == [1, 3]
