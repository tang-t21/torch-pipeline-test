from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.pipelining import SplitPoint

from .config import BenchmarkConfig


def get_stage_layer_bounds(num_layers: int, total_stages: int, stage_index: int) -> tuple[int, int]:
    layers_per_stage = num_layers // total_stages
    start = stage_index * layers_per_stage
    end = start + layers_per_stage
    return start, end


def get_stage_indices(
    rank: int,
    world_size: int,
    stages_per_rank: int,
    style: str = "contiguous",
) -> list[int]:
    if style == "contiguous":
        return [rank * stages_per_rank + offset for offset in range(stages_per_rank)]
    if style == "loop":
        return [rank + offset * world_size for offset in range(stages_per_rank)]
    raise ValueError(f"Unsupported stage mapping style '{style}'.")


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(attended)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: int) -> None:
        super().__init__()
        inner_size = hidden_size * mlp_ratio
        self.fc1 = nn.Linear(hidden_size, inner_size)
        self.fc2 = nn.Linear(inner_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states), approximate="tanh"))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, num_heads)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, mlp_ratio)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.ln1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.ln2(hidden_states))
        return hidden_states


class SyntheticGPT(nn.Module):
    def __init__(
        self,
        config: BenchmarkConfig,
        layer_ids: Sequence[int] | None = None,
        include_embeddings: bool = True,
        include_output: bool = True,
    ) -> None:
        super().__init__()
        if layer_ids is None:
            layer_ids = list(range(config.num_layers))

        self.hidden_size = config.hidden_size
        self.seq_len = config.seq_len
        self.include_embeddings = include_embeddings
        self.include_output = include_output

        self.tok_embeddings = (
            nn.Embedding(config.vocab_size, config.hidden_size) if include_embeddings else None
        )
        self.pos_embeddings = (
            nn.Embedding(config.seq_len, config.hidden_size) if include_embeddings else None
        )
        self.layers = nn.ModuleDict(
            {
                str(layer_id): TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                )
                for layer_id in layer_ids
            }
        )
        self.norm = nn.LayerNorm(config.hidden_size) if include_output else None
        self.output = nn.Linear(config.hidden_size, config.vocab_size) if include_output else None
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.tok_embeddings is not None and self.pos_embeddings is not None:
            batch_size, seq_len = inputs.shape
            position_ids = torch.arange(seq_len, device=inputs.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
            hidden_states = self.tok_embeddings(inputs) + self.pos_embeddings(position_ids)
        else:
            hidden_states = inputs

        for layer in self.layers.values():
            hidden_states = layer(hidden_states)

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        if self.output is not None:
            hidden_states = self.output(hidden_states).clone()
        return hidden_states


def build_full_model(config: BenchmarkConfig) -> SyntheticGPT:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        return SyntheticGPT(config)


def build_manual_stage_model(
    config: BenchmarkConfig,
    stage_index: int,
    total_stages: int,
) -> SyntheticGPT:
    full_model = build_full_model(config)
    start, end = get_stage_layer_bounds(config.num_layers, total_stages, stage_index)
    stage_model = SyntheticGPT(
        config,
        layer_ids=range(start, end),
        include_embeddings=stage_index == 0,
        include_output=stage_index == total_stages - 1,
    )
    stage_keys = set(stage_model.state_dict().keys())
    stage_state = {
        key: value
        for key, value in full_model.state_dict().items()
        if key in stage_keys
    }
    if stage_keys != set(stage_state.keys()):
        missing = sorted(stage_keys - set(stage_state.keys()))
        raise RuntimeError(f"Manual stage {stage_index} is missing parameters: {missing}")
    stage_model.load_state_dict(stage_state, strict=True)
    return stage_model


def build_split_spec(config: BenchmarkConfig, total_stages: int) -> dict[str, SplitPoint]:
    split_spec: dict[str, SplitPoint] = {}
    for stage_index in range(1, total_stages):
        start, _ = get_stage_layer_bounds(config.num_layers, total_stages, stage_index)
        split_spec[f"layers.{start}"] = SplitPoint.BEGINNING
    return split_spec
