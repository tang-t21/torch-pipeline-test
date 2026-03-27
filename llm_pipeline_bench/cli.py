from __future__ import annotations

import argparse

from .config import ALL_DEVICES, ALL_PARTITIONS, ALL_PRECISIONS, ALL_SCHEDULES, BenchmarkConfig


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--schedule", choices=sorted(ALL_SCHEDULES), default="gpipe")
    parser.add_argument("--partition", choices=sorted(ALL_PARTITIONS), default="manual")
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--n-microbatches", type=int, default=4)
    parser.add_argument("--stages-per-rank", type=int, default=1)
    parser.add_argument(
        "--stage-layout",
        default="",
        help="Optional '<world_size>x<stages_per_rank>' override, for example '8x2'.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--precision", choices=sorted(ALL_PRECISIONS), default="float32")
    parser.add_argument("--device", choices=sorted(ALL_DEVICES), default="auto")
    parser.add_argument(
        "--backend",
        default="",
        help="Optional process-group backend override. Defaults to 'nccl' on CUDA and 'gloo' on CPU.",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace-dir", default="traces")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-name", default="")
    return parser


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        schedule=args.schedule,
        partition=args.partition,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        global_batch_size=args.global_batch_size,
        n_microbatches=args.n_microbatches,
        stages_per_rank=args.stages_per_rank,
        stage_layout=args.stage_layout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        precision=args.precision,
        device=args.device,
        backend=args.backend,
        profile=args.profile,
        trace_dir=args.trace_dir,
        results_dir=args.results_dir,
        run_name=args.run_name,
    )
